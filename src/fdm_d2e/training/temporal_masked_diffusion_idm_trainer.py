from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics
from fdm_d2e.io_utils import ensure_dir, write_json
from fdm_d2e.training.masked_diffusion_idm import (
    FDM1_ACTION_MASK,
    FDM1_ACTION_NOOP,
    canonical_action_slot_record,
    corrupt_action_slots,
    d2e_metric_tokens_from_fdm1_tokens,
    iterative_unmask_counts,
    select_topk_masked,
)
from fdm_d2e.training.masked_diffusion_idm_trainer import _screen_size, video_feature_vector
from fdm_d2e.training.torch_idm import require_torch, torch_available


def _expand_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths: list[Path] = []
    for item in values:
        matches = sorted(glob.glob(str(item)))
        paths.extend(Path(match) for match in matches)
    return paths


def _iter_jsonl(paths: Sequence[Path], *, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and emitted >= max_rows:
                    return
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
                emitted += 1
                yield row


def _target_slots(row: dict[str, Any], *, max_slots: int) -> list[str]:
    record = canonical_action_slot_record(row, max_slots=max_slots)
    return [FDM1_ACTION_NOOP if token.startswith("<FDM1_ACTION_PAD") else token for token in record.padded_tokens]


def _build_vocab(rows: Sequence[dict[str, Any]], *, max_slots: int, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for token in _target_slots(row, max_slots=max_slots):
            counts[token] = counts.get(token, 0) + 1
    counts.setdefault(FDM1_ACTION_NOOP, 1)
    vocab = ["<FDM1_ACTION_PAD>", FDM1_ACTION_MASK]
    vocab.extend(sorted(token for token, count in counts.items() if count >= min_count and token not in vocab))
    return vocab


def _temporal_offsets(config: dict[str, Any]) -> list[int]:
    raw = config.get("temporal_offsets", [-2, -1, 0, 1, 2])
    offsets = [int(value) for value in raw] if isinstance(raw, list) and raw else [-2, -1, 0, 1, 2]
    if 0 not in offsets:
        offsets.append(0)
    return sorted(dict.fromkeys(offsets))


def _center_index(offsets: Sequence[int]) -> int:
    return list(offsets).index(0) if 0 in offsets else len(offsets) // 2


def _precompute_features(rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> list[list[float]]:
    feature_paths = list(config.get("video_feature_paths", ["compact_luma_window", "compact_luma_window_mask", "frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = int(config.get("video_feature_dim", 64))
    return [video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim) for row in rows]


def _precompute_target_ids(rows: Sequence[dict[str, Any]], *, max_slots: int, token_to_index: dict[str, int]) -> list[list[int]]:
    noop = token_to_index[FDM1_ACTION_NOOP]
    return [[token_to_index.get(token, noop) for token in _target_slots(row, max_slots=max_slots)] for row in rows]


class _TemporalMaskedDiffusionDataset:
    def __init__(
        self,
        *,
        features: Sequence[Sequence[float]],
        target_ids: Sequence[Sequence[int]],
        config: dict[str, Any],
        vocab: Sequence[str],
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.features = [list(row) for row in features]
        self.target_ids = [list(row) for row in target_ids]
        self.vocab = list(vocab)
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
        self.mask_probability = float(config.get("mask_probability", 0.65))
        self.random_token_probability = float(config.get("random_token_probability", 0.10))
        self.seed = int(config.get("seed", 7))
        self.offsets = _temporal_offsets(config)
        self.loss_offsets = set(int(value) for value in config.get("temporal_loss_offsets", self.offsets))

    def __len__(self) -> int:
        return len(self.features)

    def _row_index(self, idx: int, offset: int) -> int:
        return max(0, min(len(self.features) - 1, idx + offset))

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any]:
        feature_rows: list[list[float]] = []
        corrupted_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        index_to_token = {idx_: token for token, idx_ in self.token_to_index.items()}
        mask_index = self.token_to_index[FDM1_ACTION_MASK]
        for offset_position, offset in enumerate(self.offsets):
            row_index = self._row_index(idx, offset)
            feature_rows.append(self.features[row_index])
            target = list(self.target_ids[row_index])
            target_tokens = [index_to_token.get(token_id, FDM1_ACTION_NOOP) for token_id in target]
            corrupted_tokens, loss_mask = corrupt_action_slots(
                target_tokens,
                vocab=self.vocab,
                mask_probability=self.mask_probability,
                random_token_probability=self.random_token_probability,
                rng=random.Random(self.seed + idx * 1009 + offset_position),
            )
            if offset not in self.loss_offsets:
                loss_mask = [False for _ in loss_mask]
            corrupted_rows.append([self.token_to_index.get(token, mask_index) for token in corrupted_tokens])
            target_rows.append(target)
            mask_rows.append(loss_mask)
        return (
            self.torch.tensor(feature_rows, dtype=self.torch.float32),
            self.torch.tensor(corrupted_rows, dtype=self.torch.long),
            self.torch.tensor(target_rows, dtype=self.torch.long),
            self.torch.tensor(mask_rows, dtype=self.torch.bool),
        )


def _build_temporal_model(torch: Any, *, video_dim: int, vocab_size: int, max_slots: int, offsets: Sequence[int], config: dict[str, Any]) -> Any:
    nn = torch.nn
    hidden_dim = int(config.get("hidden_dim", 256))
    layers = int(config.get("transformer_layers", 4))
    heads = int(config.get("transformer_heads", 4))
    dropout = float(config.get("dropout", 0.1))
    video_encoder_arch = str(config.get("video_encoder_arch", "flat_mlp")).lower()
    luma_window_frames = int(config.get("luma_window_frames", 5))
    luma_window_size = int(config.get("luma_window_size", 16))
    luma_window_dim = max(0, luma_window_frames * luma_window_size * luma_window_size)

    class CompactLumaWindowEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(config.get("luma_encoder_channels", 32))
            pooled_hw = int(config.get("luma_encoder_pool_hw", 2))
            self.luma_dim = min(video_dim, luma_window_dim)
            self.aux_dim = max(0, video_dim - self.luma_dim)
            self.frames = max(1, luma_window_frames)
            self.size = max(1, luma_window_size)
            self.conv = nn.Sequential(
                nn.Conv3d(1, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
                nn.Conv3d(channels, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((1, pooled_hw, pooled_hw)),
                nn.Flatten(),
            )
            conv_dim = channels * 2 * pooled_hw * pooled_hw
            aux_hidden = int(config.get("luma_aux_hidden_dim", min(hidden_dim, 128)))
            self.aux_proj = nn.Sequential(nn.Linear(self.aux_dim, aux_hidden), nn.GELU()) if self.aux_dim else None
            merged_dim = conv_dim + (aux_hidden if self.aux_proj is not None else 0)
            self.out = nn.Sequential(nn.Linear(merged_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        def forward(self, video_features: Any) -> Any:
            batch = int(video_features.shape[0])
            expected = self.frames * self.size * self.size
            luma = video_features[:, : self.luma_dim]
            if self.luma_dim < expected:
                pad = torch.zeros((batch, expected - self.luma_dim), device=video_features.device, dtype=video_features.dtype)
                luma = torch.cat([luma, pad], dim=1)
            luma = luma[:, :expected].reshape(batch, 1, self.frames, self.size, self.size)
            parts = [self.conv(luma)]
            if self.aux_proj is not None:
                parts.append(self.aux_proj(video_features[:, self.luma_dim :]))
            return self.out(torch.cat(parts, dim=1))

    class TemporalMaskedDiffusionIDM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offsets = list(offsets)
            self.video_reconstruction_dim = luma_window_dim if video_encoder_arch in {"compact_luma_window_cnn", "luma_window_cnn", "video_luma_cnn"} else 0
            if self.video_reconstruction_dim > 0:
                self.video_proj = CompactLumaWindowEncoder()
            else:
                self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            self.video_reconstruction_head = (
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.video_reconstruction_dim))
                if self.video_reconstruction_dim > 0
                else None
            )
            self.action_embed = nn.Embedding(vocab_size, hidden_dim)
            self.slot_embed = nn.Embedding(max_slots, hidden_dim)
            self.offset_embed = nn.Embedding(len(offsets), hidden_dim)
            self.type_embed = nn.Embedding(2, hidden_dim)  # 0=video, 1=action
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(hidden_dim, vocab_size)

        def video_embedding(self, video_features: Any) -> Any:
            flat = video_features.reshape(-1, video_features.shape[-1])
            encoded = self.video_proj(flat)
            return encoded.reshape(video_features.shape[0], video_features.shape[1], -1)

        def reconstruct_video(self, video_features: Any) -> Any:
            if self.video_reconstruction_head is None:
                raise RuntimeError("video reconstruction head unavailable")
            flat = video_features.reshape(-1, video_features.shape[-1])
            encoded = self.video_proj(flat)
            pred = self.video_reconstruction_head(encoded)
            return pred.reshape(video_features.shape[0], video_features.shape[1], -1)

        def forward(self, video_features: Any, corrupted_ids: Any) -> Any:
            batch, window, _ = video_features.shape
            slots = corrupted_ids.shape[-1]
            device = video_features.device
            offset_positions = torch.arange(window, device=device)
            video_tokens = self.video_embedding(video_features) + self.offset_embed(offset_positions).unsqueeze(0) + self.type_embed(torch.zeros(window, dtype=torch.long, device=device)).unsqueeze(0)
            action = self.action_embed(corrupted_ids)
            slot_positions = torch.arange(slots, device=device)
            action = action + self.offset_embed(offset_positions).view(1, window, 1, -1) + self.slot_embed(slot_positions).view(1, 1, slots, -1) + self.type_embed(torch.ones((), dtype=torch.long, device=device)).view(1, 1, 1, -1)
            sequence = torch.cat([video_tokens, action.reshape(batch, window * slots, -1)], dim=1)
            encoded = self.encoder(sequence)
            action_encoded = encoded[:, window:, :].reshape(batch, window, slots, -1)
            return self.head(action_encoded)

    return TemporalMaskedDiffusionIDM()


def _masked_video_reconstruction_loss(model: Any, torch: Any, features: Any, *, config: dict[str, Any]) -> Any:
    if not hasattr(model, "reconstruct_video"):
        return torch.tensor(0.0, device=features.device)
    recon_dim = int(getattr(model, "video_reconstruction_dim", 0) or 0)
    if recon_dim <= 0:
        return torch.tensor(0.0, device=features.device)
    target = features[:, :, : min(recon_dim, features.shape[-1])]
    if target.shape[-1] < recon_dim:
        pad = torch.zeros((*target.shape[:2], recon_dim - target.shape[-1]), device=features.device, dtype=features.dtype)
        target = torch.cat([target, pad], dim=-1)
    corrupted = features.clone()
    mask_probability = float(config.get("video_encoder_mask_probability", config.get("video_encoder_pretrain_mask_probability", 0.65)))
    mask = torch.rand(target.shape, device=features.device) < mask_probability
    if not bool(mask.any()):
        mask = torch.ones_like(target, dtype=torch.bool)
    source = corrupted[:, :, : min(recon_dim, corrupted.shape[-1])]
    source_mask = mask[:, :, : source.shape[-1]]
    source[source_mask] = 0.0
    corrupted[:, :, : source.shape[-1]] = source
    pred = model.reconstruct_video(corrupted)
    if bool(config.get("video_encoder_reconstruct_masked_only", True)):
        return torch.nn.functional.mse_loss(pred[mask], target[mask])
    return torch.nn.functional.mse_loss(pred, target)


def _pretrain_video_encoder(model: Any, torch: Any, loader: Any, *, config: dict[str, Any], device: Any) -> list[dict[str, Any]]:
    epochs = int(config.get("video_encoder_pretrain_epochs", 0) or 0)
    if epochs <= 0 or not hasattr(model, "reconstruct_video") or int(getattr(model, "video_reconstruction_dim", 0) or 0) <= 0:
        return []
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("video_encoder_pretrain_lr", config.get("lr", 2e-4))),
        weight_decay=float(config.get("video_encoder_pretrain_weight_decay", config.get("weight_decay", 0.01))),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for features, _corrupted, _targets, _mask in loader:
            features = features.to(device)
            loss = _masked_video_reconstruction_loss(model, torch, features, config=config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            batch = int(features.shape[0])
            total += float(loss.detach().cpu()) * batch
            examples += batch
        history.append({"epoch": epoch + 1, "video_reconstruction_loss": total / max(1, examples), "examples": examples})
    return history


def _class_weights(torch: Any, vocab: Sequence[str], config: dict[str, Any], *, device: Any) -> Any:
    weights = torch.ones(len(vocab), dtype=torch.float32, device=device)
    for idx, token in enumerate(vocab):
        if token == FDM1_ACTION_NOOP:
            weights[idx] = float(config.get("noop_loss_weight", 1.0))
        elif token == "<FDM1_ACTION_PAD>":
            weights[idx] = float(config.get("pad_loss_weight", 0.0))
        elif token.startswith("KEY_"):
            weights[idx] = float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            weights[idx] = float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("FDM1_MOUSE_DX_", "FDM1_MOUSE_DY_")):
            weights[idx] = float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith("SCROLL_"):
            weights[idx] = float(config.get("scroll_loss_weight", config.get("action_loss_weight", 1.0)))
    return weights


def _predict_temporal_tokens_batch(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[Sequence[float]],
    *,
    start_index: int,
    all_features: Sequence[Sequence[float]],
    config: dict[str, Any],
    vocab: Sequence[str],
    device: Any,
) -> list[list[str]]:
    offsets = _temporal_offsets(config)
    center = _center_index(offsets)
    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    mask_index = token_to_index[FDM1_ACTION_MASK]
    noop_index = token_to_index.get(FDM1_ACTION_NOOP, mask_index)
    window_features: list[list[list[float]]] = []
    for local_idx, _row in enumerate(rows):
        global_idx = start_index + local_idx
        window_features.append([list(all_features[max(0, min(len(all_features) - 1, global_idx + offset))]) for offset in offsets])
    if not window_features:
        return []
    model.eval()
    feature_tensor = torch.tensor(window_features, dtype=torch.float32, device=device)
    batch = int(feature_tensor.shape[0])
    corrupted = torch.full((batch, len(offsets), max_slots), mask_index, dtype=torch.long, device=device)
    masked = torch.ones((batch, len(offsets), max_slots), dtype=torch.bool, device=device)
    counts = iterative_unmask_counts(len(offsets) * max_slots, steps=int(config.get("diffusion_steps", 16)))
    with torch.no_grad():
        for count in counts:
            if not bool(masked.any()):
                break
            logits = model(feature_tensor, corrupted)
            probs = torch.softmax(logits, dim=-1)
            best_prob, best_id = torch.max(probs, dim=-1)
            for batch_idx in range(batch):
                flat_probs = [float(value) for value in best_prob[batch_idx].reshape(-1).detach().cpu().tolist()]
                flat_masked = [bool(value) for value in masked[batch_idx].reshape(-1).detach().cpu().tolist()]
                selected = select_topk_masked(flat_probs, flat_masked, k=count)
                for flat_idx in selected:
                    off_idx = flat_idx // max_slots
                    slot_idx = flat_idx % max_slots
                    corrupted[batch_idx, off_idx, slot_idx] = best_id[batch_idx, off_idx, slot_idx]
                    masked[batch_idx, off_idx, slot_idx] = False
        if bool(masked.any()):
            logits = model(feature_tensor, corrupted)
            best_id = torch.argmax(logits, dim=-1)
            corrupted = torch.where(masked, best_id, corrupted)
    predictions: list[list[str]] = []
    for batch_idx, row in enumerate(rows):
        center_ids = [int(value) for value in corrupted[batch_idx, center, :].detach().cpu().tolist()]
        fdm1_tokens = [str(vocab[idx]) for idx in center_ids if idx != noop_index and str(vocab[idx]) not in {"<FDM1_ACTION_PAD>", FDM1_ACTION_MASK, FDM1_ACTION_NOOP}]
        width, height = _screen_size(row)
        predictions.append(d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height) or [FDM1_ACTION_NOOP])
    return predictions


def train_temporal_masked_diffusion_idm(config: dict[str, Any]) -> dict[str, Any]:
    if not torch_available():
        raise RuntimeError("torch unavailable; run `uv sync --extra train` or use the MLXP training image")
    torch = require_torch()
    start = time.time()
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_temporal_masked_diffusion_d2e"))
    train_paths = _expand_paths(config.get("train_records")) + _expand_paths(config.get("train_record_paths"))
    target_paths = _expand_paths(config.get("target_records")) + _expand_paths(config.get("target_record_paths"))
    train_rows = list(_iter_jsonl(train_paths, max_rows=int(config["max_train_rows"]) if config.get("max_train_rows") is not None else None))
    target_rows = list(_iter_jsonl(target_paths, max_rows=int(config["max_target_rows"]) if config.get("max_target_rows") is not None else None))
    if not train_rows:
        raise ValueError("no train rows found for temporal masked-diffusion IDM")
    if not target_rows:
        raise ValueError("no target rows found for temporal masked-diffusion IDM")
    offsets = _temporal_offsets(config)
    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    feature_dim = int(config.get("video_feature_dim", 64))
    vocab = _build_vocab(train_rows, max_slots=max_slots, min_count=int(config.get("vocab_min_count", 1)))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    train_features = _precompute_features(train_rows, config=config)
    target_features = _precompute_features(target_rows, config=config)
    train_target_ids = _precompute_target_ids(train_rows, max_slots=max_slots, token_to_index=token_to_index)
    dataset = _TemporalMaskedDiffusionDataset(features=train_features, target_ids=train_target_ids, config={**config, "max_slots": max_slots}, vocab=vocab)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu")
    model = _build_temporal_model(torch, video_dim=feature_dim, vocab_size=len(vocab), max_slots=max_slots, offsets=offsets, config=config).to(device)
    video_pretrain_history = _pretrain_video_encoder(model, torch, loader, config=config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    class_weights = _class_weights(torch, vocab, config, device=device)
    video_reconstruction_aux_weight = float(config.get("video_reconstruction_aux_weight", 0.0) or 0.0)
    history: list[dict[str, Any]] = []
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        total_loss = 0.0
        total_action = 0.0
        total_video = 0.0
        total_targets = 0
        for features, corrupted_ids, target_ids, loss_mask in loader:
            features = features.to(device)
            corrupted_ids = corrupted_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)
            logits = model(features, corrupted_ids)
            if bool(loss_mask.any()):
                action_loss = torch.nn.functional.cross_entropy(logits[loss_mask], target_ids[loss_mask], weight=class_weights)
            else:
                action_loss = torch.tensor(0.0, device=device)
            video_loss = _masked_video_reconstruction_loss(model, torch, features, config=config) if video_reconstruction_aux_weight > 0.0 else torch.tensor(0.0, device=device)
            loss = action_loss + video_reconstruction_aux_weight * video_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            count = int(loss_mask.sum().detach().cpu())
            batch = int(features.shape[0])
            total_loss += float(loss.detach().cpu()) * max(1, count)
            total_action += float(action_loss.detach().cpu()) * max(1, count)
            total_video += float(video_loss.detach().cpu()) * batch
            total_targets += count
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total_targets),
            "action_loss": total_action / max(1, total_targets),
            "video_reconstruction_loss": total_video / max(1, len(dataset)),
            "masked_targets": total_targets,
        })
    checkpoint_path = Path(output_dir) / "checkpoint.pt"
    torch.save({
        "schema":"temporal_masked_diffusion_idm_checkpoint.v1",
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config,
        "max_slots": max_slots,
        "feature_dim": feature_dim,
        "temporal_offsets": offsets,
    }, checkpoint_path)
    predictions_path = Path(output_dir) / "predictions.jsonl"
    prediction_batch_size = max(1, int(config.get("prediction_batch_size", config.get("batch_size", 64))))
    with predictions_path.open("w", encoding="utf-8") as handle:
        for start_idx in range(0, len(target_rows), prediction_batch_size):
            batch_rows = target_rows[start_idx : start_idx + prediction_batch_size]
            batch_predictions = _predict_temporal_tokens_batch(
                model,
                torch,
                batch_rows,
                target_features[start_idx : start_idx + prediction_batch_size],
                start_index=start_idx,
                all_features=target_features,
                config={**config, "max_slots": max_slots},
                vocab=vocab,
                device=device,
            )
            for row, predicted_tokens in zip(batch_rows, batch_predictions):
                handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": predicted_tokens}, sort_keys=True) + "\n")
    metrics_path = Path(output_dir) / "paper_metrics.json"
    write_paper_idm_metrics(
        prediction_paths=[predictions_path],
        target_paths=target_paths,
        output_path=metrics_path,
        model_name=str(config.get("model_name", "temporal_masked_diffusion_idm")),
        max_rows=len(target_rows),
    )
    summary = {
        "schema":"temporal_masked_diffusion_idm_train_summary.v1",
        "status":"pass",
        "model_name":str(config.get("model_name", "temporal_masked_diffusion_idm")),
        "recipe_alignment":"public FDM-1-shaped noncausal masked-diffusion IDM over temporal action-token sequences conditioned on all D2E frame-window tokens in the local window.",
        "train_rows":len(train_rows),
        "target_rows":len(target_rows),
        "vocab_size":len(vocab),
        "max_slots":max_slots,
        "temporal_offsets":offsets,
        "temporal_window":len(offsets),
        "video_encoder_arch":str(config.get("video_encoder_arch", "flat_mlp")),
        "video_encoder_pretrain_history":video_pretrain_history,
        "history":history,
        "loss_weights":{
            "noop_loss_weight":float(config.get("noop_loss_weight", 1.0)),
            "pad_loss_weight":float(config.get("pad_loss_weight", 0.0)),
            "action_loss_weight":float(config.get("action_loss_weight", 1.0)),
            "keyboard_loss_weight":float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_button_loss_weight":float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_move_loss_weight":float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0))),
            "video_reconstruction_aux_weight":video_reconstruction_aux_weight,
        },
        "device":str(device),
        "checkpoint_path":str(checkpoint_path),
        "predictions_path":str(predictions_path),
        "metrics_path":str(metrics_path),
        "wall_clock_seconds":time.time()-start,
        "claim_boundary":"Temporal prefix trainer scaffold; not G005 completion evidence without full-corpus 4xH200 run, recipe-alignment audit, paper/G-IDM target win, and split statistics.",
    }
    summary_path = Path(config.get("summary_out", Path(output_dir) / "summary.json"))
    write_json(summary_path, summary)
    write_json(Path(output_dir) / "resolved_config.json", config)
    return summary
