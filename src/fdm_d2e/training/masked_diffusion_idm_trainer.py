from __future__ import annotations

import glob
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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


def _nested_get(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def video_feature_vector(row: dict[str, Any], *, feature_paths: Sequence[str], dim: int) -> list[float]:
    """Compact video-window feature vector for the first recipe-aligned trainer.

    This is an explicit bootstrap approximation for D2E rows that already carry
    frame/window summaries.  Promotion to full G005 should replace or initialize
    it with cached video-token features while preserving the masked-diffusion IDM
    objective and non-causal frame conditioning.
    """

    values: list[float] = []
    for path in feature_paths:
        values.extend(_float_list(_nested_get(row, path)))
    if not values:
        values = [float(row.get("bin_index", 0) or 0) / 1000.0]
    if len(values) < dim:
        values.extend([0.0] * (dim - len(values)))
    return values[:dim]


def _screen_size(row: dict[str, Any]) -> tuple[int, int]:
    for key in ("screen", "frame", "metadata"):
        value = row.get(key)
        if isinstance(value, dict):
            width = value.get("width") or value.get("screen_width")
            height = value.get("height") or value.get("screen_height")
            if width and height:
                return max(1, int(width)), max(1, int(height))
    return 854, 480


def _target_slots(row: dict[str, Any], *, max_slots: int) -> list[str]:
    record = canonical_action_slot_record(row, max_slots=max_slots)
    tokens = list(record.padded_tokens)
    return [FDM1_ACTION_NOOP if token.startswith("<FDM1_ACTION_PAD") else token for token in tokens]


def _build_vocab(rows: Sequence[dict[str, Any]], *, max_slots: int, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for token in _target_slots(row, max_slots=max_slots):
            counts[token] = counts.get(token, 0) + 1
    vocab = ["<FDM1_ACTION_PAD>", FDM1_ACTION_MASK]
    if FDM1_ACTION_NOOP not in counts:
        counts[FDM1_ACTION_NOOP] = 1
    vocab.extend(sorted(token for token, count in counts.items() if count >= int(min_count) and token not in vocab))
    return vocab


class _MaskedDiffusionDataset:  # lightweight to keep import safe without torch Dataset at module import time
    def __init__(self, rows: Sequence[dict[str, Any]], *, config: dict[str, Any], vocab: Sequence[str]) -> None:
        torch = require_torch()
        self.torch = torch
        self.rows = list(rows)
        self.vocab = list(vocab)
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
        self.feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
        self.feature_dim = int(config.get("video_feature_dim", 64))
        self.mask_probability = float(config.get("mask_probability", 0.65))
        self.random_token_probability = float(config.get("random_token_probability", 0.10))
        self.seed = int(config.get("seed", 7))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any]:
        row = self.rows[idx]
        target_tokens = _target_slots(row, max_slots=self.max_slots)
        corrupted, loss_mask = corrupt_action_slots(
            target_tokens,
            vocab=self.vocab,
            mask_probability=self.mask_probability,
            random_token_probability=self.random_token_probability,
            rng=random.Random(self.seed + idx),
        )
        unk = self.token_to_index[FDM1_ACTION_NOOP]
        corrupted_ids = [self.token_to_index.get(token, unk) for token in corrupted]
        target_ids = [self.token_to_index.get(token, unk) for token in target_tokens]
        features = video_feature_vector(row, feature_paths=self.feature_paths, dim=self.feature_dim)
        return (
            self.torch.tensor(features, dtype=self.torch.float32),
            self.torch.tensor(corrupted_ids, dtype=self.torch.long),
            self.torch.tensor(target_ids, dtype=self.torch.long),
            self.torch.tensor(loss_mask, dtype=self.torch.bool),
        )


def _build_model(torch: Any, *, video_dim: int, vocab_size: int, max_slots: int, config: dict[str, Any]) -> Any:
    nn = torch.nn
    hidden_dim = int(config.get("hidden_dim", 256))
    layers = int(config.get("transformer_layers", 4))
    heads = int(config.get("transformer_heads", 4))
    dropout = float(config.get("dropout", 0.1))

    class MaskedDiffusionIDM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            self.action_embed = nn.Embedding(vocab_size, hidden_dim)
            self.slot_embed = nn.Embedding(max_slots + 1, hidden_dim)
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

        def forward(self, video_features: Any, corrupted_slots: Any) -> Any:
            batch = video_features.shape[0]
            video_token = self.video_proj(video_features).unsqueeze(1) + self.slot_embed(
                torch.zeros(batch, dtype=torch.long, device=video_features.device)
            ).unsqueeze(1)
            slot_positions = torch.arange(1, max_slots + 1, device=video_features.device).unsqueeze(0).expand(batch, max_slots)
            action_tokens = self.action_embed(corrupted_slots) + self.slot_embed(slot_positions)
            encoded = self.encoder(torch.cat([video_token, action_tokens], dim=1))
            return self.head(encoded[:, 1:, :])

    return MaskedDiffusionIDM()


def _predict_tokens_for_row(model: Any, torch: Any, row: dict[str, Any], *, config: dict[str, Any], vocab: Sequence[str], device: Any) -> list[str]:
    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = int(config.get("video_feature_dim", 64))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    mask_index = token_to_index[FDM1_ACTION_MASK]
    tokens = [FDM1_ACTION_MASK for _ in range(max_slots)]
    masked = [True for _ in range(max_slots)]
    features = torch.tensor([video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim)], dtype=torch.float32, device=device)
    counts = iterative_unmask_counts(max_slots, steps=int(config.get("diffusion_steps", 16)))
    model.eval()
    with torch.no_grad():
        for count in counts:
            if not any(masked):
                break
            corrupted = [token_to_index.get(token, mask_index) for token in tokens]
            logits = model(features, torch.tensor([corrupted], dtype=torch.long, device=device))[0]
            probs = torch.softmax(logits, dim=-1)
            best_prob, best_id = torch.max(probs, dim=-1)
            selected = select_topk_masked([float(value) for value in best_prob.detach().cpu()], masked, k=count)
            for idx in selected:
                tokens[idx] = str(vocab[int(best_id[idx].detach().cpu())])
                masked[idx] = False
        if any(masked):
            corrupted = [token_to_index.get(token, mask_index) for token in tokens]
            logits = model(features, torch.tensor([corrupted], dtype=torch.long, device=device))[0]
            best_id = torch.argmax(logits, dim=-1)
            for idx, is_masked in enumerate(masked):
                if is_masked:
                    tokens[idx] = str(vocab[int(best_id[idx].detach().cpu())])
    width, height = _screen_size(row)
    return d2e_metric_tokens_from_fdm1_tokens(tokens, screen_width=width, screen_height=height)


def train_masked_diffusion_idm(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("masked_diffusion_idm")
    if isinstance(nested, dict):
        # The model configs keep recipe hyperparameters under
        # `masked_diffusion_idm` for readability.  Promote them to trainer
        # defaults while letting explicit top-level runtime keys win.
        config = {**nested, **config}
    if not torch_available():
        raise RuntimeError("torch unavailable; run `uv sync --extra train` or use the MLXP training image")
    torch = require_torch()
    start = time.time()
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_masked_diffusion_d2e_prefix320k"))
    train_paths = _expand_paths(config.get("train_records")) + _expand_paths(config.get("train_record_paths"))
    target_paths = _expand_paths(config.get("target_records")) + _expand_paths(config.get("target_record_paths"))
    max_train_rows = config.get("max_train_rows")
    max_target_rows = config.get("max_target_rows")
    train_rows = list(_iter_jsonl(train_paths, max_rows=int(max_train_rows) if max_train_rows is not None else None))
    target_rows = list(_iter_jsonl(target_paths, max_rows=int(max_target_rows) if max_target_rows is not None else None))
    if not train_rows:
        raise ValueError("no train rows found for masked-diffusion IDM")
    if not target_rows:
        raise ValueError("no target rows found for masked-diffusion IDM")

    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    feature_dim = int(config.get("video_feature_dim", 64))
    vocab = _build_vocab(train_rows, max_slots=max_slots, min_count=int(config.get("vocab_min_count", 1)))
    dataset = _MaskedDiffusionDataset(train_rows, config={**config, "max_slots": max_slots}, vocab=vocab)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu")
    model = _build_model(torch, video_dim=feature_dim, vocab_size=len(vocab), max_slots=max_slots, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    class_weights = torch.ones(len(vocab), dtype=torch.float32, device=device)
    for token, idx in token_to_index.items():
        if token == FDM1_ACTION_NOOP:
            class_weights[idx] = float(config.get("noop_loss_weight", 1.0))
        elif token == "<FDM1_ACTION_PAD>":
            class_weights[idx] = float(config.get("pad_loss_weight", 0.0))
        elif token.startswith("KEY_"):
            class_weights[idx] = float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            class_weights[idx] = float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("FDM1_MOUSE_DX_", "FDM1_MOUSE_DY_")):
            class_weights[idx] = float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith("SCROLL_"):
            class_weights[idx] = float(config.get("scroll_loss_weight", config.get("action_loss_weight", 1.0)))
    history: list[dict[str, Any]] = []
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        total_loss = 0.0
        total_targets = 0
        for features, corrupted_ids, target_ids, loss_mask in loader:
            features = features.to(device)
            corrupted_ids = corrupted_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)
            logits = model(features, corrupted_ids)
            if not bool(loss_mask.any()):
                continue
            loss = torch.nn.functional.cross_entropy(logits[loss_mask], target_ids[loss_mask], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            count = int(loss_mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * count
            total_targets += count
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, total_targets), "masked_targets": total_targets})

    checkpoint_path = Path(output_dir) / "checkpoint.pt"
    torch.save(
        {
            "schema": "masked_diffusion_idm_checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": config,
            "max_slots": max_slots,
            "feature_dim": feature_dim,
        },
        checkpoint_path,
    )
    predictions_path = Path(output_dir) / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in target_rows:
            predicted_tokens = _predict_tokens_for_row(model, torch, row, config={**config, "max_slots": max_slots}, vocab=vocab, device=device)
            handle.write(
                json.dumps(
                    {
                        "sequence_id": row.get("sequence_id"),
                        "predicted_tokens": predicted_tokens or [FDM1_ACTION_NOOP],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    metrics_path = Path(output_dir) / "paper_metrics.json"
    write_paper_idm_metrics(
        prediction_paths=[predictions_path],
        target_paths=target_paths,
        output_path=metrics_path,
        model_name=str(config.get("model_name", "masked_diffusion_idm")),
        max_rows=len(target_rows),
    )
    summary = {
        "schema": "masked_diffusion_idm_train_summary.v1",
        "status": "pass",
        "model_name": str(config.get("model_name", "masked_diffusion_idm")),
        "recipe_alignment": "public FDM-1-shaped noncausal masked-diffusion IDM over action tokens; bootstrap video features are explicitly approximate.",
        "train_rows": len(train_rows),
        "target_rows": len(target_rows),
        "vocab_size": len(vocab),
        "max_slots": max_slots,
        "loss_weights": {
            "noop_loss_weight": float(config.get("noop_loss_weight", 1.0)),
            "pad_loss_weight": float(config.get("pad_loss_weight", 0.0)),
            "action_loss_weight": float(config.get("action_loss_weight", 1.0)),
            "keyboard_loss_weight": float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_button_loss_weight": float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_move_loss_weight": float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0))),
        },
        "device": str(device),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "wall_clock_seconds": time.time() - start,
        "claim_boundary": "Prefix trainer scaffold; not G005 completion evidence without full-corpus 4xH200 run, recipe-alignment audit, paper-target win, and split statistics.",
    }
    summary_path = Path(config.get("summary_out", Path(output_dir) / "summary.json"))
    write_json(summary_path, summary)
    write_json(Path(output_dir) / "resolved_config.json", config)
    return summary
