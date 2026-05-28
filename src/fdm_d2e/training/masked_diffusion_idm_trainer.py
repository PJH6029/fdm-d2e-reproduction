from __future__ import annotations

import glob
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator, write_paper_idm_metrics
from fdm_d2e.io_utils import ensure_dir, write_json
from fdm_d2e.training.masked_diffusion_idm import (
    FDM1_ACTION_MASK,
    FDM1_ACTION_NOOP,
    FDM1_MOUSE_AXIS_BINS,
    FDM1_MOUSE_AXIS_ZERO_INDEX,
    canonical_action_slot_record,
    canonical_fdm1_action_tokens,
    corrupt_action_slots,
    d2e_metric_tokens_from_fdm1_tokens,
    fdm1_mouse_axis_class,
    fdm1_mouse_axis_token_from_class,
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
    """Flatten numeric frame/window summaries into a deterministic feature list.

    The masked-diffusion IDM is meant to be conditioned on video/window tokens.
    Earlier prefix probes only consumed shallow lists such as ``frame.features``;
    luma-window rows materialized for G005 store frame windows as nested
    ``list[list[float]]`` structures plus small numeric masks.  Flatten nested
    numeric containers here so recipe-aligned probes can use richer video-window
    evidence without adding a separate supervised/action-history feature path.
    """

    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return [0.0]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_float_list(item))
        return out
    if isinstance(value, dict):
        out: list[float] = []
        for key in sorted(value):
            out.extend(_float_list(value.get(key)))
        return out
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

_FACTOR_BUTTON_PREFIXES = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")


def _aggregate_mouse_delta(row: dict[str, Any]) -> tuple[float, float]:
    dx = 0.0
    dy = 0.0
    saw_event = False
    for event in row.get("events", []) or []:
        if not isinstance(event, dict) or event.get("type") != "mouse_move":
            continue
        saw_event = True
        dx += float(event.get("dx", 0) or 0)
        dy += float(event.get("dy", 0) or 0)
    if saw_event:
        return dx, dy
    for token in row.get("ground_truth_tokens", []) or []:
        value = None
        try:
            from fdm_d2e.tokenization.actions import token_to_delta_class

            value = token_to_delta_class(str(token))
        except Exception:
            value = None
        if value is None:
            continue
        if str(token).startswith("MOUSE_DX_"):
            dx += float(value)
        elif str(token).startswith("MOUSE_DY_"):
            dy += float(value)
    return dx, dy


def _factorized_token_vocab(
    rows: Sequence[dict[str, Any]],
    *,
    prefixes: tuple[str, ...],
    max_tokens: int,
    min_count: int,
) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for token in canonical_fdm1_action_tokens(row, include_noop=False):
            if token.startswith(prefixes):
                counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, count in ranked if count >= int(min_count)][: max(1, int(max_tokens))]


def _factorized_targets(
    row: dict[str, Any],
    *,
    key_vocab: Sequence[str],
    button_vocab: Sequence[str],
) -> dict[str, Any]:
    width, height = _screen_size(row)
    dx, dy = _aggregate_mouse_delta(row)
    tokens = set(canonical_fdm1_action_tokens(row, include_noop=False))
    return {
        "mouse_x_class": fdm1_mouse_axis_class(dx, screen_extent=width),
        "mouse_y_class": fdm1_mouse_axis_class(dy, screen_extent=height),
        "key_labels": [1.0 if token in tokens else 0.0 for token in key_vocab],
        "button_labels": [1.0 if token in tokens else 0.0 for token in button_vocab],
        "button_class": next((idx + 1 for idx, token in enumerate(button_vocab) if token in tokens), 0),
    }


class _FactorizedMaskedDiffusionDataset:
    """Typed masked action-token planes for the FDM-1-shaped IDM objective."""

    def __init__(self, rows: Sequence[dict[str, Any]], *, config: dict[str, Any], key_vocab: Sequence[str], button_vocab: Sequence[str]) -> None:
        torch = require_torch()
        self.torch = torch
        self.rows = list(rows)
        self.config = dict(config)
        self.key_vocab = list(key_vocab)
        self.button_vocab = list(button_vocab)
        self.feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
        self.feature_dim = int(config.get("video_feature_dim", 64))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        row = self.rows[idx]
        target = _factorized_targets(row, key_vocab=self.key_vocab, button_vocab=self.button_vocab)
        features = video_feature_vector(row, feature_paths=self.feature_paths, dim=self.feature_dim)
        return (
            self.torch.tensor(features, dtype=self.torch.float32),
            self.torch.tensor(int(target["mouse_x_class"]), dtype=self.torch.long),
            self.torch.tensor(int(target["mouse_y_class"]), dtype=self.torch.long),
            self.torch.tensor(target["key_labels"], dtype=self.torch.float32),
            self.torch.tensor(target["button_labels"], dtype=self.torch.float32),
            self.torch.tensor(int(target["button_class"]), dtype=self.torch.long),
        )


def _build_factorized_model(torch: Any, *, video_dim: int, key_count: int, button_count: int, config: dict[str, Any]) -> Any:
    nn = torch.nn
    hidden_dim = int(config.get("hidden_dim", 256))
    layers = int(config.get("transformer_layers", 4))
    heads = int(config.get("transformer_heads", 4))
    dropout = float(config.get("dropout", 0.1))
    plane_count = 5  # video + masked mouse-x, mouse-y, key-set, button-set planes
    video_encoder_arch = str(config.get("video_encoder_arch", "flat_mlp")).lower()
    luma_window_frames = int(config.get("luma_window_frames", 5))
    luma_window_size = int(config.get("luma_window_size", 16))
    luma_window_dim = max(0, luma_window_frames * luma_window_size * luma_window_size)

    class CompactLumaWindowEncoder(nn.Module):
        """Small video encoder for compact D2E luma-window bootstrap tokens."""

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
            batch = video_features.shape[0]
            luma = video_features[:, : self.luma_dim]
            expected = self.frames * self.size * self.size
            if self.luma_dim < expected:
                pad = torch.zeros((batch, expected - self.luma_dim), device=video_features.device, dtype=video_features.dtype)
                luma = torch.cat([luma, pad], dim=1)
            luma = luma[:, :expected].reshape(batch, 1, self.frames, self.size, self.size)
            parts = [self.conv(luma)]
            if self.aux_proj is not None:
                parts.append(self.aux_proj(video_features[:, self.luma_dim :]))
            return self.out(torch.cat(parts, dim=1))

    class FactorizedMaskedDiffusionIDM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if video_encoder_arch in {"compact_luma_window_cnn", "luma_window_cnn", "video_luma_cnn"}:
                self.video_proj = CompactLumaWindowEncoder()
            else:
                self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            self.mask_embed = nn.Parameter(torch.randn(plane_count, hidden_dim) * 0.02)
            self.plane_embed = nn.Embedding(plane_count, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.mouse_x_head = nn.Linear(hidden_dim, FDM1_MOUSE_AXIS_BINS)
            self.mouse_y_head = nn.Linear(hidden_dim, FDM1_MOUSE_AXIS_BINS)
            self.key_head = nn.Linear(hidden_dim, key_count) if key_count else None
            self.button_head = nn.Linear(hidden_dim, button_count) if button_count else None
            self.button_class_head = (
                nn.Linear(hidden_dim, button_count + 1)
                if button_count and bool(config.get("button_transition_softmax", False))
                else None
            )
            self.button_event_head = (
                nn.Linear(hidden_dim, 1)
                if button_count and bool(config.get("button_event_auxiliary", False))
                else None
            )

        def forward(self, video_features: Any) -> dict[str, Any]:
            batch = video_features.shape[0]
            positions = torch.arange(plane_count, device=video_features.device)
            tokens = self.mask_embed.unsqueeze(0).expand(batch, plane_count, -1) + self.plane_embed(positions).unsqueeze(0)
            tokens[:, 0, :] = self.video_proj(video_features) + self.plane_embed(positions[0]).unsqueeze(0)
            encoded = self.encoder(tokens)
            out: dict[str, Any] = {
                "mouse_x": self.mouse_x_head(encoded[:, 1, :]),
                "mouse_y": self.mouse_y_head(encoded[:, 2, :]),
                "key": None if self.key_head is None else self.key_head(encoded[:, 3, :]),
                "button": None if self.button_head is None else self.button_head(encoded[:, 4, :]),
                "button_class": None if self.button_class_head is None else self.button_class_head(encoded[:, 4, :]),
                "button_event": None if self.button_event_head is None else self.button_event_head(encoded[:, 4, :]).squeeze(-1),
            }
            return out

    return FactorizedMaskedDiffusionIDM()


def _positive_weight(torch: Any, rows: Sequence[dict[str, Any]], *, vocab: Sequence[str], prefix: str, cap: float) -> Any:
    if not vocab:
        return None
    positives = [0 for _ in vocab]
    for row in rows:
        tokens = set(canonical_fdm1_action_tokens(row, include_noop=False))
        for idx, token in enumerate(vocab):
            positives[idx] += int(token in tokens)
    total = max(1, len(rows))
    weights = []
    for count in positives:
        if count <= 0:
            weights.append(float(cap))
        else:
            weights.append(min(float(cap), max(1.0, (total - count) / float(count))))
    return torch.tensor(weights, dtype=torch.float32)


def _binary_positive_weight(torch: Any, positives: int, total: int, *, cap: float) -> Any:
    positives = max(0, int(positives))
    total = max(1, int(total))
    if positives <= 0:
        weight = float(cap)
    else:
        weight = min(float(cap), max(1.0, (total - positives) / float(positives)))
    return torch.tensor(weight, dtype=torch.float32)


def _button_class_weight(torch: Any, rows: Sequence[dict[str, Any]], *, button_vocab: Sequence[str], config: dict[str, Any]) -> Any:
    counts = [0 for _ in range(len(button_vocab) + 1)]
    for row in rows:
        cls = int(_factorized_targets(row, key_vocab=[], button_vocab=button_vocab)["button_class"])
        counts[max(0, min(cls, len(counts) - 1))] += 1
    total = max(1, sum(counts))
    cap = float(config.get("button_class_pos_weight_cap", config.get("button_pos_weight_cap", 100.0)))
    no_button_weight = float(config.get("button_class_no_button_weight", 1.0))
    weights = [max(0.0, no_button_weight)]
    for count in counts[1:]:
        if count <= 0:
            weights.append(cap)
        else:
            weights.append(min(cap, max(1.0, (total - count) / float(count))))
    return torch.tensor(weights, dtype=torch.float32)


def _button_class_conditional_prior_offsets(
    rows: Sequence[dict[str, Any]],
    *,
    button_vocab: Sequence[str],
    config: dict[str, Any],
) -> list[float]:
    """Return train-label prior offsets for button-token conditional logits.

    The relaxed-budget diagnostic showed a recipe-shaped button-class head
    collapsing to the most frequent mouse-button token.  FDM-1's public IDM
    still predicts discrete action tokens, so the leakage-safe correction is to
    keep the event/no-event probability from the model while redistributing the
    *conditional* button-token mass with train-only token priors.  Target labels
    are not used here.
    """

    if not bool(config.get("button_class_conditional_prior_correction", False)):
        return []
    if not button_vocab:
        return []
    counts = [0 for _ in button_vocab]
    for row in rows:
        tokens = set(canonical_fdm1_action_tokens(row, include_noop=False))
        for idx, token in enumerate(button_vocab):
            counts[idx] += int(token in tokens)
    smoothing = max(0.0, float(config.get("button_class_prior_smoothing", 1.0)))
    total = sum(counts) + smoothing * len(counts)
    if total <= 0.0:
        return [0.0 for _ in button_vocab]
    alpha = float(config.get("button_class_conditional_prior_alpha", 1.0))
    offsets = [-alpha * math.log(max(1e-12, (count + smoothing) / total)) for count in counts]
    mean_offset = sum(offsets) / max(1, len(offsets))
    return [float(value - mean_offset) for value in offsets]


def _multiclass_focal_cross_entropy(logits: Any, targets: Any, *, weight: Any = None, gamma: float = 2.0) -> Any:
    torch = require_torch()
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)).clamp(min=1e-6, max=1.0)
    focal = (1.0 - pt).pow(float(gamma)) * ce
    return focal.mean()


def _button_probabilities_from_output(out: dict[str, Any], torch: Any, *, config: dict[str, Any]) -> tuple[list[float], float | None]:
    """Return button-token probabilities and an event probability for inference.

    The optional transition-softmax head represents the mouse-button plane as a
    single discrete action token plus a no-button class.  That is closer to the
    public FDM-1 action-token view than independent BCE over rare down/up
    events, while still preserving the typed masked-diffusion IDM scaffold.
    """

    class_event_prob: float | None = None
    class_button_probs: list[float] = []
    if out.get("button_class") is not None:
        class_logits = out["button_class"][0]
        class_probs = torch.softmax(class_logits, dim=-1)
        class_probs_list = class_probs.detach().cpu().tolist()
        if class_probs_list:
            class_event_prob = max(0.0, 1.0 - float(class_probs_list[0]))
            offsets = config.get("button_class_conditional_logit_offsets")
            if (
                bool(config.get("button_class_conditional_prior_correction", False))
                and isinstance(offsets, list)
                and len(offsets) == max(0, int(class_logits.numel()) - 1)
                and class_event_prob > 0.0
            ):
                offset_tensor = torch.tensor(offsets, dtype=class_logits.dtype, device=class_logits.device)
                conditional = torch.softmax(class_logits[1:] + offset_tensor, dim=-1).detach().cpu().tolist()
                class_button_probs = [float(class_event_prob) * float(value) for value in conditional]
            else:
                class_button_probs = [float(value) for value in class_probs_list[1:]]

    bce_button_probs: list[float] = []
    if out.get("button") is not None:
        bce_button_probs = [float(value) for value in torch.sigmoid(out["button"][0]).detach().cpu().tolist()]

    prob_source = str(config.get("button_probability_source", "button_class" if class_button_probs else "bce")).lower()
    if prob_source == "button_class" and class_button_probs:
        button_probs = class_button_probs
    elif prob_source == "max" and class_button_probs and bce_button_probs:
        button_probs = [max(float(a), float(b)) for a, b in zip(class_button_probs, bce_button_probs)]
    else:
        button_probs = bce_button_probs or class_button_probs

    aux_event_prob = float(torch.sigmoid(out["button_event"][0]).detach().cpu()) if out.get("button_event") is not None else None
    event_source = str(config.get("button_event_probability_source", "button_class" if class_event_prob is not None else "auxiliary")).lower()
    if event_source == "button_class" and class_event_prob is not None:
        event_prob = class_event_prob
    elif event_source == "max" and class_event_prob is not None and aux_event_prob is not None:
        event_prob = max(class_event_prob, aux_event_prob)
    elif event_source == "auxiliary" and aux_event_prob is not None:
        event_prob = aux_event_prob
    else:
        event_prob = aux_event_prob if aux_event_prob is not None else class_event_prob
    return button_probs, event_prob


def _predict_factorized_tokens(model: Any, torch: Any, row: dict[str, Any], *, config: dict[str, Any], key_vocab: Sequence[str], button_vocab: Sequence[str], device: Any) -> list[str]:
    feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = int(config.get("video_feature_dim", 64))
    features = torch.tensor([video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim)], dtype=torch.float32, device=device)
    key_threshold = float(config.get("key_threshold", 0.5))
    button_threshold = float(config.get("button_threshold", 0.5))
    button_event_threshold = float(config.get("button_event_threshold", 1.1))
    key_token_thresholds = config.get("key_token_thresholds") if isinstance(config.get("key_token_thresholds"), dict) else {}
    button_token_thresholds = config.get("button_token_thresholds") if isinstance(config.get("button_token_thresholds"), dict) else {}
    max_keys = int(config.get("max_predicted_keys", 4))
    max_buttons = int(config.get("max_predicted_buttons", 2))
    button_event_force_topk = int(config.get("button_event_force_topk", 1))
    button_event_min_token_probability = float(config.get("button_event_min_token_probability", 0.0))
    button_event_budget_score_threshold = config.get("button_event_budget_score_threshold")
    button_event_budget_score_threshold = None if button_event_budget_score_threshold is None else float(button_event_budget_score_threshold)
    button_event_budget_applies_to_all_buttons = bool(config.get("button_event_budget_applies_to_all_buttons", False))
    button_event_budget_rank_all_scores = bool(config.get("button_event_budget_rank_all_scores", False))
    width, height = _screen_size(row)
    model.eval()
    with torch.no_grad():
        out = model(features)
        x_class = int(torch.argmax(out["mouse_x"], dim=-1).detach().cpu()[0])
        y_class = int(torch.argmax(out["mouse_y"], dim=-1).detach().cpu()[0])
        fdm1_tokens: list[str] = []
        if x_class != FDM1_MOUSE_AXIS_ZERO_INDEX:
            fdm1_tokens.append(fdm1_mouse_axis_token_from_class("x", x_class))
        if y_class != FDM1_MOUSE_AXIS_ZERO_INDEX:
            fdm1_tokens.append(fdm1_mouse_axis_token_from_class("y", y_class))
        if out.get("key") is not None and key_vocab:
            probs = torch.sigmoid(out["key"][0]).detach().cpu().tolist()
            selected = sorted(
                (
                    (float(prob), idx)
                    for idx, prob in enumerate(probs)
                    if prob >= float(key_token_thresholds.get(str(key_vocab[idx]), key_threshold))
                ),
                key=lambda item: (-item[0], item[1]),
            )[:max_keys]
            fdm1_tokens.extend(str(key_vocab[idx]) for _, idx in selected)
        if button_vocab and (out.get("button") is not None or out.get("button_class") is not None):
            probs, event_prob = _button_probabilities_from_output(out, torch, config=config)
            max_button_prob = max([float(prob) for prob in probs] or [0.0])
            event_budget_pass = (
                True
                if button_event_budget_score_threshold is None or event_prob is None
                else (float(event_prob) * max_button_prob) >= button_event_budget_score_threshold
            )
            selected = sorted(
                (
                    (float(prob), idx)
                    for idx, prob in enumerate(probs)
                    if prob >= float(button_token_thresholds.get(str(button_vocab[idx]), button_threshold))
                    and (
                        not button_event_budget_applies_to_all_buttons
                        or button_event_budget_score_threshold is None
                        or event_prob is None
                        or (float(event_prob) * float(prob)) >= button_event_budget_score_threshold
                    )
                ),
                key=lambda item: (-item[0], item[1]),
            )[:max_buttons]
            if (
                not selected
                and out.get("button_event") is not None
                and button_event_force_topk > 0
                and event_prob is not None
                and (button_event_budget_rank_all_scores or float(event_prob) >= button_event_threshold)
                and event_budget_pass
            ):
                selected = sorted(
                    (
                        (float(prob), idx)
                        for idx, prob in enumerate(probs)
                        if button_event_budget_rank_all_scores or float(prob) >= button_event_min_token_probability
                    ),
                    key=lambda item: (-item[0], item[1]),
                )[: min(max_buttons, button_event_force_topk)]
            fdm1_tokens.extend(str(button_vocab[idx]) for _, idx in selected)
    return d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height)


def _threshold_candidates(config: dict[str, Any]) -> list[float]:
    raw = config.get("threshold_candidates")
    if isinstance(raw, list) and raw:
        return sorted({max(0.0, min(1.0, float(value))) for value in raw})
    return [round(value / 20.0, 2) for value in range(1, 20)]


def _with_dynamic_threshold_candidates(
    candidates: Sequence[float],
    probabilities: Sequence[float],
    *,
    enabled: bool,
    max_dynamic: int = 64,
) -> list[float]:
    """Augment coarse calibration thresholds with probability quantiles.

    Sparse button heads often place all probabilities in a narrow band such as
    0.45..0.50.  A fixed 0.05 grid can only choose "all rows" or "no rows",
    hiding useful ranking evidence.  Quantile thresholds are still calibrated
    only on held-out calibration rows, preserving the split-safe recipe while
    letting bounded-FPR calibration select a non-saturated operating point.
    """

    base = {max(0.0, min(1.0, float(value))) for value in candidates}
    if not enabled:
        return sorted(base)
    clean = sorted(max(0.0, min(1.0, float(value))) for value in probabilities if value is not None)
    if not clean:
        return sorted(base)
    max_dynamic = max(1, int(max_dynamic))
    if len(clean) <= max_dynamic:
        dynamic = clean
    else:
        dynamic = []
        for idx in range(max_dynamic):
            q = idx / float(max_dynamic - 1) if max_dynamic > 1 else 0.5
            dynamic.append(clean[min(len(clean) - 1, int(round(q * (len(clean) - 1))))])
    # Include just-above minima to avoid degenerate "predict every row" when
    # the smallest observed probability is shared by many negatives.
    expanded = set(base)
    eps = float(1e-6)
    for value in dynamic:
        expanded.add(max(0.0, min(1.0, value)))
        expanded.add(max(0.0, min(1.0, value + eps)))
    return sorted(expanded)


def _score_key_threshold(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    key_vocab: Sequence[str],
    button_vocab: Sequence[str],
    device: Any,
    threshold: float,
) -> dict[str, Any]:
    acc = _PaperMetricAccumulator(empty_bins_as_correct=False)
    pred_key_total = 0
    for row in rows:
        pred = _predict_factorized_tokens(
            model,
            torch,
            row,
            config={**config, "key_threshold": threshold, "button_threshold": 1.1},
            key_vocab=key_vocab,
            button_vocab=button_vocab,
            device=device,
        )
        pred_key_total += sum(1 for token in pred if token.startswith("KEY_"))
        acc.update(pred, [str(token) for token in row.get("ground_truth_tokens", [])])
    metrics = acc.metrics()
    key_accuracy = metrics["paper_compatible"]["keyboard"]["key_accuracy"]
    strict = metrics["strict_local"]["keyboard"]["accuracy"]
    return {
        "threshold": threshold,
        "score": float(key_accuracy or 0.0) + 0.1 * float(strict or 0.0),
        "key_accuracy": key_accuracy,
        "strict_accuracy": strict,
        "predicted_key_tokens": pred_key_total,
    }


def _score_button_threshold(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    key_vocab: Sequence[str],
    button_vocab: Sequence[str],
    device: Any,
    threshold: float,
) -> dict[str, Any]:
    acc = _PaperMetricAccumulator(empty_bins_as_correct=False)
    pred_button_total = 0
    for row in rows:
        pred = _predict_factorized_tokens(
            model,
            torch,
            row,
            config={**config, "key_threshold": 1.1, "button_threshold": threshold, "button_event_threshold": 1.1, "button_event_force_topk": 0},
            key_vocab=key_vocab,
            button_vocab=button_vocab,
            device=device,
        )
        pred_button_total += sum(1 for token in pred if token.startswith(_FACTOR_BUTTON_PREFIXES))
        acc.update(pred, [str(token) for token in row.get("ground_truth_tokens", [])])
    metrics = acc.metrics()
    paper = metrics["paper_compatible"]["mouse_button"]
    strict = metrics["strict_local"]["mouse_button"]
    f1 = strict.get("f1") or 0.0
    fpr = strict.get("no_button_false_positive_rate")
    max_fpr = float(config.get("calibration_max_no_button_fpr", 0.10))
    fpr_penalty = 0.0 if fpr is None or fpr <= max_fpr else (fpr - max_fpr)
    return {
        "threshold": threshold,
        "score": float(f1) + 0.25 * float(paper.get("button_accuracy") or 0.0) - fpr_penalty,
        "button_accuracy": paper.get("button_accuracy"),
        "f1": strict.get("f1"),
        "precision": strict.get("precision"),
        "recall": strict.get("recall"),
        "no_button_false_positive_rate": fpr,
        "predicted_button_tokens": pred_button_total,
    }


def _collect_factorized_probability_rows(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    key_vocab: Sequence[str],
    button_vocab: Sequence[str],
    device: Any,
) -> list[dict[str, Any]]:
    feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = int(config.get("video_feature_dim", 64))
    collected: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            features = torch.tensor([video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim)], dtype=torch.float32, device=device)
            out = model(features)
            target = _factorized_targets(row, key_vocab=key_vocab, button_vocab=button_vocab)
            key_probs = torch.sigmoid(out["key"][0]).detach().cpu().tolist() if out.get("key") is not None and key_vocab else []
            button_probs, button_event_prob = _button_probabilities_from_output(out, torch, config=config) if button_vocab else ([], None)
            collected.append(
                {
                    "key_probs": [float(value) for value in key_probs],
                    "button_probs": [float(value) for value in button_probs],
                    "button_event_prob": button_event_prob,
                    "key_labels": [int(value) for value in target["key_labels"]],
                    "button_labels": [int(value) for value in target["button_labels"]],
                    "button_event_label": int(any(target["button_labels"])),
                    "button_class_label": int(target["button_class"]),
                }
            )
    return collected


def _calibrate_button_event_threshold(
    probability_rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[float],
    max_false_positive_rate: float,
    beta: float = 2.0,
    dynamic_thresholds: bool = False,
    dynamic_max_candidates: int = 64,
    min_token_probability: float = 0.0,
    min_token_candidates: Sequence[float] | None = None,
    calibrate_min_token_probability: bool = False,
    dynamic_min_token_thresholds: bool = False,
    dynamic_min_token_max_candidates: int = 32,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    beta_sq = max(0.01, float(beta) ** 2)
    event_candidates = _with_dynamic_threshold_candidates(
        candidates,
        [float(row["button_event_prob"]) for row in probability_rows if row.get("button_event_prob") is not None],
        enabled=dynamic_thresholds,
        max_dynamic=dynamic_max_candidates,
    )
    if calibrate_min_token_probability:
        token_base = min_token_candidates if min_token_candidates is not None else candidates
        token_candidates = _with_dynamic_threshold_candidates(
            token_base,
            [max([float(value) for value in row.get("button_probs", [])] or [0.0]) for row in probability_rows],
            enabled=dynamic_min_token_thresholds,
            max_dynamic=dynamic_min_token_max_candidates,
        )
    else:
        token_candidates = [max(0.0, min(1.0, float(min_token_probability)))]
    for threshold in event_candidates:
        for token_probability in token_candidates:
            tp = fp = fn = tn = 0
            for row in probability_rows:
                prob = row.get("button_event_prob")
                if prob is None:
                    continue
                truth = bool(row.get("button_event_label"))
                max_button_prob = max([float(value) for value in row.get("button_probs", [])] or [0.0])
                pred = float(prob) >= float(threshold) and max_button_prob >= float(token_probability)
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif (not pred) and truth:
                    fn += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0
            fbeta = ((1.0 + beta_sq) * tp) / (((1.0 + beta_sq) * tp) + beta_sq * fn + fp) if (((1.0 + beta_sq) * tp) + beta_sq * fn + fp) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fpr_penalty = max(0.0, fpr - float(max_false_positive_rate))
            rows.append(
                {
                    "threshold": float(threshold),
                    "min_token_probability": float(token_probability),
                    "score": float(fbeta) - fpr_penalty,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fbeta": fbeta,
                    "false_positive_rate": fpr,
                }
            )
    if not rows:
        return {"status": "skipped", "reason": "no_button_event_probs", "selected": None, "rows": []}
    feasible = [row for row in rows if float(row["false_positive_rate"]) <= float(max_false_positive_rate)]
    pool = feasible or rows
    # Recall is the target failure mode, but FPR must stay bounded for downstream
    # no-button gates.  Use F-beta first, then recall and FPR, then prefer a
    # stricter token-probability guard before lowering the event threshold.
    best = max(
        pool,
        key=lambda row: (
            row["score"],
            row.get("recall") or 0.0,
            -float(row.get("false_positive_rate") or 0.0),
            float(row.get("min_token_probability") or 0.0),
            -row["threshold"],
        ),
    )
    return {
        "status": "pass",
        "selected": float(best["threshold"]),
        "selected_min_token_probability": float(best.get("min_token_probability") or 0.0),
        "max_false_positive_rate": float(max_false_positive_rate),
        "beta": float(beta),
        "dynamic_thresholds": bool(dynamic_thresholds),
        "calibrate_min_token_probability": bool(calibrate_min_token_probability),
        "dynamic_min_token_thresholds": bool(dynamic_min_token_thresholds),
        "candidate_count": len(event_candidates),
        "min_token_candidate_count": len(token_candidates),
        "selected_row": best,
        "rows": rows,
    }


def _calibrate_per_token_thresholds(
    probability_rows: Sequence[dict[str, Any]],
    *,
    vocab: Sequence[str],
    prob_key: str,
    label_key: str,
    candidates: Sequence[float],
    max_false_positive_rate: float | None = None,
    dynamic_thresholds: bool = False,
    dynamic_max_candidates: int = 64,
) -> dict[str, Any]:
    selected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for idx, token in enumerate(vocab):
        positives = sum(int((row.get(label_key) or [])[idx]) for row in probability_rows if idx < len(row.get(label_key) or []))
        negatives = max(0, len(probability_rows) - positives)
        token_candidates = _with_dynamic_threshold_candidates(
            candidates,
            [
                float((row.get(prob_key) or [])[idx])
                for row in probability_rows
                if idx < len(row.get(prob_key) or [])
            ],
            enabled=dynamic_thresholds,
            max_dynamic=dynamic_max_candidates,
        )
        best: dict[str, Any] | None = None
        for threshold in token_candidates:
            tp = fp = fn = tn = 0
            for row in probability_rows:
                probs = row.get(prob_key) or []
                labels = row.get(label_key) or []
                if idx >= len(probs) or idx >= len(labels):
                    continue
                pred = float(probs[idx]) >= float(threshold)
                truth = bool(labels[idx])
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif (not pred) and truth:
                    fn += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            penalty = 0.0
            if max_false_positive_rate is not None and fpr > max_false_positive_rate:
                penalty = fpr - max_false_positive_rate
            # Prefer high F1, then fewer false positives, then lower threshold for recall.
            score = float(f1) - penalty
            candidate = {
                "token": str(token),
                "threshold": float(threshold),
                "score": score,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "positives": positives,
                "negatives": negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
                "candidate_count": len(token_candidates),
            }
            if best is None or (candidate["score"], -candidate["fp"], -candidate["threshold"]) > (best["score"], -best["fp"], -best["threshold"]):
                best = candidate
        if best is None:
            best = {"token": str(token), "threshold": 1.1, "score": 0.0, "tp": 0, "fp": 0, "fn": positives, "tn": negatives, "positives": positives, "negatives": negatives, "precision": None, "recall": None, "f1": 0.0, "false_positive_rate": 0.0}
        # If the model has no positive evidence for this token, disable it instead of allowing common-token overfire.
        if positives <= 0 or best.get("tp", 0) <= 0:
            best = {**best, "threshold": 1.1, "disabled_no_true_positive": True}
        selected[str(token)] = float(best["threshold"])
        rows.append(best)
    return {"selected": selected, "rows": rows}


def _button_event_label_rate(rows: Sequence[dict[str, Any]], *, button_vocab: Sequence[str]) -> dict[str, Any]:
    total = len(rows)
    positives = 0
    for row in rows:
        tokens = set(canonical_fdm1_action_tokens(row, include_noop=False))
        positives += int(any(token in tokens for token in button_vocab))
    return {
        "rows": total,
        "positives": positives,
        "rate": (positives / total) if total else 0.0,
    }


def _calibrate_button_event_budget(
    probability_rows: Sequence[dict[str, Any]],
    *,
    rate_rows: Sequence[dict[str, Any]],
    button_vocab: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Choose an unlabeled confidence budget for forced button unmasking.

    FDM-1's public IDM inference performs iterative high-confidence unmasking.
    For sparse D2E mouse-button events, an event head can be calibrated on
    train-heldout rows yet still overfire on target rows.  This budget keeps the
    recipe shape but caps forced button unmasking to a train-label event-rate
    prior, selecting only the highest-confidence unlabeled target candidates.
    Target labels are never inspected here.
    """

    label_rate = _button_event_label_rate(rate_rows, button_vocab=button_vocab)
    multiplier = float(config.get("button_event_budget_rate_multiplier", 1.0))
    cap_rate = config.get("button_event_budget_cap_rate")
    budget_rate = max(0.0, label_rate["rate"] * multiplier)
    if cap_rate is not None:
        budget_rate = min(budget_rate, max(0.0, float(cap_rate)))
    target_count = len(probability_rows)
    max_forced = int(math.ceil(target_count * budget_rate)) if budget_rate > 0.0 else 0
    if label_rate["positives"] > 0 and target_count > 0 and bool(config.get("button_event_budget_min_one", True)):
        max_forced = max(1, max_forced)

    event_threshold = float(config.get("button_event_threshold", 1.1))
    min_token_probability = float(config.get("button_event_min_token_probability", 0.0))
    rank_all_scores = bool(config.get("button_event_budget_rank_all_scores", False))
    scored: list[dict[str, Any]] = []
    for idx, row in enumerate(probability_rows):
        event_prob = row.get("button_event_prob")
        if event_prob is None:
            continue
        max_button_prob = max([float(value) for value in row.get("button_probs", [])] or [0.0])
        event_prob = float(event_prob)
        score = event_prob * max_button_prob
        passes_thresholds = rank_all_scores or (event_prob >= event_threshold and max_button_prob >= min_token_probability)
        if passes_thresholds:
            scored.append(
                {
                    "index": idx,
                    "event_prob": event_prob,
                    "max_button_prob": max_button_prob,
                    "score": score,
                }
            )
    scored.sort(key=lambda row: (-float(row["score"]), -float(row["event_prob"]), -float(row["max_button_prob"]), int(row["index"])))
    selected = scored[:max_forced] if max_forced > 0 else []
    score_threshold = float(selected[-1]["score"]) if selected else 2.0
    return {
        "schema": "button_event_confidence_budget.v1",
        "status": "pass",
        "rate_source": str(config.get("button_event_budget_rate_source", "calibration_labels")),
        "rate_source_rows": label_rate["rows"],
        "rate_source_positives": label_rate["positives"],
        "rate_source_positive_rate": label_rate["rate"],
        "rate_multiplier": multiplier,
        "cap_rate": None if cap_rate is None else float(cap_rate),
        "budget_rate": budget_rate,
        "target_rows_scored": target_count,
        "threshold_candidate_count": len(scored),
        "max_forced_events": max_forced,
        "score_threshold": score_threshold,
        "selected_preview": selected[:10],
        "claim_boundary": "Uses labeled train/calibration event-rate prior plus unlabeled target confidence scores only; target labels are not used.",
    }


def _button_event_budget_predicts_row(
    row: dict[str, Any],
    *,
    score_threshold: float,
    config: dict[str, Any],
    button_vocab: Sequence[str],
) -> bool:
    """Mirror the final mouse-button budget gate for calibration rows.

    The budget multiplier must be selected without target labels.  To make that
    selection faithful to inference, evaluate held-out train/calibration rows
    with the same direct-token and event-forced score gates used by
    ``_predict_factorized_tokens``.
    """

    event_prob_raw = row.get("button_event_prob")
    if event_prob_raw is None:
        return False
    event_prob = float(event_prob_raw)
    probs = [float(value) for value in row.get("button_probs", [])]
    if not probs:
        return False
    button_threshold = float(config.get("button_threshold", 0.5))
    button_token_thresholds = config.get("button_token_thresholds") if isinstance(config.get("button_token_thresholds"), dict) else {}
    budget_applies_to_all = bool(config.get("button_event_budget_applies_to_all_buttons", False))
    rank_all_scores = bool(config.get("button_event_budget_rank_all_scores", False))

    for idx, prob in enumerate(probs[: len(button_vocab)]):
        token = str(button_vocab[idx])
        threshold = float(button_token_thresholds.get(token, button_threshold))
        if prob >= threshold and (
            not budget_applies_to_all
            or (event_prob * prob) >= float(score_threshold)
        ):
            return True

    max_button_prob = max(probs)
    return (
        (rank_all_scores or event_prob >= float(config.get("button_event_threshold", 1.1)))
        and (rank_all_scores or max_button_prob >= float(config.get("button_event_min_token_probability", 0.0)))
        and (event_prob * max_button_prob) >= float(score_threshold)
    )


def _score_button_event_budget(
    probability_rows: Sequence[dict[str, Any]],
    *,
    score_threshold: float,
    config: dict[str, Any],
    button_vocab: Sequence[str],
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in probability_rows:
        pred = _button_event_budget_predicts_row(
            row,
            score_threshold=float(score_threshold),
            config=config,
            button_vocab=button_vocab,
        )
        truth = bool(row.get("button_event_label"))
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif (not pred) and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    beta = float(config.get("button_event_budget_calibration_beta", config.get("button_event_calibration_beta", 2.0)))
    beta_sq = max(0.01, beta * beta)
    fbeta = ((1.0 + beta_sq) * tp) / (((1.0 + beta_sq) * tp) + beta_sq * fn + fp) if (((1.0 + beta_sq) * tp) + beta_sq * fn + fp) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
        "false_positive_rate": fpr,
        "predicted_examples": tp + fp,
        "positive_examples": tp + fn,
        "negative_examples": fp + tn,
    }


def _calibrate_button_event_budget_multiplier(
    probability_rows: Sequence[dict[str, Any]],
    *,
    rate_rows: Sequence[dict[str, Any]],
    button_vocab: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    raw_candidates = config.get("button_event_budget_rate_multiplier_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return {"status": "skipped", "reason": "no_multiplier_candidates"}

    candidates = sorted({max(0.0, float(value)) for value in raw_candidates})
    max_fpr = float(config.get("button_event_budget_calibration_max_no_button_fpr", config.get("calibration_max_no_button_fpr", 0.10)))
    rows: list[dict[str, Any]] = []
    for multiplier in candidates:
        candidate_config = {**config, "button_event_budget_rate_multiplier": multiplier}
        budget = _calibrate_button_event_budget(
            probability_rows,
            rate_rows=rate_rows,
            button_vocab=button_vocab,
            config=candidate_config,
        )
        metrics = _score_button_event_budget(
            probability_rows,
            score_threshold=float(budget["score_threshold"]),
            config=candidate_config,
            button_vocab=button_vocab,
        )
        fpr_penalty = max(0.0, float(metrics["false_positive_rate"]) - max_fpr)
        score = float(metrics["fbeta"]) - fpr_penalty
        rows.append(
            {
                "multiplier": float(multiplier),
                "score": score,
                "max_false_positive_rate": max_fpr,
                "budget": {
                    key: budget.get(key)
                    for key in [
                        "rate_source_rows",
                        "rate_source_positives",
                        "rate_source_positive_rate",
                        "budget_rate",
                        "max_forced_events",
                        "score_threshold",
                        "threshold_candidate_count",
                    ]
                },
                "metrics": metrics,
            }
        )
    feasible = [row for row in rows if float(row["metrics"]["false_positive_rate"]) <= max_fpr]
    pool = feasible or rows
    selected = max(
        pool,
        key=lambda row: (
            row["score"],
            float(row["metrics"].get("recall") or 0.0),
            float(row["metrics"].get("precision") or 0.0),
            -float(row["metrics"].get("false_positive_rate") or 0.0),
            float(row["multiplier"]),
        ),
    )
    return {
        "schema": "button_event_budget_multiplier_calibration.v1",
        "status": "pass",
        "selected_multiplier": float(selected["multiplier"]),
        "max_false_positive_rate": max_fpr,
        "candidate_count": len(rows),
        "selected_row": selected,
        "rows": rows,
        "claim_boundary": "Selects the mouse-button confidence-budget multiplier from held-out train/calibration labels only; target labels are not used.",
    }


def _calibrate_factorized_thresholds(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    key_vocab: Sequence[str],
    button_vocab: Sequence[str],
    device: Any,
) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped", "reason": "no_calibration_rows"}
    candidates = _threshold_candidates(config)
    key_rows = [
        _score_key_threshold(model, torch, rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device, threshold=threshold)
        for threshold in candidates
    ]
    button_rows = [
        _score_button_threshold(model, torch, rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device, threshold=threshold)
        for threshold in candidates
    ]
    best_key = max(key_rows, key=lambda row: (row["score"], row["threshold"])) if key_rows else None
    best_button = max(button_rows, key=lambda row: (row["score"], row["threshold"])) if button_rows else None
    if best_key is not None:
        config["key_threshold"] = float(best_key["threshold"])
    if best_button is not None:
        config["button_threshold"] = float(best_button["threshold"])
    per_token_payload: dict[str, Any] = {"status": "skipped", "reason": "disabled"}
    if config.get("calibrate_per_token_thresholds", True):
        probability_rows = _collect_factorized_probability_rows(model, torch, rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device)
        dynamic_thresholds = bool(config.get("calibration_dynamic_thresholds", False))
        dynamic_max_candidates = int(config.get("calibration_dynamic_threshold_max_candidates", 64))
        key_token_payload = _calibrate_per_token_thresholds(
            probability_rows,
            vocab=key_vocab,
            prob_key="key_probs",
            label_key="key_labels",
            candidates=candidates,
            dynamic_thresholds=dynamic_thresholds,
            dynamic_max_candidates=dynamic_max_candidates,
        )
        button_token_payload = _calibrate_per_token_thresholds(
            probability_rows,
            vocab=button_vocab,
            prob_key="button_probs",
            label_key="button_labels",
            candidates=candidates,
            max_false_positive_rate=float(config.get("calibration_max_no_button_fpr", 0.10)),
            dynamic_thresholds=dynamic_thresholds,
            dynamic_max_candidates=dynamic_max_candidates,
        )
        config["key_token_thresholds"] = key_token_payload["selected"]
        config["button_token_thresholds"] = button_token_payload["selected"]
        button_event_payload = {"status": "skipped", "reason": "button_event_auxiliary_disabled"}
        if config.get("button_event_auxiliary"):
            button_event_payload = _calibrate_button_event_threshold(
                probability_rows,
                candidates=candidates,
                max_false_positive_rate=float(config.get("button_event_calibration_max_no_button_fpr", config.get("calibration_max_no_button_fpr", 0.10))),
                beta=float(config.get("button_event_calibration_beta", 2.0)),
                dynamic_thresholds=dynamic_thresholds,
                dynamic_max_candidates=dynamic_max_candidates,
                min_token_probability=float(config.get("button_event_min_token_probability", 0.0)),
                min_token_candidates=config.get("button_event_min_token_probability_candidates"),
                calibrate_min_token_probability=bool(config.get("button_event_calibrate_min_token_probability", False)),
                dynamic_min_token_thresholds=bool(config.get("button_event_min_token_dynamic_thresholds", dynamic_thresholds)),
                dynamic_min_token_max_candidates=int(config.get("button_event_min_token_dynamic_threshold_max_candidates", 32)),
            )
            if button_event_payload.get("selected") is not None:
                config["button_event_threshold"] = float(button_event_payload["selected"])
            if button_event_payload.get("selected_min_token_probability") is not None:
                config["button_event_min_token_probability"] = float(button_event_payload["selected_min_token_probability"])
        per_token_payload = {
            "status": "pass",
            "key_token_thresholds": key_token_payload,
            "button_token_thresholds": button_token_payload,
            "button_event_threshold": button_event_payload,
        }
    return {
        "schema": "factorized_threshold_calibration.v1",
        "status": "pass",
        "rows": len(rows),
        "candidates": candidates,
        "selected": {
            "key_threshold": None if best_key is None else best_key["threshold"],
            "button_threshold": None if best_button is None else best_button["threshold"],
        },
        "per_token": per_token_payload,
        "key_sweep": key_rows,
        "button_sweep": button_rows,
        "claim_boundary": "Calibration over prefix/held-out training rows only; final G005 requires split-safe calibration evidence.",
    }


def train_factorized_masked_diffusion_idm(config: dict[str, Any]) -> dict[str, Any]:
    if not torch_available():
        raise RuntimeError("torch unavailable; run `uv sync --extra train` or use the MLXP training image")
    torch = require_torch()
    start = time.time()
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_factorized_masked_diffusion_d2e"))
    train_paths = _expand_paths(config.get("train_records")) + _expand_paths(config.get("train_record_paths"))
    target_paths = _expand_paths(config.get("target_records")) + _expand_paths(config.get("target_record_paths"))
    max_train_rows = config.get("max_train_rows")
    max_target_rows = config.get("max_target_rows")
    train_rows = list(_iter_jsonl(train_paths, max_rows=int(max_train_rows) if max_train_rows is not None else None))
    target_rows = list(_iter_jsonl(target_paths, max_rows=int(max_target_rows) if max_target_rows is not None else None))
    if not train_rows:
        raise ValueError("no train rows found for factorized masked-diffusion IDM")
    if not target_rows:
        raise ValueError("no target rows found for factorized masked-diffusion IDM")

    calibration_rows: list[dict[str, Any]] = []
    fit_rows = train_rows
    calibration_fraction = float(config.get("factorized_calibration_fraction", 0.0) or 0.0)
    calibration_max_rows = int(config.get("factorized_calibration_max_rows", 2000))
    if config.get("calibrate_thresholds") and calibration_fraction > 0.0 and len(train_rows) >= 10:
        calibration_count = min(calibration_max_rows, max(1, int(len(train_rows) * calibration_fraction)))
        calibration_rows = train_rows[-calibration_count:]
        fit_rows = train_rows[:-calibration_count] or train_rows

    key_vocab = _factorized_token_vocab(
        train_rows,
        prefixes=("KEY_",),
        max_tokens=int(config.get("max_key_tokens", 128)),
        min_count=int(config.get("key_min_count", 1)),
    )
    button_vocab = _factorized_token_vocab(
        train_rows,
        prefixes=_FACTOR_BUTTON_PREFIXES,
        max_tokens=int(config.get("max_button_tokens", 8)),
        min_count=int(config.get("button_min_count", 1)),
    )
    button_class_offsets = _button_class_conditional_prior_offsets(train_rows, button_vocab=button_vocab, config=config)
    if button_class_offsets:
        config["button_class_conditional_logit_offsets"] = button_class_offsets
    feature_dim = int(config.get("video_feature_dim", 64))
    dataset = _FactorizedMaskedDiffusionDataset(fit_rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu")
    model = _build_factorized_model(torch, video_dim=feature_dim, key_count=len(key_vocab), button_count=len(button_vocab), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    key_pos_weight = _positive_weight(torch, train_rows, vocab=key_vocab, prefix="KEY_", cap=float(config.get("key_pos_weight_cap", 100.0)))
    button_pos_weight = _positive_weight(torch, train_rows, vocab=button_vocab, prefix="MOUSE_", cap=float(config.get("button_pos_weight_cap", 100.0)))
    button_class_weight = _button_class_weight(torch, train_rows, button_vocab=button_vocab, config=config) if config.get("button_transition_softmax") and button_vocab else None
    button_event_pos_weight = None
    if config.get("button_event_auxiliary") and button_vocab:
        button_event_positives = 0
        for row in train_rows:
            tokens = set(canonical_fdm1_action_tokens(row, include_noop=False))
            button_event_positives += int(any(token in tokens for token in button_vocab))
        button_event_pos_weight = _binary_positive_weight(
            torch,
            button_event_positives,
            len(train_rows),
            cap=float(config.get("button_event_pos_weight_cap", config.get("button_pos_weight_cap", 100.0))),
        )
    if key_pos_weight is not None:
        key_pos_weight = key_pos_weight.to(device)
    if button_pos_weight is not None:
        button_pos_weight = button_pos_weight.to(device)
    if button_class_weight is not None:
        button_class_weight = button_class_weight.to(device)
    if button_event_pos_weight is not None:
        button_event_pos_weight = button_event_pos_weight.to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        totals = {"loss": 0.0, "mouse_x": 0.0, "mouse_y": 0.0, "key": 0.0, "button": 0.0, "button_class": 0.0, "button_event": 0.0, "examples": 0}
        for features, mouse_x, mouse_y, key_labels, button_labels, button_class in loader:
            features = features.to(device)
            mouse_x = mouse_x.to(device)
            mouse_y = mouse_y.to(device)
            key_labels = key_labels.to(device)
            button_labels = button_labels.to(device)
            button_class = button_class.to(device)
            out = model(features)
            mouse_x_loss = torch.nn.functional.cross_entropy(out["mouse_x"], mouse_x)
            mouse_y_loss = torch.nn.functional.cross_entropy(out["mouse_y"], mouse_y)
            key_loss = torch.tensor(0.0, device=device)
            if out.get("key") is not None and key_labels.numel():
                key_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["key"], key_labels, pos_weight=key_pos_weight)
            button_loss = torch.tensor(0.0, device=device)
            if out.get("button") is not None and button_labels.numel():
                button_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["button"], button_labels, pos_weight=button_pos_weight)
            button_class_loss = torch.tensor(0.0, device=device)
            if out.get("button_class") is not None:
                button_class_loss_type = str(config.get("button_class_loss", "cross_entropy")).lower()
                if button_class_loss_type in {"focal", "focal_cross_entropy", "weighted_focal"}:
                    button_class_loss = _multiclass_focal_cross_entropy(
                        out["button_class"],
                        button_class,
                        weight=button_class_weight,
                        gamma=float(config.get("button_class_focal_gamma", config.get("focal_gamma", 2.0))),
                    )
                else:
                    button_class_loss = torch.nn.functional.cross_entropy(out["button_class"], button_class, weight=button_class_weight)
            button_event_loss = torch.tensor(0.0, device=device)
            if out.get("button_event") is not None and button_labels.numel():
                button_event_labels = (button_labels.sum(dim=1) > 0).float()
                button_event_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["button_event"],
                    button_event_labels,
                    pos_weight=button_event_pos_weight,
                )
            loss = (
                float(config.get("mouse_x_loss_weight", config.get("mouse_loss_weight", 1.0))) * mouse_x_loss
                + float(config.get("mouse_y_loss_weight", config.get("mouse_loss_weight", 1.0))) * mouse_y_loss
                + float(config.get("key_loss_weight", 1.0)) * key_loss
                + float(config.get("button_loss_weight", 1.0)) * button_loss
                + float(config.get("button_class_loss_weight", 1.0)) * button_class_loss
                + float(config.get("button_event_loss_weight", 0.0)) * button_event_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            batch = int(features.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * batch
            totals["mouse_x"] += float(mouse_x_loss.detach().cpu()) * batch
            totals["mouse_y"] += float(mouse_y_loss.detach().cpu()) * batch
            totals["key"] += float(key_loss.detach().cpu()) * batch
            totals["button"] += float(button_loss.detach().cpu()) * batch
            totals["button_class"] += float(button_class_loss.detach().cpu()) * batch
            totals["button_event"] += float(button_event_loss.detach().cpu()) * batch
            totals["examples"] += batch
        denom = max(1, int(totals.pop("examples")))
        history.append({"epoch": epoch + 1, **{key: value / denom for key, value in totals.items()}, "examples": denom})

    checkpoint_path = Path(output_dir) / "checkpoint.pt"
    torch.save(
        {
            "schema": "factorized_masked_diffusion_idm_checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "key_vocab": key_vocab,
            "button_vocab": button_vocab,
            "config": config,
            "feature_dim": feature_dim,
        },
        checkpoint_path,
    )
    threshold_calibration = {"status": "skipped", "reason": "disabled"}
    if config.get("calibrate_thresholds"):
        threshold_calibration = _calibrate_factorized_thresholds(
            model,
            torch,
            calibration_rows or fit_rows[-min(len(fit_rows), int(config.get("factorized_calibration_max_rows", 2000))) :],
            config=config,
            key_vocab=key_vocab,
            button_vocab=button_vocab,
            device=device,
        )

    button_event_budget = {"status": "skipped", "reason": "disabled"}
    if config.get("button_event_budgeted_unmasking") and config.get("button_event_auxiliary") and button_vocab:
        budget_max_rows = config.get("button_event_budget_max_target_rows")
        budget_rows = target_rows[: int(budget_max_rows)] if budget_max_rows is not None else target_rows
        budget_calibration = {"status": "skipped", "reason": "no_multiplier_candidates"}
        target_probability_rows = _collect_factorized_probability_rows(
            model,
            torch,
            budget_rows,
            config=config,
            key_vocab=key_vocab,
            button_vocab=button_vocab,
            device=device,
        )
        budget_rate_source = str(config.get("button_event_budget_rate_source", "calibration_labels"))
        if budget_rate_source == "fit_labels":
            rate_rows = fit_rows
        elif budget_rate_source == "train_labels":
            rate_rows = train_rows
        else:
            rate_rows = calibration_rows or fit_rows[-min(len(fit_rows), int(config.get("factorized_calibration_max_rows", 2000))) :]
        if isinstance(config.get("button_event_budget_rate_multiplier_candidates"), list) and config.get("button_event_budget_rate_multiplier_candidates"):
            budget_calibration_rows = calibration_rows or fit_rows[-min(len(fit_rows), int(config.get("factorized_calibration_max_rows", 2000))) :]
            calibration_probability_rows = _collect_factorized_probability_rows(
                model,
                torch,
                budget_calibration_rows,
                config=config,
                key_vocab=key_vocab,
                button_vocab=button_vocab,
                device=device,
            )
            budget_calibration = _calibrate_button_event_budget_multiplier(
                calibration_probability_rows,
                rate_rows=rate_rows,
                button_vocab=button_vocab,
                config=config,
            )
            if budget_calibration.get("status") == "pass":
                config["button_event_budget_rate_multiplier"] = float(budget_calibration["selected_multiplier"])
        button_event_budget = _calibrate_button_event_budget(
            target_probability_rows,
            rate_rows=rate_rows,
            button_vocab=button_vocab,
            config=config,
        )
        button_event_budget["multiplier_calibration"] = budget_calibration
        config["button_event_budget_score_threshold"] = float(button_event_budget["score_threshold"])
        config["button_event_budget_max_forced_events"] = int(button_event_budget["max_forced_events"])

    predictions_path = Path(output_dir) / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in target_rows:
            predicted_tokens = _predict_factorized_tokens(model, torch, row, config=config, key_vocab=key_vocab, button_vocab=button_vocab, device=device)
            handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": predicted_tokens or [FDM1_ACTION_NOOP]}, sort_keys=True) + "\n")

    metrics_path = Path(output_dir) / "paper_metrics.json"
    write_paper_idm_metrics(
        prediction_paths=[predictions_path],
        target_paths=target_paths,
        output_path=metrics_path,
        model_name=str(config.get("model_name", "factorized_masked_diffusion_idm")),
        max_rows=len(target_rows),
    )
    summary = {
        "schema": "factorized_masked_diffusion_idm_train_summary.v1",
        "status": "pass",
        "model_name": str(config.get("model_name", "factorized_masked_diffusion_idm")),
        "recipe_alignment": "public FDM-1-shaped noncausal masked-diffusion IDM with typed masked action-token planes for mouse/key/button factors.",
        "train_rows": len(train_rows),
        "target_rows": len(target_rows),
        "fit_rows": len(fit_rows),
        "calibration_rows": len(calibration_rows),
        "key_vocab_size": len(key_vocab),
        "button_vocab_size": len(button_vocab),
        "device": str(device),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "threshold_calibration": threshold_calibration,
        "button_event_budget": button_event_budget,
        "factorization": {
            "mouse_axis_bins": FDM1_MOUSE_AXIS_BINS,
            "key_vocab": key_vocab,
            "button_vocab": button_vocab,
            "key_threshold": float(config.get("key_threshold", 0.5)),
            "button_threshold": float(config.get("button_threshold", 0.5)),
            "button_transition_softmax": bool(config.get("button_transition_softmax", False)),
            "button_probability_source": str(config.get("button_probability_source", "button_class" if config.get("button_transition_softmax") else "bce")),
            "button_event_auxiliary": bool(config.get("button_event_auxiliary", False)),
            "button_event_probability_source": str(config.get("button_event_probability_source", "button_class" if config.get("button_transition_softmax") else "auxiliary")),
            "button_event_threshold": float(config.get("button_event_threshold", 1.1)),
            "button_event_force_topk": int(config.get("button_event_force_topk", 1)),
            "button_event_min_token_probability": float(config.get("button_event_min_token_probability", 0.0)),
            "button_event_budget_score_threshold": None if config.get("button_event_budget_score_threshold") is None else float(config.get("button_event_budget_score_threshold")),
            "button_event_budget_max_forced_events": None if config.get("button_event_budget_max_forced_events") is None else int(config.get("button_event_budget_max_forced_events")),
            "button_event_budget_applies_to_all_buttons": bool(config.get("button_event_budget_applies_to_all_buttons", False)),
            "button_event_budget_rank_all_scores": bool(config.get("button_event_budget_rank_all_scores", False)),
            "button_class_loss": str(config.get("button_class_loss", "cross_entropy")),
            "button_class_conditional_prior_correction": bool(config.get("button_class_conditional_prior_correction", False)),
            "button_class_conditional_prior_alpha": float(config.get("button_class_conditional_prior_alpha", 1.0)),
            "button_class_conditional_logit_offset_count": len(config.get("button_class_conditional_logit_offsets", []) if isinstance(config.get("button_class_conditional_logit_offsets"), list) else []),
            "key_token_threshold_count": len(config.get("key_token_thresholds", {}) if isinstance(config.get("key_token_thresholds"), dict) else {}),
            "button_token_threshold_count": len(config.get("button_token_thresholds", {}) if isinstance(config.get("button_token_thresholds"), dict) else {}),
        },
        "wall_clock_seconds": time.time() - start,
        "claim_boundary": "Factorized prefix trainer scaffold; not G005 completion evidence without full-corpus 4xH200 run, recipe-alignment audit, paper-target win, and split statistics.",
    }
    summary_path = Path(config.get("summary_out", Path(output_dir) / "summary.json"))
    write_json(summary_path, summary)
    write_json(Path(output_dir) / "resolved_config.json", config)
    return summary


def train_masked_diffusion_idm(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("masked_diffusion_idm")
    if isinstance(nested, dict):
        # The model configs keep recipe hyperparameters under
        # `masked_diffusion_idm` for readability.  Promote them to trainer
        # defaults while letting explicit top-level runtime keys win.
        config = {**nested, **config}
    if config.get("factorized_action_tokens") or str(config.get("trainer_mode", "")).lower() == "factorized":
        return train_factorized_masked_diffusion_idm(config)
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
    threshold_calibration = {"status": "skipped", "reason": "disabled"}
    if config.get("calibrate_thresholds"):
        threshold_calibration = _calibrate_factorized_thresholds(
            model,
            torch,
            calibration_rows or fit_rows[-min(len(fit_rows), int(config.get("factorized_calibration_max_rows", 2000))) :],
            config=config,
            key_vocab=key_vocab,
            button_vocab=button_vocab,
            device=device,
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
