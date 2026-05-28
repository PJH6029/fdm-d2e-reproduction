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

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any, Any]:
        row = self.rows[idx]
        target = _factorized_targets(row, key_vocab=self.key_vocab, button_vocab=self.button_vocab)
        features = video_feature_vector(row, feature_paths=self.feature_paths, dim=self.feature_dim)
        return (
            self.torch.tensor(features, dtype=self.torch.float32),
            self.torch.tensor(int(target["mouse_x_class"]), dtype=self.torch.long),
            self.torch.tensor(int(target["mouse_y_class"]), dtype=self.torch.long),
            self.torch.tensor(target["key_labels"], dtype=self.torch.float32),
            self.torch.tensor(target["button_labels"], dtype=self.torch.float32),
        )


def _build_factorized_model(torch: Any, *, video_dim: int, key_count: int, button_count: int, config: dict[str, Any]) -> Any:
    nn = torch.nn
    hidden_dim = int(config.get("hidden_dim", 256))
    layers = int(config.get("transformer_layers", 4))
    heads = int(config.get("transformer_heads", 4))
    dropout = float(config.get("dropout", 0.1))
    plane_count = 5  # video + masked mouse-x, mouse-y, key-set, button-set planes

    class FactorizedMaskedDiffusionIDM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
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


def _predict_factorized_tokens(model: Any, torch: Any, row: dict[str, Any], *, config: dict[str, Any], key_vocab: Sequence[str], button_vocab: Sequence[str], device: Any) -> list[str]:
    feature_paths = list(config.get("video_feature_paths", ["frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = int(config.get("video_feature_dim", 64))
    features = torch.tensor([video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim)], dtype=torch.float32, device=device)
    key_threshold = float(config.get("key_threshold", 0.5))
    button_threshold = float(config.get("button_threshold", 0.5))
    max_keys = int(config.get("max_predicted_keys", 4))
    max_buttons = int(config.get("max_predicted_buttons", 2))
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
            selected = sorted(((float(prob), idx) for idx, prob in enumerate(probs) if prob >= key_threshold), key=lambda item: (-item[0], item[1]))[:max_keys]
            fdm1_tokens.extend(str(key_vocab[idx]) for _, idx in selected)
        if out.get("button") is not None and button_vocab:
            probs = torch.sigmoid(out["button"][0]).detach().cpu().tolist()
            selected = sorted(((float(prob), idx) for idx, prob in enumerate(probs) if prob >= button_threshold), key=lambda item: (-item[0], item[1]))[:max_buttons]
            fdm1_tokens.extend(str(button_vocab[idx]) for _, idx in selected)
    return d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height)


def _threshold_candidates(config: dict[str, Any]) -> list[float]:
    raw = config.get("threshold_candidates")
    if isinstance(raw, list) and raw:
        return sorted({max(0.0, min(1.0, float(value))) for value in raw})
    return [round(value / 20.0, 2) for value in range(1, 20)]


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
            config={**config, "key_threshold": 1.1, "button_threshold": threshold},
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
    return {
        "schema": "factorized_threshold_calibration.v1",
        "status": "pass",
        "rows": len(rows),
        "candidates": candidates,
        "selected": {
            "key_threshold": None if best_key is None else best_key["threshold"],
            "button_threshold": None if best_button is None else best_button["threshold"],
        },
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
    feature_dim = int(config.get("video_feature_dim", 64))
    dataset = _FactorizedMaskedDiffusionDataset(fit_rows, config=config, key_vocab=key_vocab, button_vocab=button_vocab)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu")
    model = _build_factorized_model(torch, video_dim=feature_dim, key_count=len(key_vocab), button_count=len(button_vocab), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    key_pos_weight = _positive_weight(torch, train_rows, vocab=key_vocab, prefix="KEY_", cap=float(config.get("key_pos_weight_cap", 100.0)))
    button_pos_weight = _positive_weight(torch, train_rows, vocab=button_vocab, prefix="MOUSE_", cap=float(config.get("button_pos_weight_cap", 100.0)))
    if key_pos_weight is not None:
        key_pos_weight = key_pos_weight.to(device)
    if button_pos_weight is not None:
        button_pos_weight = button_pos_weight.to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        totals = {"loss": 0.0, "mouse_x": 0.0, "mouse_y": 0.0, "key": 0.0, "button": 0.0, "examples": 0}
        for features, mouse_x, mouse_y, key_labels, button_labels in loader:
            features = features.to(device)
            mouse_x = mouse_x.to(device)
            mouse_y = mouse_y.to(device)
            key_labels = key_labels.to(device)
            button_labels = button_labels.to(device)
            out = model(features)
            mouse_x_loss = torch.nn.functional.cross_entropy(out["mouse_x"], mouse_x)
            mouse_y_loss = torch.nn.functional.cross_entropy(out["mouse_y"], mouse_y)
            key_loss = torch.tensor(0.0, device=device)
            if out.get("key") is not None and key_labels.numel():
                key_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["key"], key_labels, pos_weight=key_pos_weight)
            button_loss = torch.tensor(0.0, device=device)
            if out.get("button") is not None and button_labels.numel():
                button_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["button"], button_labels, pos_weight=button_pos_weight)
            loss = (
                float(config.get("mouse_x_loss_weight", config.get("mouse_loss_weight", 1.0))) * mouse_x_loss
                + float(config.get("mouse_y_loss_weight", config.get("mouse_loss_weight", 1.0))) * mouse_y_loss
                + float(config.get("key_loss_weight", 1.0)) * key_loss
                + float(config.get("button_loss_weight", 1.0)) * button_loss
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
        "factorization": {
            "mouse_axis_bins": FDM1_MOUSE_AXIS_BINS,
            "key_vocab": key_vocab,
            "button_vocab": button_vocab,
            "key_threshold": float(config.get("key_threshold", 0.5)),
            "button_threshold": float(config.get("button_threshold", 0.5)),
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
