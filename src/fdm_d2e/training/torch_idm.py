from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_jsonl, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import token_to_delta_class
from fdm_d2e.training.neural_idm import record_features, target_mouse_delta, tokens_from_delta

MOUSE_AXIS_CLASSES = ["N5", "N4", "N3", "N2", "N1", "Z0", "P1", "P2", "P3", "P4", "P5"]


def categorical_token_vocab(records: list[dict[str, Any]], *, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for row in records:
        for token in row.get("ground_truth_tokens", []):
            if token.startswith("KEY_") or (
                token.startswith("MOUSE_")
                and not token.startswith("MOUSE_DX_")
                and not token.startswith("MOUSE_DY_")
            ):
                counts[token] = counts.get(token, 0) + 1
    return sorted(token for token, count in counts.items() if count >= min_count)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Torch IDM training requires `uv sync --extra train` or the cluster training image") from exc
    return torch


class TorchUnavailableError(RuntimeError):
    pass


def _build_mlp(torch, input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float):
    if depth <= 0:
        return torch.nn.Linear(input_dim, output_dim)
    layers = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([torch.nn.Linear(dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(torch.nn.Linear(dim, output_dim))
    return torch.nn.Sequential(*layers)


def _luma_temporal_layout(input_dim: int, feature_mode: str, config: dict[str, Any]) -> dict[str, int]:
    if feature_mode != "summary_luma16_stack5_time":
        raise ValueError("model_arch=luma_temporal_conv requires feature_mode=summary_luma16_stack5_time")
    luma_size = int(config.get("visual_luma_size", 16))
    stack_frames = int(config.get("visual_stack_frames", 5))
    if luma_size <= 0 or stack_frames < 2:
        raise ValueError("visual_luma_size must be positive and visual_stack_frames must be >=2")
    if luma_size != 16 or stack_frames != 5:
        raise ValueError("summary_luma16_stack5_time requires visual_luma_size=16 and visual_stack_frames=5")
    summary_dim = 16
    temporal_dim = 12
    plane_dim = luma_size * luma_size
    visual_planes = stack_frames + (stack_frames - 1)
    visual_dim = visual_planes * plane_dim
    expected_feature_dim = summary_dim + visual_dim + temporal_dim
    if input_dim < expected_feature_dim:
        raise ValueError(
            f"input_dim {input_dim} is too small for {feature_mode} visual layout "
            f"(expected at least {expected_feature_dim})"
        )
    return {
        "summary_dim": summary_dim,
        "temporal_dim": temporal_dim,
        "luma_size": luma_size,
        "stack_frames": stack_frames,
        "visual_planes": visual_planes,
        "visual_offset": summary_dim,
        "visual_dim": visual_dim,
        "expected_feature_dim": expected_feature_dim,
        "aux_dim": summary_dim + temporal_dim + max(0, input_dim - expected_feature_dim),
    }


def _build_luma_temporal_conv_model(
    torch,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    feature_mode: str,
    config: dict[str, Any],
):
    layout = _luma_temporal_layout(input_dim, feature_mode, config)
    conv_channels = int(config.get("visual_conv_channels", 8))
    pool_hw = int(config.get("visual_conv_pool_hw", 4))
    if conv_channels <= 0 or pool_hw <= 0:
        raise ValueError("visual_conv_channels and visual_conv_pool_hw must be positive")

    class LumaTemporalConvIDM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layout = dict(layout)
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv3d(1, conv_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                torch.nn.GELU(),
                torch.nn.Conv3d(conv_channels, conv_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool3d((1, pool_hw, pool_hw)),
                torch.nn.Flatten(),
            )
            encoded_dim = conv_channels * pool_hw * pool_hw
            self.head = _build_mlp(
                torch,
                encoded_dim + int(layout["aux_dim"]),
                output_dim,
                hidden_dim,
                depth,
                dropout,
            )

        def forward(self, x):
            visual_start = int(self.layout["visual_offset"])
            visual_end = visual_start + int(self.layout["visual_dim"])
            size = int(self.layout["luma_size"])
            planes = int(self.layout["visual_planes"])
            summary = x[:, :visual_start]
            visual = x[:, visual_start:visual_end].reshape(x.shape[0], 1, planes, size, size)
            aux = torch.cat([summary, x[:, visual_end:]], dim=1)
            encoded = self.encoder(visual)
            return self.head(torch.cat([encoded, aux], dim=1))

    return LumaTemporalConvIDM()


def _build_model(
    torch,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    *,
    config: dict[str, Any] | None = None,
    feature_mode: str = "summary",
):
    config = config or {}
    model_arch = str(config.get("model_arch", "mlp"))
    if model_arch == "mlp":
        return _build_mlp(torch, input_dim, output_dim, hidden_dim, depth, dropout)
    if model_arch == "luma_temporal_conv":
        return _build_luma_temporal_conv_model(
            torch,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            feature_mode=feature_mode,
            config=config,
        )
    raise ValueError(f"unsupported model_arch: {model_arch}")


def _split_calibration_records(
    records: list[dict[str, Any]],
    *,
    fraction: float,
    min_calibration_per_recording: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically hold out the tail of each training recording for calibration.

    Calibration records come only from the training split.  The tail split keeps
    threshold tuning away from the test/heldout split while preserving temporal
    order within each recording.
    """

    if fraction <= 0 or len(records) < 2:
        return list(records), []
    fraction = min(0.95, float(fraction))
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_recording.setdefault(str(row.get("recording_id", "")), []).append(row)
    fit: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for rows in by_recording.values():
        ordered = sorted(rows, key=lambda item: int(item.get("timestamp_ns", 0)))
        if len(ordered) < 2:
            fit.extend(ordered)
            continue
        n_cal = max(int(round(len(ordered) * fraction)), int(min_calibration_per_recording))
        n_cal = min(max(1, n_cal), len(ordered) - 1)
        fit.extend(ordered[:-n_cal])
        calibration.extend(ordered[-n_cal:])
    return fit, calibration


def _category_label_matrix(records: list[dict[str, Any]], vocab: list[str]) -> list[list[int]]:
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    rows = [[0 for _ in vocab] for _ in records]
    for row_idx, row in enumerate(records):
        for token in set(row.get("ground_truth_tokens", [])):
            if token in vocab_index:
                rows[row_idx][vocab_index[token]] = 1
    return rows


def _axis_suffix_from_delta(value: float, axis_prefix: str) -> str:
    token = tokens_from_delta(value if axis_prefix == "MOUSE_DX_" else 0.0, value if axis_prefix == "MOUSE_DY_" else 0.0)
    for item in token:
        if item.startswith(axis_prefix):
            return item.removeprefix(axis_prefix)
    raise ValueError(f"failed to build axis token for {axis_prefix}")


def _axis_class_indices(records: list[dict[str, Any]], axis_classes: list[str]) -> tuple[list[int], list[int]]:
    class_index = {label: idx for idx, label in enumerate(axis_classes)}
    dx_indices: list[int] = []
    dy_indices: list[int] = []
    for row in records:
        dx, dy = target_mouse_delta(row)
        dx_indices.append(class_index[_axis_suffix_from_delta(dx, "MOUSE_DX_")])
        dy_indices.append(class_index[_axis_suffix_from_delta(dy, "MOUSE_DY_")])
    return dx_indices, dy_indices


def _axis_class_to_delta(axis_class: str) -> float:
    value = token_to_delta_class(f"MOUSE_DX_{axis_class}")
    if value is None:
        raise ValueError(f"unsupported mouse axis class: {axis_class}")
    return float(value)


def _select_threshold(
    scores: list[float],
    labels: list[int],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[float, dict[str, Any]]:
    positives = sum(1 for label in labels if label)
    if not scores or positives == 0:
        return default_threshold, {"positive_count": positives, "score": None, "precision": None, "recall": None}
    beta2 = beta * beta
    best_threshold = default_threshold
    best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -abs(default_threshold - 0.5))
    best_stats: dict[str, Any] = {}
    for threshold in grid:
        tp = fp = fn = 0
        for score, label in zip(scores, labels):
            pred = score >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif (not pred) and label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision == 0.0 and recall == 0.0:
            f_score = 0.0
        else:
            f_score = (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
        # Prefer F-score, then precision to avoid noisy rare-token flood, then
        # recall, then a conservative/higher threshold for deterministic ties.
        key = (f_score, precision, recall, threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_stats = {
                "positive_count": positives,
                "score": f_score,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best_threshold, best_stats


def _calibrated_category_thresholds_from_scores(
    score_rows: list[list[float]],
    label_rows: list[list[int]],
    vocab: list[str],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    per_token: dict[str, Any] = {}
    for idx, token in enumerate(vocab):
        scores = [float(row[idx]) for row in score_rows]
        labels = [int(row[idx]) for row in label_rows]
        threshold, stats = _select_threshold(
            scores,
            labels,
            default_threshold=default_threshold,
            grid=grid,
            beta=beta,
        )
        thresholds[token] = float(threshold)
        per_token[token] = {"threshold": float(threshold), **stats}
    diagnostics = {
        "default_threshold": default_threshold,
        "beta": beta,
        "grid": grid,
        "per_token": per_token,
    }
    return thresholds, diagnostics


def _category_group(token: str) -> str:
    if token.startswith("KEY_"):
        return "keyboard"
    if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
        return "mouse_button"
    return "other"


def _is_mouse_button_token(token: str) -> bool:
    return _category_group(token) == "mouse_button"


def _button_label(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({token for token in row.get("ground_truth_tokens", []) if _is_mouse_button_token(str(token))}))


def button_softmax_classes(records: list[dict[str, Any]], *, min_count: int = 1) -> list[tuple[str, ...]]:
    """Return deterministic exact-set mouse-button classes for a softmax head.

    Class 0 is always the empty/no-button class.  Positive classes are exact
    button-token sets observed in training.  This makes "predict no click" an
    explicit learned alternative instead of asking independent binary logits to
    abstain via a fragile post-hoc threshold.
    """

    counts: dict[tuple[str, ...], int] = {(): 0}
    for row in records:
        label = _button_label(row)
        counts[label] = counts.get(label, 0) + 1
    classes = [()]
    for label in sorted(label for label, count in counts.items() if label and count >= min_count):
        classes.append(label)
    return classes


def _button_class_counts(records: list[dict[str, Any]], classes: list[tuple[str, ...]]) -> dict[str, int]:
    class_index = {label: idx for idx, label in enumerate(classes)}
    counts = {str(idx): 0 for idx in range(len(classes))}
    for row in records:
        idx = class_index.get(_button_label(row), 0)
        counts[str(idx)] = counts.get(str(idx), 0) + 1
    return counts


def _button_target_indices(records: list[dict[str, Any]], classes: list[tuple[str, ...]]) -> list[int]:
    class_index = {label: idx for idx, label in enumerate(classes)}
    return [int(class_index.get(_button_label(row), 0)) for row in records]


def _button_class_metadata(records: list[dict[str, Any]], classes: list[tuple[str, ...]]) -> list[dict[str, Any]]:
    counts = _button_class_counts(records, classes)
    return [
        {
            "index": idx,
            "tokens": list(tokens),
            "name": "NO_BUTTON" if idx == 0 else "+".join(tokens),
            "train_count": counts.get(str(idx), 0),
        }
        for idx, tokens in enumerate(classes)
    ]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(float(value) - max_value) for value in values]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def _calibrated_button_softmax_threshold_from_scores(
    score_rows: list[list[float]],
    label_indices: list[int],
    button_classes: list[tuple[str, ...]],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[float, dict[str, Any]]:
    positives = sum(1 for label in label_indices if label != 0)
    if not score_rows or positives == 0 or len(button_classes) <= 1:
        return default_threshold, {"positive_examples": positives, "score": None}
    beta2 = beta * beta
    best_threshold = default_threshold
    best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, default_threshold)
    best_stats: dict[str, Any] = {}
    for threshold in grid:
        tp = fp = fn = predicted_positive = 0
        for scores, gold_idx in zip(score_rows, label_indices):
            if not scores:
                pred_idx = 0
            else:
                top_idx = max(range(len(scores)), key=lambda idx: scores[idx])
                pred_idx = top_idx if top_idx != 0 and float(scores[top_idx]) >= threshold else 0
            if pred_idx != 0:
                predicted_positive += 1
            if gold_idx != 0 and pred_idx == gold_idx:
                tp += 1
            elif gold_idx != 0:
                fn += 1
                if pred_idx != 0:
                    fp += 1
            elif pred_idx != 0:
                fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = (
            (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
            if precision or recall
            else 0.0
        )
        # Prefer calibrated precision before recall; false positive click spam
        # is unsafe for downstream desktop/game harness execution.
        key = (score, precision, recall, threshold)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = {
                "positive_examples": positives,
                "score": score,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "predicted_positive_examples": predicted_positive,
            }
    return best_threshold, best_stats


def _calibrated_group_thresholds_from_scores(
    score_rows: list[list[float]],
    label_rows: list[list[int]],
    vocab: list[str],
    *,
    default_threshold: float,
    grid: list[float],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {token: default_threshold for token in vocab}
    per_group: dict[str, Any] = {}
    groups = sorted({_category_group(token) for token in vocab})
    for group in groups:
        indices = [idx for idx, token in enumerate(vocab) if _category_group(token) == group]
        if not indices:
            continue
        eligible = [row_idx for row_idx, labels in enumerate(label_rows) if any(labels[idx] for idx in indices)]
        if not eligible:
            per_group[group] = {"threshold": default_threshold, "positive_examples": 0, "accuracy": None}
            continue
        best_threshold = default_threshold
        best_key: tuple[float, float] = (-1.0, default_threshold)
        best_correct = 0
        for threshold in grid:
            correct = 0
            for row_idx in eligible:
                pred = tuple(vocab[idx] for idx in indices if score_rows[row_idx][idx] >= threshold)
                gold = tuple(vocab[idx] for idx in indices if label_rows[row_idx][idx])
                correct += int(pred == gold)
            accuracy = correct / len(eligible)
            # Prefer exact-set accuracy, then a conservative/higher threshold.
            key = (accuracy, threshold)
            if key > best_key:
                best_key = key
                best_threshold = threshold
                best_correct = correct
        for idx in indices:
            thresholds[vocab[idx]] = float(best_threshold)
        per_group[group] = {
            "threshold": float(best_threshold),
            "positive_examples": len(eligible),
            "accuracy": best_correct / len(eligible),
            "correct": best_correct,
        }
    diagnostics = {
        "default_threshold": default_threshold,
        "grid": grid,
        "per_group": per_group,
    }
    return thresholds, diagnostics


def _calibrated_group_fbeta_thresholds_from_scores(
    score_rows: list[list[float]],
    label_rows: list[list[int]],
    vocab: list[str],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds = {token: default_threshold for token in vocab}
    per_group: dict[str, Any] = {}
    beta2 = beta * beta
    groups = sorted({_category_group(token) for token in vocab})
    for group in groups:
        indices = [idx for idx, token in enumerate(vocab) if _category_group(token) == group]
        if not indices:
            continue
        positives = sum(1 for labels in label_rows if any(labels[idx] for idx in indices))
        if positives == 0:
            per_group[group] = {"threshold": default_threshold, "positive_examples": 0, "score": None}
            continue
        best_threshold = default_threshold
        best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, default_threshold)
        best_stats: dict[str, Any] = {}
        for threshold in grid:
            tp = fp = fn = 0
            for scores, labels in zip(score_rows, label_rows):
                pred = tuple(vocab[idx] for idx in indices if scores[idx] >= threshold)
                gold = tuple(vocab[idx] for idx in indices if labels[idx])
                if gold and pred == gold:
                    tp += 1
                elif gold:
                    fn += 1
                    if pred:
                        fp += 1
                elif pred:
                    fp += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            score = (
                (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
                if precision or recall
                else 0.0
            )
            # Prefer F-beta, then precision to suppress click spam, then recall.
            key = (score, precision, recall, threshold)
            if key > best_key:
                best_key = key
                best_threshold = threshold
                best_stats = {
                    "positive_examples": positives,
                    "score": score,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
        for idx in indices:
            thresholds[vocab[idx]] = float(best_threshold)
        per_group[group] = {"threshold": float(best_threshold), **best_stats}
    diagnostics = {
        "default_threshold": default_threshold,
        "beta": beta,
        "grid": grid,
        "per_group": per_group,
    }
    return thresholds, diagnostics


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _button_name(token: str) -> str | None:
    for prefix, name in (
        ("MOUSE_LEFT_", "left"),
        ("MOUSE_RIGHT_", "right"),
        ("MOUSE_MIDDLE_", "middle"),
    ):
        if token.startswith(prefix):
            return name
    return None


def _apply_button_tokens(state: dict[str, float], tokens: list[str]) -> None:
    for token in tokens:
        name = _button_name(token)
        if name is None:
            continue
        if token.endswith("_DOWN"):
            state[name] = 1.0
        elif token.endswith("_UP"):
            state[name] = 0.0


def _history_vector(
    history: list[list[str]],
    button_state: dict[str, float],
    vocab: list[str],
    *,
    history_len: int,
) -> list[float]:
    if history_len <= 0:
        return []
    values: list[float] = []
    vocab_set_by_slot = []
    for offset in range(history_len):
        tokens = history[-1 - offset] if offset < len(history) else []
        vocab_set_by_slot.append(set(tokens))
        dx, dy = target_mouse_delta({"ground_truth_tokens": tokens})
        # Delta tokens are small integer classes; keep them in a stable range
        # relative to binary categorical indicators.
        values.extend([float(dx) / 8.0, float(dy) / 8.0])
    for token_set in vocab_set_by_slot:
        values.extend([1.0 if token in token_set else 0.0 for token in vocab])
    values.extend([float(button_state.get(name, 0.0)) for name in ("left", "right", "middle")])
    return values


def _empty_button_state() -> dict[str, float]:
    return {"left": 0.0, "right": 0.0, "middle": 0.0}


def _append_history(
    history: list[list[str]],
    button_state: dict[str, float],
    tokens: list[str],
    *,
    history_len: int,
) -> None:
    history.append(list(tokens))
    if len(history) > history_len:
        del history[:-history_len]
    _apply_button_tokens(button_state, list(tokens))


def _action_history_features(
    records: list[dict[str, Any]],
    vocab: list[str],
    *,
    history_len: int,
    seed_records: list[dict[str, Any]] | None = None,
    token_rows: dict[str, list[str]] | None = None,
) -> list[list[float]]:
    """Build causal action-history features aligned to ``records``.

    Features for a row are computed before that row's tokens are appended, so
    training can use teacher-forced prior actions while heldout inference can
    use predicted prior actions without peeking at heldout labels.
    """

    if history_len <= 0:
        return [[] for _ in records]
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}

    def ensure(recording_id: str) -> tuple[list[list[str]], dict[str, float]]:
        if recording_id not in histories:
            histories[recording_id] = []
            button_states[recording_id] = _empty_button_state()
        return histories[recording_id], button_states[recording_id]

    for row in sorted(seed_records or [], key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0)))):
        history, button_state = ensure(str(row.get("recording_id", "")))
        _append_history(history, button_state, list(row.get("ground_truth_tokens", [])), history_len=history_len)

    features_by_id: dict[str, list[float]] = {}
    ordered = sorted(records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0))))
    for row in ordered:
        recording_id = str(row.get("recording_id", ""))
        history, button_state = ensure(recording_id)
        features_by_id[str(row["sequence_id"])] = _history_vector(history, button_state, vocab, history_len=history_len)
        tokens = (token_rows or {}).get(str(row["sequence_id"]), list(row.get("ground_truth_tokens", [])))
        _append_history(history, button_state, list(tokens), history_len=history_len)
    return [features_by_id[str(row["sequence_id"])] for row in records]


def _mouse_baseline_deltas(
    records: list[dict[str, Any]],
    *,
    mode: str,
    train_records: list[dict[str, Any]] | None = None,
) -> list[tuple[float, float]]:
    if mode == "none":
        return [(0.0, 0.0) for _ in records]
    if mode not in {"causal_last_seen", "target_last_seen_train"}:
        raise ValueError(f"unsupported mouse_baseline mode: {mode}")

    fallback = (0.0, 0.0)
    last_by_recording: dict[str, tuple[float, float]] = {}
    last_by_game: dict[str, tuple[float, float]] = {}
    if mode == "target_last_seen_train":
        if train_records is None:
            raise ValueError("target_last_seen_train requires train_records")
        for row in sorted(train_records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0)))):
            delta = target_mouse_delta(row)
            last_by_recording[str(row.get("recording_id", ""))] = delta
            last_by_game[str(row.get("game", "unknown"))] = delta
        if train_records:
            fallback = target_mouse_delta(train_records[-1])

    baselines_by_id: dict[str, tuple[float, float]] = {}
    ordered = sorted(records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0))))
    for row in ordered:
        baseline = (
            last_by_recording.get(str(row.get("recording_id", "")))
            or last_by_game.get(str(row.get("game", "unknown")))
            or fallback
        )
        baselines_by_id[str(row["sequence_id"])] = baseline
        if mode == "causal_last_seen":
            delta = target_mouse_delta(row)
            last_by_recording[str(row.get("recording_id", ""))] = delta
            last_by_game[str(row.get("game", "unknown"))] = delta
            fallback = delta
    return [baselines_by_id[str(row["sequence_id"])] for row in records]


def _seed_mouse_delta_state(
    records: list[dict[str, Any]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float]]:
    last_by_recording: dict[str, tuple[float, float]] = {}
    last_by_game: dict[str, tuple[float, float]] = {}
    fallback = (0.0, 0.0)
    for row in sorted(records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0)))):
        delta = target_mouse_delta(row)
        last_by_recording[str(row.get("recording_id", ""))] = delta
        last_by_game[str(row.get("game", "unknown"))] = delta
        fallback = delta
    return last_by_recording, last_by_game, fallback


def _tensorize(
    torch,
    records: list[dict[str, Any]],
    device: str,
    vocab: list[str],
    *,
    feature_mode: str,
    mouse_baselines: list[tuple[float, float]] | None = None,
    residual_mouse: bool = False,
    history_features: list[list[float]] | None = None,
):
    mouse_baselines = mouse_baselines or [(0.0, 0.0) for _ in records]
    history_features = history_features or [[] for _ in records]
    def features_for(row: dict[str, Any], baseline: tuple[float, float], history: list[float]) -> list[float]:
        values = record_features(row, feature_mode=feature_mode)
        if residual_mouse:
            values = values + [float(baseline[0]), float(baseline[1])]
        if history:
            values = values + [float(value) for value in history]
        return values

    xs = torch.tensor(
        [features_for(row, baseline, history) for row, baseline, history in zip(records, mouse_baselines, history_features)],
        dtype=torch.float32,
        device=device,
    )
    target_deltas = [target_mouse_delta(row) for row in records]
    if residual_mouse:
        mouse_targets = [(tx - bx, ty - by) for (tx, ty), (bx, by) in zip(target_deltas, mouse_baselines)]
    else:
        mouse_targets = target_deltas
    mouse_y = torch.tensor(mouse_targets, dtype=torch.float32, device=device)
    cat_y = torch.zeros((len(records), len(vocab)), dtype=torch.float32, device=device)
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    for row_idx, row in enumerate(records):
        for token in set(row.get("ground_truth_tokens", [])):
            if token in vocab_index:
                cat_y[row_idx, vocab_index[token]] = 1.0
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (xs - mean) / std, mouse_y, cat_y, mean.squeeze(0).detach().cpu().tolist(), std.squeeze(0).detach().cpu().tolist()


def _categorical_loss(torch, logits, targets, pos_weight, config: dict[str, Any]):
    mode = str(config.get("categorical_loss", "bce"))
    if mode == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
    if mode != "focal":
        raise ValueError(f"unsupported categorical_loss: {mode}")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    p_t = (probs * targets) + ((1.0 - probs) * (1.0 - targets))
    gamma = float(config.get("focal_gamma", 2.0))
    loss = ((1.0 - p_t).clamp_min(1e-6) ** gamma) * bce
    if "focal_alpha" in config:
        alpha = float(config["focal_alpha"])
        alpha_factor = (alpha * targets) + ((1.0 - alpha) * (1.0 - targets))
        loss = alpha_factor * loss
    return loss.mean()


def _fit_torch_model(
    torch,
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    history_vocab: list[str],
    button_classes: list[tuple[str, ...]],
    button_head_mode: str,
    mouse_head_mode: str,
    mouse_axis_classes: list[str],
    feature_mode: str,
    residual_mouse: bool,
    action_history_len: int,
):
    train_mouse_baselines = _mouse_baseline_deltas(
        records,
        mode="causal_last_seen" if residual_mouse else "none",
    )
    history_features = _action_history_features(
        records,
        history_vocab,
        history_len=action_history_len,
    )
    train_x, mouse_y, cat_y, mean, std = _tensorize(
        torch,
        records,
        device,
        category_vocab,
        feature_mode=feature_mode,
        mouse_baselines=train_mouse_baselines,
        residual_mouse=residual_mouse,
        history_features=history_features,
    )
    cat_pos_weight = None
    if category_vocab:
        positives = cat_y.sum(dim=0)
        negatives = max(1, cat_y.shape[0]) - positives
        cat_pos_weight = (negatives / positives.clamp_min(1.0)).clamp(
            max=float(config.get("categorical_pos_weight_cap", 20.0))
        )
    button_y = None
    button_class_weight = None
    if button_head_mode == "softmax" and button_classes:
        button_y = torch.tensor(_button_target_indices(records, button_classes), dtype=torch.long, device=device)
        if len(button_classes) > 1:
            counts = torch.bincount(button_y, minlength=len(button_classes)).to(dtype=torch.float32)
            total = counts.sum().clamp_min(1.0)
            button_class_weight = (total / (counts.clamp_min(1.0) * len(button_classes))).clamp(
                max=float(config.get("button_softmax_class_weight_cap", 20.0))
            )
            button_class_weight[0] = button_class_weight[0] * float(config.get("button_softmax_no_button_weight", 1.0))
            if len(button_classes) > 1:
                button_class_weight[1:] = button_class_weight[1:] * float(config.get("button_softmax_positive_weight", 1.0))
    mouse_axis_dx_y = mouse_axis_dy_y = None
    mouse_axis_class_weight = None
    if mouse_head_mode == "axis_softmax":
        dx_indices, dy_indices = _axis_class_indices(records, mouse_axis_classes)
        mouse_axis_dx_y = torch.tensor(dx_indices, dtype=torch.long, device=device)
        mouse_axis_dy_y = torch.tensor(dy_indices, dtype=torch.long, device=device)
        all_indices = torch.tensor(dx_indices + dy_indices, dtype=torch.long, device=device)
        counts = torch.bincount(all_indices, minlength=len(mouse_axis_classes)).to(dtype=torch.float32)
        total = counts.sum().clamp_min(1.0)
        mouse_axis_class_weight = (total / (counts.clamp_min(1.0) * len(mouse_axis_classes))).clamp(
            max=float(config.get("mouse_axis_class_weight_cap", 20.0))
        )
    input_dim = int(train_x.shape[1])
    button_output_dim = len(button_classes) if button_head_mode == "softmax" else 0
    mouse_axis_output_dim = (2 * len(mouse_axis_classes)) if mouse_head_mode == "axis_softmax" else 0
    model = _build_model(
        torch,
        input_dim=input_dim,
        output_dim=2 + len(category_vocab) + button_output_dim + mouse_axis_output_dim,
        hidden_dim=int(config.get("hidden_dim", 128)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.05)),
        config=config,
        feature_mode=feature_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(train_x.shape[0], device=device)
        losses = []
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_x[idx])
            mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y[idx])
            category_end = 2 + len(category_vocab)
            button_end = category_end + button_output_dim
            if category_vocab:
                cat_loss = _categorical_loss(torch, pred[:, 2:category_end], cat_y[idx], cat_pos_weight, config)
            else:
                cat_loss = torch.tensor(0.0, device=device)
            if button_head_mode == "softmax" and button_y is not None and len(button_classes) > 1:
                button_loss = torch.nn.functional.cross_entropy(
                    pred[:, category_end : category_end + len(button_classes)],
                    button_y[idx],
                    weight=button_class_weight,
                )
            else:
                button_loss = torch.tensor(0.0, device=device)
            if (
                mouse_head_mode == "axis_softmax"
                and mouse_axis_dx_y is not None
                and mouse_axis_dy_y is not None
            ):
                axis_count = len(mouse_axis_classes)
                dx_logits = pred[:, button_end : button_end + axis_count]
                dy_logits = pred[:, button_end + axis_count : button_end + (2 * axis_count)]
                mouse_axis_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(dx_logits, mouse_axis_dx_y[idx], weight=mouse_axis_class_weight)
                    + torch.nn.functional.cross_entropy(dy_logits, mouse_axis_dy_y[idx], weight=mouse_axis_class_weight)
                )
            else:
                mouse_axis_loss = torch.tensor(0.0, device=device)
            loss = (
                float(config.get("mouse_regression_loss_weight", 1.0)) * mouse_loss
                + float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)) * mouse_axis_loss
                + float(config.get("categorical_loss_weight", 0.5)) * cat_loss
                + float(config.get("button_softmax_loss_weight", 1.0)) * button_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses)})
    return model, mean, std, history, input_dim, train_mouse_baselines


def _predict_raw_outputs(
    torch,
    model,
    records: list[dict[str, Any]],
    *,
    device: str,
    mean: list[float],
    std: list[float],
    feature_mode: str,
    mouse_baselines: list[tuple[float, float]],
    residual_mouse: bool,
    history_features: list[list[float]] | None = None,
) -> list[list[float]]:
    history_features = history_features or [[] for _ in records]
    target_features = []
    for row, baseline, history in zip(records, mouse_baselines, history_features):
        values = record_features(row, feature_mode=feature_mode)
        if residual_mouse:
            values = values + [float(baseline[0]), float(baseline[1])]
        if history:
            values = values + [float(value) for value in history]
        target_features.append(values)
    raw_target_x = torch.tensor(target_features, dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6)
    target_x = (raw_target_x - mean_t) / std_t
    model.eval()
    with torch.no_grad():
        return model(target_x).detach().cpu().tolist()


def _prediction_from_output(
    output: list[float],
    *,
    base_dx: float,
    base_dy: float,
    residual_mouse: bool,
    category_vocab: list[str],
    category_thresholds: dict[str, float],
    category_threshold: float,
    button_head_mode: str = "multilabel",
    button_classes: list[tuple[str, ...]] | None = None,
    button_softmax_threshold: float = 0.5,
    mouse_head_mode: str = "regression",
    mouse_axis_classes: list[str] | None = None,
) -> tuple[float, float, list[str]]:
    dx, dy = float(output[0]), float(output[1])
    if residual_mouse:
        dx += float(base_dx)
        dy += float(base_dy)
    category_end = 2 + len(category_vocab)
    button_end = category_end
    if button_head_mode == "softmax":
        button_end = category_end + len(button_classes or [])
    if mouse_head_mode == "axis_softmax":
        axis_classes = mouse_axis_classes or MOUSE_AXIS_CLASSES
        axis_count = len(axis_classes)
        dx_logits = output[button_end : button_end + axis_count]
        dy_logits = output[button_end + axis_count : button_end + (2 * axis_count)]
        if dx_logits and dy_logits:
            dx_idx = max(range(len(dx_logits)), key=lambda idx: dx_logits[idx])
            dy_idx = max(range(len(dy_logits)), key=lambda idx: dy_logits[idx])
            dx = _axis_class_to_delta(axis_classes[dx_idx])
            dy = _axis_class_to_delta(axis_classes[dy_idx])
    tokens = tokens_from_delta(float(dx), float(dy))
    for token, logit in zip(category_vocab, output[2:category_end]):
        prob = _sigmoid(float(logit))
        if prob >= float(category_thresholds.get(token, category_threshold)):
            tokens.append(token)
    if button_head_mode == "softmax":
        classes = button_classes or []
        button_logits = output[category_end : category_end + len(classes)]
        probs = _softmax([float(value) for value in button_logits])
        if probs:
            best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
            if best_idx != 0 and probs[best_idx] >= float(button_softmax_threshold):
                tokens.extend(classes[best_idx])
    return dx, dy, tokens


def _predict_autoregressive_target(
    torch,
    model,
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    device: str,
    mean: list[float],
    std: list[float],
    feature_mode: str,
    target_mouse_baselines: list[tuple[float, float]],
    residual_mouse: bool,
    history_vocab: list[str],
    category_vocab: list[str],
    category_thresholds: dict[str, float],
    category_threshold: float,
    button_head_mode: str,
    button_classes: list[tuple[str, ...]],
    button_softmax_threshold: float,
    mouse_head_mode: str,
    mouse_axis_classes: list[str],
    action_history_len: int,
) -> dict[str, dict[str, Any]]:
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}

    def ensure(recording_id: str) -> tuple[list[list[str]], dict[str, float]]:
        if recording_id not in histories:
            histories[recording_id] = []
            button_states[recording_id] = _empty_button_state()
        return histories[recording_id], button_states[recording_id]

    for row in sorted(train_records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0)))):
        history, button_state = ensure(str(row.get("recording_id", "")))
        _append_history(history, button_state, list(row.get("ground_truth_tokens", [])), history_len=action_history_len)

    baseline_by_id = {str(row["sequence_id"]): baseline for row, baseline in zip(target_records, target_mouse_baselines)}
    last_mouse_by_recording, last_mouse_by_game, last_mouse_fallback = _seed_mouse_delta_state(train_records)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6)
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    ordered = sorted(target_records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0))))
    for row in ordered:
        sequence_id = str(row["sequence_id"])
        recording_id = str(row.get("recording_id", ""))
        game = str(row.get("game", "unknown"))
        history, button_state = ensure(recording_id)
        history_features = _history_vector(history, button_state, history_vocab, history_len=action_history_len)
        if residual_mouse:
            base_dx, base_dy = (
                last_mouse_by_recording.get(recording_id)
                or last_mouse_by_game.get(game)
                or last_mouse_fallback
            )
        else:
            base_dx, base_dy = baseline_by_id[sequence_id]
        values = record_features(row, feature_mode=feature_mode)
        if residual_mouse:
            values = values + [float(base_dx), float(base_dy)]
        values = values + history_features
        raw_x = torch.tensor([values], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model((raw_x - mean_t) / std_t).detach().cpu().tolist()[0]
        dx, dy, tokens = _prediction_from_output(
            output,
            base_dx=base_dx,
            base_dy=base_dy,
            residual_mouse=residual_mouse,
            category_vocab=category_vocab,
            category_thresholds=category_thresholds,
            category_threshold=category_threshold,
            button_head_mode=button_head_mode,
            button_classes=button_classes,
            button_softmax_threshold=button_softmax_threshold,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
        )
        outputs[sequence_id] = {"output": output, "dx": dx, "dy": dy, "tokens": tokens}
        _append_history(history, button_state, tokens, history_len=action_history_len)
        if residual_mouse:
            predicted_delta = (float(dx), float(dy))
            last_mouse_by_recording[recording_id] = predicted_delta
            last_mouse_by_game[game] = predicted_delta
            last_mouse_fallback = predicted_delta
    return outputs


def train_torch_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not bool(config.get("force_cpu", False)):
        device = "cuda"
    else:
        device = "cpu"
    train_records = read_jsonl(config["train_records"])
    target_records = read_jsonl(config["target_records"])
    feature_mode = str(config.get("feature_mode", "summary"))
    mouse_target_mode = str(config.get("mouse_target_mode", "absolute"))
    if mouse_target_mode not in {"absolute", "residual_last_seen"}:
        raise ValueError(f"unsupported mouse_target_mode: {mouse_target_mode}")
    residual_mouse = mouse_target_mode == "residual_last_seen"
    mouse_head_mode = str(config.get("mouse_head_mode", "regression"))
    if mouse_head_mode not in {"regression", "axis_softmax"}:
        raise ValueError(f"unsupported mouse_head_mode: {mouse_head_mode}")
    raw_axis_classes = config.get("mouse_axis_classes")
    mouse_axis_classes = (
        [str(value) for value in raw_axis_classes]
        if isinstance(raw_axis_classes, list) and raw_axis_classes
        else list(MOUSE_AXIS_CLASSES)
    )
    history_vocab = categorical_token_vocab(train_records, min_count=int(config.get("categorical_min_count", 1)))
    button_head_mode = str(config.get("button_head_mode", "multilabel"))
    if button_head_mode not in {"multilabel", "softmax"}:
        raise ValueError(f"unsupported button_head_mode: {button_head_mode}")
    button_classes: list[tuple[str, ...]] = []
    if button_head_mode == "softmax":
        category_vocab = [token for token in history_vocab if not _is_mouse_button_token(token)]
        button_classes = button_softmax_classes(
            train_records,
            min_count=int(config.get("button_softmax_min_count", 1)),
        )
    else:
        category_vocab = history_vocab
    action_history_len = int(config.get("action_history_len", 0))
    if action_history_len < 0:
        raise ValueError("action_history_len must be non-negative")
    category_threshold = float(config.get("category_threshold", 0.35))
    category_thresholds = {token: category_threshold for token in category_vocab}
    button_softmax_threshold = float(config.get("button_softmax_threshold", 0.5))
    button_softmax_threshold_mode = str(config.get("button_softmax_threshold_mode", "global"))
    if button_softmax_threshold_mode not in {"global", "fbeta_calibrated"}:
        raise ValueError(f"unsupported button_softmax_threshold_mode: {button_softmax_threshold_mode}")
    calibration_info: dict[str, Any] = {
        "category_threshold_mode": str(config.get("category_threshold_mode", "global")),
        "category_threshold": category_threshold,
        "category_thresholds": category_thresholds,
        "button_head_mode": button_head_mode,
        "button_softmax_threshold": button_softmax_threshold,
        "button_softmax_threshold_mode": button_softmax_threshold_mode,
    }
    if button_head_mode == "softmax":
        calibration_info["button_softmax_classes"] = _button_class_metadata(train_records, button_classes)
    threshold_mode = calibration_info["category_threshold_mode"]
    if threshold_mode not in {"global", "per_token_calibrated", "group_exact_calibrated", "group_fbeta_calibrated"}:
        raise ValueError(f"unsupported category_threshold_mode: {threshold_mode}")

    calibrated_model = None
    calibrated_mean = calibrated_std = calibrated_history = None
    needs_category_calibration = (
        threshold_mode in {"per_token_calibrated", "group_exact_calibrated", "group_fbeta_calibrated"}
        and bool(category_vocab)
    )
    needs_button_calibration = (
        button_head_mode == "softmax"
        and button_softmax_threshold_mode == "fbeta_calibrated"
        and len(button_classes) > 1
    )
    if needs_category_calibration or needs_button_calibration:
        fraction = float(config.get("category_calibration_fraction", 0.0))
        fit_records, calibration_records = _split_calibration_records(
            train_records,
            fraction=fraction,
            min_calibration_per_recording=int(config.get("category_calibration_min_per_recording", 1)),
        )
        if not calibration_records:
            fit_records, calibration_records = train_records, train_records
        torch.manual_seed(seed)
        calibration_model, calibration_mean, calibration_std, calibration_history, _, _ = _fit_torch_model(
            torch,
            fit_records,
            config=config,
            device=device,
            category_vocab=category_vocab,
            history_vocab=history_vocab,
            button_classes=button_classes,
            button_head_mode=button_head_mode,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
            feature_mode=feature_mode,
            residual_mouse=residual_mouse,
            action_history_len=action_history_len,
        )
        calibration_baselines = _mouse_baseline_deltas(
            calibration_records,
            mode="target_last_seen_train" if residual_mouse else "none",
            train_records=fit_records,
        )
        calibration_history_features = _action_history_features(
            calibration_records,
            history_vocab,
            history_len=action_history_len,
            seed_records=fit_records,
        )
        calibration_outputs = _predict_raw_outputs(
            torch,
            calibration_model,
            calibration_records,
            device=device,
            mean=calibration_mean,
            std=calibration_std,
            feature_mode=feature_mode,
            mouse_baselines=calibration_baselines,
            residual_mouse=residual_mouse,
            history_features=calibration_history_features,
        )
        raw_grid = config.get("category_calibration_grid")
        grid = (
            [float(value) for value in raw_grid]
            if isinstance(raw_grid, list)
            else [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        if needs_category_calibration:
            score_rows = [
                [_sigmoid(float(logit)) for logit in output[2 : 2 + len(category_vocab)]]
                for output in calibration_outputs
            ]
            label_rows = _category_label_matrix(calibration_records, category_vocab)
            if threshold_mode == "group_exact_calibrated":
                category_thresholds, threshold_diagnostics = _calibrated_group_thresholds_from_scores(
                    score_rows,
                    label_rows,
                    category_vocab,
                    default_threshold=category_threshold,
                    grid=grid,
                )
            elif threshold_mode == "group_fbeta_calibrated":
                category_thresholds, threshold_diagnostics = _calibrated_group_fbeta_thresholds_from_scores(
                    score_rows,
                    label_rows,
                    category_vocab,
                    default_threshold=category_threshold,
                    grid=grid,
                    beta=float(config.get("category_calibration_beta", 1.0)),
                )
            else:
                category_thresholds, threshold_diagnostics = _calibrated_category_thresholds_from_scores(
                    score_rows,
                    label_rows,
                    category_vocab,
                    default_threshold=category_threshold,
                    grid=grid,
                    beta=float(config.get("category_calibration_beta", 1.0)),
                )
        else:
            threshold_diagnostics = None
        if needs_button_calibration:
            button_start = 2 + len(category_vocab)
            button_score_rows = [
                _softmax([float(value) for value in output[button_start : button_start + len(button_classes)]])
                for output in calibration_outputs
            ]
            button_softmax_threshold, button_threshold_diagnostics = _calibrated_button_softmax_threshold_from_scores(
                button_score_rows,
                _button_target_indices(calibration_records, button_classes),
                button_classes,
                default_threshold=button_softmax_threshold,
                grid=grid,
                beta=float(config.get("button_softmax_calibration_beta", config.get("category_calibration_beta", 0.5))),
            )
        else:
            button_threshold_diagnostics = None
        calibration_info.update(
            {
                "category_thresholds": category_thresholds,
                "category_calibration_records": len(calibration_records),
                "category_calibration_fit_records": len(fit_records),
                "category_calibration_refit_full_train": bool(config.get("category_calibration_refit_full_train", True)),
                "category_calibration_history_tail": calibration_history[-3:],
                "category_threshold_diagnostics": threshold_diagnostics,
                "button_softmax_threshold": button_softmax_threshold,
                "button_softmax_threshold_diagnostics": button_threshold_diagnostics,
            }
        )
        if not bool(config.get("category_calibration_refit_full_train", True)):
            calibrated_model = calibration_model
            calibrated_mean = calibration_mean
            calibrated_std = calibration_std
            calibrated_history = calibration_history

    if calibrated_model is not None:
        model = calibrated_model
        mean = calibrated_mean
        std = calibrated_std
        history = calibrated_history
        input_dim = len(mean)
    else:
        torch.manual_seed(seed)
        model, mean, std, history, input_dim, _ = _fit_torch_model(
            torch,
            train_records,
            config=config,
            device=device,
            category_vocab=category_vocab,
            history_vocab=history_vocab,
            button_classes=button_classes,
            button_head_mode=button_head_mode,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
            feature_mode=feature_mode,
            residual_mouse=residual_mouse,
            action_history_len=action_history_len,
        )
    out_dir = Path(config.get("output_dir", "outputs/idm_torch"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "config": config,
            "history": history,
            "category_thresholds": category_thresholds,
            "category_vocab": category_vocab,
            "history_vocab": history_vocab,
            "button_head_mode": button_head_mode,
            "button_classes": [list(tokens) for tokens in button_classes],
            "button_softmax_threshold": button_softmax_threshold,
            "mouse_head_mode": mouse_head_mode,
            "mouse_axis_classes": mouse_axis_classes,
            "model_arch": str(config.get("model_arch", "mlp")),
        },
        checkpoint_path,
    )
    # Predict heldout pseudo-labels.
    target_mouse_baselines = _mouse_baseline_deltas(
        target_records,
        mode="target_last_seen_train" if residual_mouse else "none",
        train_records=train_records,
    )
    autoregressive_outputs: dict[str, dict[str, Any]] | None = None
    if action_history_len > 0:
        autoregressive_outputs = _predict_autoregressive_target(
            torch,
            model,
            train_records,
            target_records,
            device=device,
            mean=mean,
            std=std,
            feature_mode=feature_mode,
            target_mouse_baselines=target_mouse_baselines,
            residual_mouse=residual_mouse,
            history_vocab=history_vocab,
            category_vocab=category_vocab,
            category_thresholds=category_thresholds,
            category_threshold=category_threshold,
            button_head_mode=button_head_mode,
            button_classes=button_classes,
            button_softmax_threshold=button_softmax_threshold,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
            action_history_len=action_history_len,
        )
        deltas = [autoregressive_outputs[str(row["sequence_id"])]["output"] for row in target_records]
    else:
        deltas = _predict_raw_outputs(
            torch,
            model,
            target_records,
            device=device,
            mean=mean,
            std=std,
            feature_mode=feature_mode,
            mouse_baselines=target_mouse_baselines,
            residual_mouse=residual_mouse,
        )
    fingerprint_mouse_baselines = _mouse_baseline_deltas(
        train_records,
        mode="causal_last_seen" if residual_mouse else "none",
    )
    fingerprint_history_features = _action_history_features(
        train_records,
        history_vocab,
        history_len=action_history_len,
    )
    train_hash = stable_hash_json(
        [
            {
                "id": row["sequence_id"],
                "tokens": row.get("ground_truth_tokens", []),
                "features": record_features(row, feature_mode=feature_mode),
                "mouse_baseline": baseline,
                "action_history": history_features,
                "feature_mode": feature_mode,
                "mouse_target_mode": mouse_target_mode,
                "action_history_len": action_history_len,
            }
            for row, baseline, history_features in zip(train_records, fingerprint_mouse_baselines, fingerprint_history_features)
        ]
    )
    pseudo_rows = []
    predictions = []
    for row, output, (base_dx, base_dy) in zip(target_records, deltas, target_mouse_baselines):
        if autoregressive_outputs is not None:
            pred = autoregressive_outputs[str(row["sequence_id"])]
            dx, dy, tokens = float(pred["dx"]), float(pred["dy"]), list(pred["tokens"])
        else:
            dx, dy, tokens = _prediction_from_output(
                output,
                base_dx=float(base_dx),
                base_dy=float(base_dy),
                residual_mouse=residual_mouse,
                category_vocab=category_vocab,
                category_thresholds=category_thresholds,
                category_threshold=category_threshold,
                button_head_mode=button_head_mode,
                button_classes=button_classes,
                button_softmax_threshold=button_softmax_threshold,
                mouse_head_mode=mouse_head_mode,
                mouse_axis_classes=mouse_axis_classes,
            )
        confidence = max(0.05, min(0.99, 1.0 / (1.0 + abs(float(dx)) + abs(float(dy)))))
        pseudo = {
            "schema": "idm_pseudolabel.v1",
            "sequence_id": row["sequence_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "predicted_tokens": tokens,
            "label_source": "idm_generated",
            "confidence": confidence,
            "model": str(config.get("model_name", "torch_mlp_idm")),
            "training_split_hash": train_hash,
            "input_window": {"frame_ref": row.get("frame", {}).get("path", ""), "frame_index": int(row.get("frame", {}).get("index", 0))},
        }
        validate_named(pseudo, "idm_pseudolabel.schema.json")
        pseudo_rows.append(pseudo)
        predictions.append({"sequence_id": row["sequence_id"], "recording_id": row.get("recording_id"), "game": row.get("game"), "timestamp_ns": row["timestamp_ns"], "predicted_tokens": tokens})
    pseudo_path = out_dir / "pseudolabels.jsonl"
    filtered_path = out_dir / "pseudolabels.filtered.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    threshold = float(config.get("confidence_threshold", 0.15))
    write_jsonl(pseudo_path, pseudo_rows)
    write_jsonl(filtered_path, [row for row in pseudo_rows if row["confidence"] >= threshold])
    write_jsonl(predictions_path, predictions)
    metrics = compute_metrics(predictions, target_records)
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(config.get("model_name", "torch_mlp_idm")),
        "dataset_fingerprint": train_hash,
        "train_records": len(train_records),
        "target_records": len(target_records),
        "pseudo_label_path": str(pseudo_path),
        "filtered_pseudo_label_path": str(filtered_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "calibration": {
            "confidence_threshold": threshold,
            "kept": sum(1 for row in pseudo_rows if row["confidence"] >= threshold),
            "total": len(pseudo_rows),
            "last_train_loss": history[-1]["loss"] if history else None,
            **calibration_info,
        },
        "categorical_vocab": category_vocab,
        "history_vocab": history_vocab,
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "model_arch": str(config.get("model_arch", "mlp")),
        "mouse_target_mode": mouse_target_mode,
        "mouse_head_mode": mouse_head_mode,
        "mouse_axis_classes": mouse_axis_classes if mouse_head_mode == "axis_softmax" else [],
        "mouse_axis_loss_weight": float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)),
        "mouse_regression_loss_weight": float(config.get("mouse_regression_loss_weight", 1.0)),
        "mouse_axis_class_weight_cap": float(config.get("mouse_axis_class_weight_cap", 20.0)),
        "category_threshold": category_threshold,
        "category_threshold_mode": threshold_mode,
        "category_thresholds": category_thresholds,
        "categorical_loss": str(config.get("categorical_loss", "bce")),
        "categorical_pos_weight_cap": float(config.get("categorical_pos_weight_cap", 20.0)),
        "button_head_mode": button_head_mode,
        "button_softmax_threshold": button_softmax_threshold,
        "button_softmax_threshold_mode": button_softmax_threshold_mode,
        "button_softmax_classes": _button_class_metadata(train_records, button_classes) if button_head_mode == "softmax" else [],
        "button_softmax_loss_weight": float(config.get("button_softmax_loss_weight", 1.0)),
        "button_softmax_class_weight_cap": float(config.get("button_softmax_class_weight_cap", 20.0)),
        "action_history_len": action_history_len,
        "action_history_feedback": "autoregressive_predicted" if action_history_len > 0 else "none",
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    endpoints_path = config.get("endpoints")
    stat_comparison = None
    if endpoints_path:
        predictions_by_name = build_baseline_predictions(train_records, target_records)
        predictions_by_name[str(config.get("model_name", "torch_mlp_idm"))] = predictions
        stat_comparison = compare_systems(predictions_by_name, target_records, load_config(endpoints_path))
        write_json(out_dir / "statistical_comparison.json", stat_comparison)
    summary = {
        "schema": "torch_idm_train_summary.v1",
        "metadata": metadata,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "statistical_comparison": stat_comparison,
        "history_tail": history[-5:],
        "device": device,
    }
    write_json(config.get("summary_out", out_dir / "summary.json"), summary)
    return summary


def predict_torch_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    if torch.cuda.is_available() and not bool(config.get("force_cpu", False)):
        device = "cuda"
    else:
        device = "cpu"
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = dict(checkpoint.get("config", {}))
    records = read_jsonl(config["target_records"])
    seed_records = read_jsonl(config["seed_records"]) if config.get("seed_records") else []
    feature_mode = str(model_config.get("feature_mode", "summary"))
    residual_mouse = str(model_config.get("mouse_target_mode", "absolute")) == "residual_last_seen"
    category_vocab = [str(value) for value in checkpoint.get("category_vocab", [])]
    history_vocab = [str(value) for value in checkpoint.get("history_vocab", [])]
    button_head_mode = str(checkpoint.get("button_head_mode", "multilabel"))
    button_classes = [tuple(str(token) for token in row) for row in checkpoint.get("button_classes", [])]
    button_softmax_threshold = float(checkpoint.get("button_softmax_threshold", 0.5))
    mouse_head_mode = str(checkpoint.get("mouse_head_mode", "regression"))
    mouse_axis_classes = [str(value) for value in checkpoint.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)]
    mean = [float(value) for value in checkpoint["mean"]]
    std = [float(value) for value in checkpoint["std"]]
    output_dim = 2 + len(category_vocab)
    if button_head_mode == "softmax":
        output_dim += len(button_classes)
    if mouse_head_mode == "axis_softmax":
        output_dim += 2 * len(mouse_axis_classes)
    model = _build_model(
        torch,
        input_dim=len(mean),
        output_dim=output_dim,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        depth=int(model_config.get("depth", 3)),
        dropout=float(model_config.get("dropout", 0.05)),
        config=model_config,
        feature_mode=feature_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    category_threshold = float(model_config.get("category_threshold", 0.35))
    category_thresholds = {
        str(token): float(value)
        for token, value in (checkpoint.get("category_thresholds") or {}).items()
    } or {token: category_threshold for token in category_vocab}
    action_history_len = int(model_config.get("action_history_len", 0))
    target_mouse_baselines = _mouse_baseline_deltas(
        records,
        mode="target_last_seen_train" if residual_mouse else "none",
        train_records=seed_records,
    )
    predicted: list[dict[str, Any]] = []
    if action_history_len > 0:
        outputs_by_id = _predict_autoregressive_target(
            torch,
            model,
            seed_records,
            records,
            device=device,
            mean=mean,
            std=std,
            feature_mode=feature_mode,
            target_mouse_baselines=target_mouse_baselines,
            residual_mouse=residual_mouse,
            history_vocab=history_vocab,
            category_vocab=category_vocab,
            category_thresholds=category_thresholds,
            category_threshold=category_threshold,
            button_head_mode=button_head_mode,
            button_classes=button_classes,
            button_softmax_threshold=button_softmax_threshold,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
            action_history_len=action_history_len,
        )
        for row in records:
            pred = outputs_by_id[str(row["sequence_id"])]
            predicted.append({"dx": float(pred["dx"]), "dy": float(pred["dy"]), "tokens": list(pred["tokens"])})
    else:
        raw_outputs = _predict_raw_outputs(
            torch,
            model,
            records,
            device=device,
            mean=mean,
            std=std,
            feature_mode=feature_mode,
            mouse_baselines=target_mouse_baselines,
            residual_mouse=residual_mouse,
        )
        for output, (base_dx, base_dy) in zip(raw_outputs, target_mouse_baselines):
            dx, dy, tokens = _prediction_from_output(
                output,
                base_dx=float(base_dx),
                base_dy=float(base_dy),
                residual_mouse=residual_mouse,
                category_vocab=category_vocab,
                category_thresholds=category_thresholds,
                category_threshold=category_threshold,
                button_head_mode=button_head_mode,
                button_classes=button_classes,
                button_softmax_threshold=button_softmax_threshold,
                mouse_head_mode=mouse_head_mode,
                mouse_axis_classes=mouse_axis_classes,
            )
            predicted.append({"dx": dx, "dy": dy, "tokens": tokens})
    out_dir = Path(config.get("output_dir", "outputs/idm_torch_predict"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_hash = stable_hash_json(
        {
            "checkpoint_path": str(checkpoint_path),
            "target_records": [row["sequence_id"] for row in records],
            "seed_records": [row["sequence_id"] for row in seed_records],
        }
    )
    pseudo_rows = []
    prediction_rows = []
    for row, pred in zip(records, predicted):
        dx, dy, tokens = float(pred["dx"]), float(pred["dy"]), list(pred["tokens"])
        pseudo = {
            "schema": "idm_pseudolabel.v1",
            "sequence_id": row["sequence_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "predicted_tokens": tokens,
            "label_source": "idm_generated",
            "confidence": max(0.05, min(0.99, 1.0 / (1.0 + abs(dx) + abs(dy)))),
            "model": str(config.get("model_name", model_config.get("model_name", "torch_idm_predict"))),
            "training_split_hash": prediction_hash,
            "input_window": {
                "frame_ref": row.get("frame", {}).get("path", ""),
                "frame_index": int(row.get("frame", {}).get("index", 0)),
            },
        }
        validate_named(pseudo, "idm_pseudolabel.schema.json")
        pseudo_rows.append(pseudo)
        prediction_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "recording_id": row.get("recording_id"),
                "game": row.get("game"),
                "timestamp_ns": row["timestamp_ns"],
                "predicted_tokens": tokens,
            }
        )
    pseudo_path = out_dir / "pseudolabels.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    write_jsonl(pseudo_path, pseudo_rows)
    write_jsonl(predictions_path, prediction_rows)
    summary = {
        "schema": "torch_idm_predict_summary.v1",
        "checkpoint_path": str(checkpoint_path),
        "target_records": str(config["target_records"]),
        "seed_records": str(config.get("seed_records", "")),
        "num_records": len(records),
        "pseudolabels_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "device": device,
    }
    write_json(config.get("summary_out", out_dir / "summary.json"), summary)
    return summary
