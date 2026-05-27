from __future__ import annotations

import hashlib
import json
import os
import glob
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.config import load_config
from fdm_d2e.eval.statistics import cluster_bootstrap_delta, endpoint_value, holm_bonferroni
from fdm_d2e.io_utils import ensure_dir, read_json, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import state_tokens_from_event_tokens
from fdm_d2e.training.neural_idm import record_features, target_mouse_delta, tokens_from_delta
from fdm_d2e.training.torch_idm import (
    MOUSE_AXIS_CLASSES,
    _append_history,
    _axis_class_indices,
    _axis_suffix_from_delta,
    _build_model,
    _categorical_loss,
    _empty_button_state,
    _history_vector,
    _prediction_from_output,
    require_torch,
)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _record_paths_from_value(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _glob_record_paths(pattern: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if pattern is None:
        return []
    patterns = [pattern] if isinstance(pattern, (str, Path)) else list(pattern)
    paths: list[Path] = []
    for item in patterns:
        paths.extend(Path(match) for match in sorted(glob.glob(str(item))))
    return paths


def _record_paths_from_config(
    config: dict[str, Any],
    *,
    primary_key: str,
    paths_key: str,
    glob_key: str,
) -> list[Path]:
    explicit = config.get(paths_key)
    if explicit:
        return _record_paths_from_value(explicit)
    glob_paths = _glob_record_paths(config.get(glob_key))
    if glob_paths:
        return glob_paths
    return _record_paths_from_value(config[primary_key])


def _category_vocab_from_counts(counts: dict[str, int], min_count: int) -> list[str]:
    return sorted(token for token, count in counts.items() if count >= min_count)


def _is_category_token(token: str) -> bool:
    return token.startswith("KEY_") or (
        token.startswith("MOUSE_")
        and not token.startswith("MOUSE_DX_")
        and not token.startswith("MOUSE_DY_")
    )


def _is_keyboard_token(token: str) -> bool:
    return token.startswith("KEY_")


def _is_mouse_button_token(token: str) -> bool:
    return token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))


def _exact_set_label(row: dict[str, Any], predicate) -> tuple[str, ...]:
    return tuple(sorted({str(token) for token in row.get("ground_truth_tokens", []) if predicate(str(token))}))


def _keyboard_label(row: dict[str, Any]) -> tuple[str, ...]:
    return _exact_set_label(row, _is_keyboard_token)


def _button_label(row: dict[str, Any]) -> tuple[str, ...]:
    return _exact_set_label(row, _is_mouse_button_token)


def _label_key(label: tuple[str, ...]) -> str:
    return json.dumps(list(label), ensure_ascii=False, separators=(",", ":"))


def _label_from_key(key: str) -> tuple[str, ...]:
    raw = json.loads(key)
    if not isinstance(raw, list):
        raise ValueError(f"exact-set class key must encode a list: {key}")
    return tuple(str(token) for token in raw)


def _exact_set_classes_from_counts(counts: dict[str, int], *, min_count: int) -> list[tuple[str, ...]]:
    classes: list[tuple[str, ...]] = [()]
    labels = []
    for key, count in counts.items():
        label = _label_from_key(str(key))
        if label and int(count) >= int(min_count):
            labels.append(label)
    classes.extend(sorted(set(labels)))
    return classes


def _category_vocab_for_heads(stats: dict[str, Any], config: dict[str, Any]) -> list[str]:
    keyboard_softmax = str(config.get("keyboard_head_mode", "multilabel")) == "softmax"
    button_softmax = str(config.get("button_head_mode", "multilabel")) == "softmax"
    vocab = []
    for token in [str(value) for value in stats.get("category_vocab", [])]:
        if keyboard_softmax and _is_keyboard_token(token):
            continue
        if button_softmax and _is_mouse_button_token(token):
            continue
        vocab.append(token)
    return vocab


def _keyboard_classes_for_heads(stats: dict[str, Any], config: dict[str, Any]) -> list[tuple[str, ...]]:
    if str(config.get("keyboard_head_mode", "multilabel")) != "softmax":
        return []
    if isinstance(config.get("keyboard_classes"), list):
        return [tuple(str(token) for token in row) for row in config["keyboard_classes"]]
    return _exact_set_classes_from_counts(
        {str(k): int(v) for k, v in dict(stats.get("keyboard_class_counts", {})).items()},
        min_count=int(config.get("keyboard_softmax_min_count", 1)),
    )


def _button_classes_for_heads(stats: dict[str, Any], config: dict[str, Any]) -> list[tuple[str, ...]]:
    if str(config.get("button_head_mode", "multilabel")) != "softmax":
        return []
    if isinstance(config.get("button_classes"), list):
        return [tuple(str(token) for token in row) for row in config["button_classes"]]
    return _exact_set_classes_from_counts(
        {str(k): int(v) for k, v in dict(stats.get("button_class_counts", {})).items()},
        min_count=int(config.get("button_softmax_min_count", 1)),
    )


def _exact_set_targets(torch, rows: list[dict[str, Any]], classes: list[tuple[str, ...]], label_fn, *, device: str | None = None):
    class_index = {label: idx for idx, label in enumerate(classes)}
    values = [int(class_index.get(label_fn(row), 0)) for row in rows]
    kwargs = {"dtype": torch.long}
    if device is not None:
        kwargs["device"] = device
    return torch.tensor(values, **kwargs)


def _exact_set_class_weights(
    torch,
    counts: dict[str, int],
    classes: list[tuple[str, ...]],
    *,
    cap: float,
    empty_weight: float = 1.0,
    positive_weight: float = 1.0,
    device: str,
):
    if not classes:
        return None
    class_counts = [max(1, int(counts.get(_label_key(label), 0))) for label in classes]
    total = max(1, sum(class_counts))
    denom = max(1, len(classes))
    weights = [min(float(cap), float(total) / (float(count) * float(denom))) for count in class_counts]
    if weights:
        weights[0] *= float(empty_weight)
        for idx in range(1, len(weights)):
            weights[idx] *= float(positive_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _streaming_output_dim(
    *,
    category_vocab: list[str],
    keyboard_head_mode: str,
    keyboard_classes: list[tuple[str, ...]],
    button_head_mode: str,
    button_classes: list[tuple[str, ...]],
    mouse_head_mode: str,
    mouse_axis_classes: list[str],
) -> int:
    return (
        2
        + len(category_vocab)
        + (len(keyboard_classes) if keyboard_head_mode == "softmax" else 0)
        + (len(button_classes) if button_head_mode == "softmax" else 0)
        + (2 * len(mouse_axis_classes) if mouse_head_mode == "axis_softmax" else 0)
    )


def _tokens(row: dict[str, Any]) -> list[str]:
    return list(row.get("ground_truth_tokens") or ["NOOP"])


def _action_history_len_from_config(config: dict[str, Any]) -> int:
    history_len = int(config.get("action_history_len", 0) or 0)
    if history_len < 0:
        raise ValueError("action_history_len must be non-negative")
    return history_len


def _base_record_features(row: dict[str, Any], *, feature_mode: str) -> list[float]:
    override = row.get("__streaming_idm_features")
    if override is not None:
        return [float(value) for value in override]
    return [float(value) for value in record_features(row, feature_mode=feature_mode)]


def _record_features_with_history(
    row: dict[str, Any],
    *,
    feature_mode: str,
    history_vocab: list[str],
    history_len: int,
    history: list[list[str]] | None = None,
    button_state: dict[str, float] | None = None,
) -> list[float]:
    values = [float(value) for value in record_features(row, feature_mode=feature_mode)]
    if history_len <= 0:
        return values
    if history is None or button_state is None:
        raise ValueError("action-history features require causal history state")
    return values + _history_vector(history, button_state, history_vocab, history_len=history_len)


def _supports_closed_loop_state_context(feature_mode: str) -> bool:
    return "prior_action" in str(feature_mode)


def _closed_loop_state_context_enabled(config: dict[str, Any], stats: dict[str, Any]) -> bool:
    enabled = bool(config.get("closed_loop_state_context", False)) or (
        str(config.get("state_context_source", "")).replace("-", "_").lower() == "predicted_closed_loop"
    )
    if not enabled:
        return False
    feature_mode = str(stats.get("feature_mode", config.get("feature_mode", "")))
    if not _supports_closed_loop_state_context(feature_mode):
        raise ValueError(
            "closed_loop_state_context requires an IDM feature_mode with prior_action context; "
            f"got {feature_mode!r}"
        )
    return True


def _prior_state_tokens(keys: set[str], buttons: set[str]) -> list[str]:
    tokens = [f"KEY_DOWN_{key}" for key in sorted(keys)]
    tokens.extend(f"MOUSE_{button}_DOWN" for button in sorted(buttons))
    return tokens or ["NOOP"]


def _transition_seen(tokens: Iterable[str], *, prefixes: tuple[str, ...], suffixes: tuple[str, ...] = ()) -> bool:
    for raw_token in tokens:
        token = str(raw_token)
        if prefixes and token.startswith(prefixes):
            return True
        if suffixes and token.startswith("MOUSE_") and token.endswith(suffixes):
            return True
    return False


def _age_value(value: int | None) -> int | None:
    return None if value is None else max(0, int(value))


class _ClosedLoopStateContextTracker:
    """Causal state-context replacement for event-state IDM target inference.

    The event-state materializer adds ``prior_*`` fields before each action bin.
    On target shards those fields can encode earlier target ground-truth actions.
    This tracker rebuilds the same fields from already predicted tokens (and,
    optionally, train-split seed rows) so evaluation does not consume target
    action labels through feature context.
    """

    schema = "streaming_idm_predicted_closed_loop_state_context.v1"
    source = "predicted_closed_loop_before_current_event_bin"

    def __init__(self) -> None:
        self.key_states: dict[str, set[str]] = {}
        self.button_states: dict[str, set[str]] = {}
        self.key_hold_bins: dict[str, dict[str, int]] = {}
        self.button_hold_bins: dict[str, dict[str, int]] = {}
        self.key_transition_age: dict[str, int | None] = {}
        self.button_transition_age: dict[str, int | None] = {}
        self.previous_event_tokens: dict[str, list[str]] = {}
        self.seed_rows = 0

    def row_with_prior_context(self, row: dict[str, Any]) -> dict[str, Any]:
        recording_id = str(row.get("recording_id", ""))
        keys = self.key_states.setdefault(recording_id, set())
        buttons = self.button_states.setdefault(recording_id, set())
        key_durations = self.key_hold_bins.setdefault(recording_id, {})
        button_durations = self.button_hold_bins.setdefault(recording_id, {})
        out_row = dict(row)
        out_row["prior_action_tokens"] = _prior_state_tokens(keys, buttons)
        out_row["prior_action_source"] = self.source
        out_row["prior_key_hold_bins"] = {key: int(key_durations.get(key, 0)) for key in sorted(keys)}
        out_row["prior_button_hold_bins"] = {button: int(button_durations.get(button, 0)) for button in sorted(buttons)}
        out_row["prior_since_key_transition_bins"] = _age_value(self.key_transition_age.get(recording_id))
        out_row["prior_since_button_transition_bins"] = _age_value(self.button_transition_age.get(recording_id))
        out_row["previous_event_tokens"] = self.previous_event_tokens.get(recording_id) or ["NOOP"]
        out_row["state_context_schema"] = self.schema
        return out_row

    def observe_tokens(self, row: dict[str, Any], tokens: Iterable[str]) -> None:
        recording_id = str(row.get("recording_id", ""))
        old_keys = set(self.key_states.setdefault(recording_id, set()))
        old_buttons = set(self.button_states.setdefault(recording_id, set()))
        key_durations = dict(self.key_hold_bins.setdefault(recording_id, {}))
        button_durations = dict(self.button_hold_bins.setdefault(recording_id, {}))
        token_list = [str(token) for token in tokens if str(token)]
        _state_tokens, next_keys, next_buttons = state_tokens_from_event_tokens(
            token_list,
            pressed_keys=old_keys,
            pressed_buttons=old_buttons,
            mouse_emit_mode="decompose",
            mouse_max_tokens_per_axis=32,
        )
        key_event = _transition_seen(token_list, prefixes=("KEY_PRESS_", "KEY_RELEASE_", "KEY_DOWN_"))
        button_event = _transition_seen(token_list, prefixes=(), suffixes=("_DOWN", "_UP"))
        self.key_states[recording_id] = set(next_keys)
        self.button_states[recording_id] = set(next_buttons)
        self.key_hold_bins[recording_id] = {
            key: int(key_durations.get(key, 0)) + 1 if key in old_keys else 1
            for key in sorted(next_keys)
        }
        self.button_hold_bins[recording_id] = {
            button: int(button_durations.get(button, 0)) + 1 if button in old_buttons else 1
            for button in sorted(next_buttons)
        }
        self.key_transition_age[recording_id] = 0 if key_event else (
            None if self.key_transition_age.get(recording_id) is None else int(self.key_transition_age[recording_id] or 0) + 1
        )
        self.button_transition_age[recording_id] = 0 if button_event else (
            None
            if self.button_transition_age.get(recording_id) is None
            else int(self.button_transition_age[recording_id] or 0) + 1
        )
        self.previous_event_tokens[recording_id] = token_list or ["NOOP"]


def _seed_closed_loop_state_context_from_records(
    record_paths: Sequence[Path],
    *,
    max_examples: int | None = None,
) -> _ClosedLoopStateContextTracker:
    tracker = _ClosedLoopStateContextTracker()
    seen = 0
    for row in _iter_records(record_paths):
        if max_examples is not None and seen >= int(max_examples):
            break
        tracker.observe_tokens(row, row.get("ground_truth_tokens", []) or ["NOOP"])
        seen += 1
    tracker.seed_rows = seen
    return tracker


def _history_dim(*, history_vocab: list[str], history_len: int) -> int:
    if history_len <= 0:
        return 0
    return (2 * history_len) + (len(history_vocab) * history_len) + 3


def _ensure_history_state(
    histories: dict[str, list[list[str]]],
    button_states: dict[str, dict[str, float]],
    recording_id: str,
) -> tuple[list[list[str]], dict[str, float]]:
    if recording_id not in histories:
        histories[recording_id] = []
        button_states[recording_id] = _empty_button_state()
    return histories[recording_id], button_states[recording_id]


def _action_history_parallel_by_path(config: dict[str, Any]) -> bool:
    return bool(config.get("action_history_parallel_by_path", config.get("action_history_parallel_prediction_by_path", False)))


def _action_history_seed_state_from_records(
    record_paths: Sequence[str | Path],
    *,
    history_len: int,
) -> dict[str, Any]:
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    if history_len <= 0:
        return {"schema": "streaming_idm_action_history_seed_state.v1", "history_len": 0, "histories": {}, "button_states": {}}
    for row in _iter_records(record_paths):
        history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
        _append_history(history, button_state, _tokens(row), history_len=history_len)
    return {
        "schema": "streaming_idm_action_history_seed_state.v1",
        "history_len": int(history_len),
        "recordings": len(histories),
        "histories": histories,
        "button_states": button_states,
    }


def _action_history_seed_state_partition(path: str | Path, history_len: int) -> dict[str, Any]:
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    latest_timestamp_by_recording: dict[str, int] = {}
    if history_len <= 0:
        return {"schema": "streaming_idm_action_history_seed_partition.v1", "path": str(path), "rows": 0, "recordings": 0}
    rows = 0
    for row in iter_jsonl(path):
        recording_id = str(row.get("recording_id", ""))
        history, button_state = _ensure_history_state(histories, button_states, recording_id)
        _append_history(history, button_state, _tokens(row), history_len=history_len)
        try:
            latest_timestamp_by_recording[recording_id] = int(row.get("timestamp_ns", 0))
        except Exception:
            latest_timestamp_by_recording[recording_id] = rows
        rows += 1
    return {
        "schema": "streaming_idm_action_history_seed_partition.v1",
        "path": str(path),
        "rows": int(rows),
        "recordings": len(histories),
        "histories": histories,
        "button_states": button_states,
        "latest_timestamp_by_recording": latest_timestamp_by_recording,
    }


def _action_history_seed_state_from_records_parallel(
    record_paths: Sequence[str | Path],
    *,
    history_len: int,
    workers: int,
) -> dict[str, Any]:
    paths = list(record_paths)
    if history_len <= 0:
        return {"schema": "streaming_idm_action_history_seed_state.v1", "history_len": 0, "histories": {}, "button_states": {}}
    if len(paths) <= 1 or workers <= 1:
        return _action_history_seed_state_from_records(paths, history_len=history_len)
    partition_count = min(int(workers), len(paths))
    partitions: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=partition_count) as pool:
        futures = {pool.submit(_action_history_seed_state_partition, path, int(history_len)): path for path in paths}
        for future in as_completed(futures):
            partitions.append(future.result())
    merged_histories: dict[str, list[list[str]]] = {}
    merged_button_states: dict[str, dict[str, float]] = {}
    latest_timestamp: dict[str, int] = {}
    for partition in partitions:
        histories = dict(partition.get("histories", {}))
        button_states = dict(partition.get("button_states", {}))
        timestamps = dict(partition.get("latest_timestamp_by_recording", {}))
        for recording_id, history_rows in histories.items():
            timestamp = int(timestamps.get(recording_id, 0))
            if recording_id not in latest_timestamp or timestamp >= latest_timestamp[recording_id]:
                latest_timestamp[recording_id] = timestamp
                merged_histories[recording_id] = [
                    [str(token) for token in tokens]
                    for tokens in history_rows[-int(history_len) :]
                    if isinstance(tokens, list)
                ]
                state = button_states.get(recording_id, {})
                merged_button_states[recording_id] = (
                    {str(key): float(value) for key, value in state.items()}
                    if isinstance(state, dict)
                    else _empty_button_state()
                )
    return {
        "schema": "streaming_idm_action_history_seed_state.v1",
        "history_len": int(history_len),
        "recordings": len(merged_histories),
        "histories": merged_histories,
        "button_states": merged_button_states,
        "parallel": True,
        "workers": int(partition_count),
        "partition_count": len(partitions),
        "rows": sum(int(partition.get("rows") or 0) for partition in partitions),
    }


def _empty_action_history_seed_state(*, history_len: int) -> dict[str, Any]:
    return {
        "schema": "streaming_idm_action_history_seed_state.v1",
        "history_len": max(0, int(history_len)),
        "recordings": 0,
        "histories": {},
        "button_states": {},
    }


def _action_history_seed_state_from_stats(stats: dict[str, Any], *, history_len: int) -> dict[str, Any]:
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    if history_len <= 0:
        return _empty_action_history_seed_state(history_len=history_len)
    for recording_id, tokens in dict(stats.get("last_tokens_by_recording", {})).items():
        history, button_state = _ensure_history_state(histories, button_states, str(recording_id))
        _append_history(history, button_state, [str(token) for token in tokens], history_len=history_len)
    return {
        "schema": "streaming_idm_action_history_seed_state.v1",
        "history_len": int(history_len),
        "recordings": len(histories),
        "histories": histories,
        "button_states": button_states,
        "source": "stats_tail",
    }


def _action_history_seed_state_for_parallel_prediction(
    config: dict[str, Any],
    *,
    train_paths: Sequence[str | Path],
    history_len: int,
    stats: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    mode = str(
        config.get(
            "action_history_seed_state_mode",
            config.get("parallel_action_history_seed_state_mode", "train_scan"),
        )
    ).replace("-", "_").lower()
    if mode in {"empty", "zero", "none", "target_start"}:
        return _empty_action_history_seed_state(history_len=history_len), "empty"
    if mode in {"stats_tail", "stats", "last_token"}:
        if stats is None:
            raise ValueError("action_history_seed_state_mode=stats_tail requires streaming stats")
        return _action_history_seed_state_from_stats(stats, history_len=history_len), "stats_tail"
    if mode in {"train_scan", "parent_train_scan", "train_tail"}:
        workers = int(config.get("action_history_seed_state_workers", 1) or 1)
        if bool(config.get("action_history_seed_state_parallel_by_path", workers > 1)) and workers > 1:
            return (
                _action_history_seed_state_from_records_parallel(
                    train_paths,
                    history_len=history_len,
                    workers=workers,
                ),
                "parent_train_scan_parallel",
            )
        return _action_history_seed_state_from_records(train_paths, history_len=history_len), "parent_train_scan"
    raise ValueError(
        "action_history_seed_state_mode must be one of empty/zero/none/target_start "
        "stats_tail or train_scan/parent_train_scan/train_tail"
    )


def _load_action_history_seed_state(seed_state: Any) -> tuple[dict[str, list[list[str]]], dict[str, dict[str, float]]]:
    if not isinstance(seed_state, dict):
        return {}, {}
    raw_histories = seed_state.get("histories", {})
    raw_button_states = seed_state.get("button_states", {})
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    if isinstance(raw_histories, dict):
        for recording_id, rows in raw_histories.items():
            if not isinstance(rows, list):
                continue
            histories[str(recording_id)] = [[str(token) for token in tokens] for tokens in rows if isinstance(tokens, list)]
    if isinstance(raw_button_states, dict):
        for recording_id, state in raw_button_states.items():
            if isinstance(state, dict):
                button_states[str(recording_id)] = {str(key): float(value) for key, value in state.items()}
    for recording_id in histories:
        button_states.setdefault(recording_id, _empty_button_state())
    return histories, button_states


def _empty_stats_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "mean": [],
        "m2": [],
        "category_counts": Counter(),
        "keyboard_class_counts": Counter(),
        "button_class_counts": Counter(),
        "sequence_counts": Counter(),
        "last_tokens_by_recording": {},
        "last_tokens_by_game": {},
        "source_ids": set(),
        "resolution_tiers": set(),
        "split_names": set(),
        "eval_split_tags": set(),
        "fingerprint_parts": [],
    }


def _merge_feature_moments(
    *,
    count_a: int,
    mean_a: list[float],
    m2_a: list[float],
    count_b: int,
    mean_b: list[float],
    m2_b: list[float],
) -> tuple[int, list[float], list[float]]:
    if count_b == 0:
        return count_a, mean_a, m2_a
    if count_a == 0:
        return count_b, list(mean_b), list(m2_b)
    if len(mean_a) != len(mean_b):
        raise ValueError(f"inconsistent feature dimension across record partitions: {len(mean_a)} != {len(mean_b)}")
    total = count_a + count_b
    merged_mean: list[float] = []
    merged_m2: list[float] = []
    for idx, (a_mean, b_mean) in enumerate(zip(mean_a, mean_b)):
        delta = b_mean - a_mean
        mean = a_mean + delta * (count_b / total)
        m2 = m2_a[idx] + m2_b[idx] + (delta * delta) * count_a * count_b / total
        merged_mean.append(mean)
        merged_m2.append(m2)
    return total, merged_mean, merged_m2


def _latest_token_map_update(target: dict[str, tuple[int, list[str]]], key: str, timestamp_ns: Any, tokens: list[str]) -> None:
    try:
        timestamp = int(timestamp_ns)
    except (TypeError, ValueError):
        timestamp = -1
    previous = target.get(key)
    if previous is None or timestamp >= previous[0]:
        target[key] = (timestamp, tokens)


def _scan_stats_partition(path: str | Path, feature_mode: str, *, residual_mouse: bool = False) -> dict[str, Any]:
    count = 0
    mean: list[float] = []
    m2: list[float] = []
    category_counts: Counter[str] = Counter()
    keyboard_class_counts: Counter[str] = Counter()
    button_class_counts: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    last_tokens_by_recording: dict[str, tuple[int, list[str]]] = {}
    last_tokens_by_game: dict[str, tuple[int, list[str]]] = {}
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    fingerprint = hashlib.sha256()
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    for row in iter_jsonl(path):
        mouse_baseline = _mouse_baseline_for_row(
            row,
            last_by_recording=last_mouse_by_recording,
            last_by_game=last_mouse_by_game,
            fallback=last_mouse_fallback,
        )
        features = _features_with_optional_mouse_baseline(
            row,
            _base_record_features(row, feature_mode=feature_mode),
            residual_mouse=residual_mouse,
            baseline=mouse_baseline,
        )
        if not mean:
            mean = [0.0 for _ in features]
            m2 = [0.0 for _ in features]
        if len(features) != len(mean):
            raise ValueError(f"inconsistent feature dimension in {path}: {len(features)} != {len(mean)}")
        count += 1
        for idx, value in enumerate(features):
            delta = value - mean[idx]
            mean[idx] += delta / count
            m2[idx] += delta * (value - mean[idx])
        for token in row.get("ground_truth_tokens", []):
            token = str(token)
            if _is_category_token(token):
                category_counts[token] += 1
        keyboard_class_counts[_label_key(_keyboard_label(row))] += 1
        button_class_counts[_label_key(_button_label(row))] += 1
        tokens = _tokens(row)
        sequence_counts[tuple(tokens)] += 1
        timestamp_ns = row.get("timestamp_ns")
        _latest_token_map_update(last_tokens_by_recording, str(row.get("recording_id", "")), timestamp_ns, tokens)
        _latest_token_map_update(last_tokens_by_game, str(row.get("game", "unknown")), timestamp_ns, tokens)
        if residual_mouse:
            last_mouse_fallback = _update_mouse_baseline_state(
                row,
                target_mouse_delta(row, mode="sum"),
                last_by_recording=last_mouse_by_recording,
                last_by_game=last_mouse_by_game,
            )
        if row.get("source_id") is not None:
            source_ids.add(str(row["source_id"]))
        if row.get("resolution_tier") is not None:
            resolution_tiers.add(str(row["resolution_tier"]))
        if row.get("split") is not None:
            split_names.add(str(row["split"]))
        for tag in row.get("eval_split_tags", []) or []:
            eval_split_tags.add(str(tag))
        fingerprint.update(
            json.dumps(
                {
                    "sequence_id": row.get("sequence_id"),
                    "tokens": row.get("ground_truth_tokens", []),
                    "features": features,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        fingerprint.update(b"\n")
    return {
        "path": str(path),
        "count": count,
        "mean": mean,
        "m2": m2,
        "category_counts": dict(category_counts),
        "keyboard_class_counts": dict(keyboard_class_counts),
        "button_class_counts": dict(button_class_counts),
        "sequence_counts": dict(sequence_counts),
        "last_tokens_by_recording": last_tokens_by_recording,
        "last_tokens_by_game": last_tokens_by_game,
        "source_ids": source_ids,
        "resolution_tiers": resolution_tiers,
        "split_names": split_names,
        "eval_split_tags": eval_split_tags,
        "fingerprint": fingerprint.hexdigest(),
    }


def _history_vocab_from_counts(counts: Counter[str], min_count: int) -> list[str]:
    return _category_vocab_from_counts(dict(counts), min_count)


def _scan_streaming_idm_stats_with_action_history(
    paths: Sequence[Path],
    *,
    train_records: str | Path | Sequence[str | Path],
    feature_mode: str,
    categorical_min_count: int,
    action_history_len: int,
    residual_mouse: bool = False,
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    keyboard_class_counts: Counter[str] = Counter()
    button_class_counts: Counter[str] = Counter()
    for path in paths:
        for row in iter_jsonl(path):
            for token in row.get("ground_truth_tokens", []):
                token = str(token)
                if _is_category_token(token):
                    category_counts[token] += 1
    history_vocab = _history_vocab_from_counts(category_counts, categorical_min_count)

    count = 0
    mean: list[float] = []
    m2: list[float] = []
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    last_tokens_by_recording: dict[str, tuple[int, list[str]]] = {}
    last_tokens_by_game: dict[str, tuple[int, list[str]]] = {}
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    fingerprint = hashlib.sha256()
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    fingerprint_parts: list[dict[str, Any]] = []
    for path in paths:
        part_count = 0
        part_fingerprint = hashlib.sha256()
        for row in iter_jsonl(path):
            recording_id = str(row.get("recording_id", ""))
            history, button_state = _ensure_history_state(histories, button_states, recording_id)
            mouse_baseline = _mouse_baseline_for_row(
                row,
                last_by_recording=last_mouse_by_recording,
                last_by_game=last_mouse_by_game,
                fallback=last_mouse_fallback,
            )
            features = _features_with_optional_mouse_baseline(
                row,
                _record_features_with_history(
                    row,
                    feature_mode=feature_mode,
                    history_vocab=history_vocab,
                    history_len=action_history_len,
                    history=history,
                    button_state=button_state,
                ),
                residual_mouse=residual_mouse,
                baseline=mouse_baseline,
            )
            if not mean:
                mean = [0.0 for _ in features]
                m2 = [0.0 for _ in features]
            if len(features) != len(mean):
                raise ValueError(f"inconsistent feature dimension in {path}: {len(features)} != {len(mean)}")
            count += 1
            part_count += 1
            for idx, value in enumerate(features):
                delta = value - mean[idx]
                mean[idx] += delta / count
                m2[idx] += delta * (value - mean[idx])
            tokens = _tokens(row)
            keyboard_class_counts[_label_key(_keyboard_label(row))] += 1
            button_class_counts[_label_key(_button_label(row))] += 1
            sequence_counts[tuple(tokens)] += 1
            timestamp_ns = row.get("timestamp_ns")
            _latest_token_map_update(last_tokens_by_recording, recording_id, timestamp_ns, tokens)
            _latest_token_map_update(last_tokens_by_game, str(row.get("game", "unknown")), timestamp_ns, tokens)
            if row.get("source_id") is not None:
                source_ids.add(str(row["source_id"]))
            if row.get("resolution_tier") is not None:
                resolution_tiers.add(str(row["resolution_tier"]))
            if row.get("split") is not None:
                split_names.add(str(row["split"]))
            for tag in row.get("eval_split_tags", []) or []:
                eval_split_tags.add(str(tag))
            fingerprint_row = {
                "sequence_id": row.get("sequence_id"),
                "tokens": row.get("ground_truth_tokens", []),
                "features": features,
            }
            encoded = json.dumps(fingerprint_row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            fingerprint.update(encoded)
            fingerprint.update(b"\n")
            part_fingerprint.update(encoded)
            part_fingerprint.update(b"\n")
            _append_history(history, button_state, tokens, history_len=action_history_len)
            if residual_mouse:
                last_mouse_fallback = _update_mouse_baseline_state(
                    row,
                    target_mouse_delta(row, mode="sum"),
                    last_by_recording=last_mouse_by_recording,
                    last_by_game=last_mouse_by_game,
                )
        fingerprint_parts.append({"path": str(path), "count": part_count, "fingerprint": part_fingerprint.hexdigest()})

    if count == 0:
        raise ValueError(f"no training rows found in {train_records}")
    std = [(m2[idx] / max(1, count - 1)) ** 0.5 or 1.0 for idx in range(len(mean))]
    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "feature_mode": feature_mode,
                "action_history_len": action_history_len,
                "action_history_vocab": history_vocab,
                "residual_mouse": bool(residual_mouse),
                "partitions": fingerprint_parts,
                "rows": fingerprint.hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "streaming_idm_stats.v1",
        "train_records": str(train_records) if isinstance(train_records, (str, Path)) else [str(path) for path in train_records],
        "num_examples": count,
        "feature_mode": feature_mode,
        "input_dim": len(mean),
        "mean": mean,
        "std": std,
        "category_vocab": _category_vocab_from_counts(dict(category_counts), categorical_min_count),
        "category_counts": dict(sorted(dict(category_counts).items())),
        "keyboard_class_counts": dict(sorted(dict(keyboard_class_counts).items())),
        "button_class_counts": dict(sorted(dict(button_class_counts).items())),
        "global_majority_tokens": list(sequence_counts.most_common(1)[0][0]) if sequence_counts else ["NOOP"],
        "last_tokens_by_recording": {key: list(tokens) for key, (_ts, tokens) in sorted(last_tokens_by_recording.items())},
        "last_tokens_by_game": {key: list(tokens) for key, (_ts, tokens) in sorted(last_tokens_by_game.items())},
        "source_ids": sorted(source_ids),
        "resolution_tiers": sorted(resolution_tiers),
        "split_names": sorted(split_names),
        "eval_split_tags": sorted(eval_split_tags),
        "dataset_fingerprint": dataset_fingerprint,
        "action_history_len": int(action_history_len),
        "action_history_vocab": history_vocab,
        "action_history_dim": _history_dim(history_vocab=history_vocab, history_len=action_history_len),
        "action_history_feedback": "teacher_forced_train",
        "action_history_parallel_by_path": False,
        "residual_mouse": bool(residual_mouse),
    }


def _scan_action_history_category_counts_partition(path: str | Path) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    count = 0
    for row in iter_jsonl(path):
        count += 1
        for token in row.get("ground_truth_tokens", []):
            token = str(token)
            if _is_category_token(token):
                category_counts[token] += 1
    return {"path": str(path), "count": count, "category_counts": dict(category_counts)}


def _scan_action_history_stats_partition(
    path: str | Path,
    feature_mode: str,
    history_vocab: list[str],
    action_history_len: int,
    residual_mouse: bool = False,
) -> dict[str, Any]:
    count = 0
    mean: list[float] = []
    m2: list[float] = []
    category_counts: Counter[str] = Counter()
    keyboard_class_counts: Counter[str] = Counter()
    button_class_counts: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    last_tokens_by_recording: dict[str, tuple[int, list[str]]] = {}
    last_tokens_by_game: dict[str, tuple[int, list[str]]] = {}
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    fingerprint = hashlib.sha256()
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    for row in iter_jsonl(path):
        recording_id = str(row.get("recording_id", ""))
        history, button_state = _ensure_history_state(histories, button_states, recording_id)
        mouse_baseline = _mouse_baseline_for_row(
            row,
            last_by_recording=last_mouse_by_recording,
            last_by_game=last_mouse_by_game,
            fallback=last_mouse_fallback,
        )
        features = _features_with_optional_mouse_baseline(
            row,
            _record_features_with_history(
                row,
                feature_mode=feature_mode,
                history_vocab=history_vocab,
                history_len=action_history_len,
                history=history,
                button_state=button_state,
            ),
            residual_mouse=residual_mouse,
            baseline=mouse_baseline,
        )
        if not mean:
            mean = [0.0 for _ in features]
            m2 = [0.0 for _ in features]
        if len(features) != len(mean):
            raise ValueError(f"inconsistent feature dimension in {path}: {len(features)} != {len(mean)}")
        count += 1
        for idx, value in enumerate(features):
            delta = value - mean[idx]
            mean[idx] += delta / count
            m2[idx] += delta * (value - mean[idx])
        for token in row.get("ground_truth_tokens", []):
            token = str(token)
            if _is_category_token(token):
                category_counts[token] += 1
        tokens = _tokens(row)
        keyboard_class_counts[_label_key(_keyboard_label(row))] += 1
        button_class_counts[_label_key(_button_label(row))] += 1
        sequence_counts[tuple(tokens)] += 1
        timestamp_ns = row.get("timestamp_ns")
        _latest_token_map_update(last_tokens_by_recording, recording_id, timestamp_ns, tokens)
        _latest_token_map_update(last_tokens_by_game, str(row.get("game", "unknown")), timestamp_ns, tokens)
        if row.get("source_id") is not None:
            source_ids.add(str(row["source_id"]))
        if row.get("resolution_tier") is not None:
            resolution_tiers.add(str(row["resolution_tier"]))
        if row.get("split") is not None:
            split_names.add(str(row["split"]))
        for tag in row.get("eval_split_tags", []) or []:
            eval_split_tags.add(str(tag))
        fingerprint.update(
            json.dumps(
                {
                    "sequence_id": row.get("sequence_id"),
                    "tokens": row.get("ground_truth_tokens", []),
                    "features": features,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        fingerprint.update(b"\n")
        _append_history(history, button_state, tokens, history_len=action_history_len)
        if residual_mouse:
            last_mouse_fallback = _update_mouse_baseline_state(
                row,
                target_mouse_delta(row, mode="sum"),
                last_by_recording=last_mouse_by_recording,
                last_by_game=last_mouse_by_game,
            )
    return {
        "path": str(path),
        "count": count,
        "mean": mean,
        "m2": m2,
        "category_counts": dict(category_counts),
        "keyboard_class_counts": dict(keyboard_class_counts),
        "button_class_counts": dict(button_class_counts),
        "sequence_counts": dict(sequence_counts),
        "last_tokens_by_recording": last_tokens_by_recording,
        "last_tokens_by_game": last_tokens_by_game,
        "source_ids": source_ids,
        "resolution_tiers": resolution_tiers,
        "split_names": split_names,
        "eval_split_tags": eval_split_tags,
        "fingerprint": fingerprint.hexdigest(),
    }


def _merge_action_history_stats_partitions(
    partitions: Iterable[dict[str, Any]],
    *,
    train_records: str | Path | Sequence[str | Path],
    feature_mode: str,
    categorical_min_count: int,
    action_history_len: int,
    history_vocab: list[str],
    residual_mouse: bool = False,
) -> dict[str, Any]:
    partition_list = list(partitions)
    merged = _merge_stats_partitions(
        partition_list,
        train_records=train_records,
        feature_mode=feature_mode,
        categorical_min_count=categorical_min_count,
        residual_mouse=residual_mouse,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "feature_mode": feature_mode,
                "action_history_len": int(action_history_len),
                "action_history_vocab": history_vocab,
                "residual_mouse": bool(residual_mouse),
                "partitions": sorted(
                    [
                        {"path": row.get("path", ""), "count": int(row.get("count", 0)), "fingerprint": row.get("fingerprint", "")}
                        for row in partition_list
                    ],
                    key=lambda row: str(row["path"]),
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    merged.update(
        {
            "dataset_fingerprint": fingerprint,
            "action_history_len": int(action_history_len),
            "action_history_vocab": history_vocab,
            "action_history_dim": _history_dim(history_vocab=history_vocab, history_len=action_history_len),
            "action_history_feedback": "teacher_forced_train",
            "action_history_parallel_by_path": True,
            "residual_mouse": bool(residual_mouse),
        }
    )
    return merged


def _scan_streaming_idm_stats_with_action_history_parallel(
    paths: Sequence[Path],
    *,
    train_records: str | Path | Sequence[str | Path],
    feature_mode: str,
    categorical_min_count: int,
    action_history_len: int,
    num_workers: int,
    residual_mouse: bool = False,
) -> dict[str, Any]:
    workers = min(max(1, int(num_workers)), len(paths))
    category_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_action_history_category_counts_partition, path): path for path in paths}
        for future in as_completed(futures):
            part = future.result()
            category_counts.update({str(k): int(v) for k, v in dict(part.get("category_counts", {})).items()})
    history_vocab = _history_vocab_from_counts(category_counts, categorical_min_count)
    partitions: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _scan_action_history_stats_partition,
                path,
                feature_mode,
                history_vocab,
                int(action_history_len),
                residual_mouse,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            partitions.append(future.result())
    return _merge_action_history_stats_partitions(
        partitions,
        train_records=train_records,
        feature_mode=feature_mode,
        categorical_min_count=categorical_min_count,
        action_history_len=int(action_history_len),
        history_vocab=history_vocab,
        residual_mouse=residual_mouse,
    )


def _merge_stats_partitions(
    partitions: Iterable[dict[str, Any]],
    *,
    train_records: str | Path | Sequence[str | Path],
    feature_mode: str,
    categorical_min_count: int,
    residual_mouse: bool = False,
) -> dict[str, Any]:
    acc = _empty_stats_accumulator()
    for part in partitions:
        count, mean, m2 = _merge_feature_moments(
            count_a=int(acc["count"]),
            mean_a=list(acc["mean"]),
            m2_a=list(acc["m2"]),
            count_b=int(part.get("count", 0)),
            mean_b=[float(value) for value in part.get("mean", [])],
            m2_b=[float(value) for value in part.get("m2", [])],
        )
        acc["count"] = count
        acc["mean"] = mean
        acc["m2"] = m2
        acc["category_counts"].update({str(k): int(v) for k, v in dict(part.get("category_counts", {})).items()})
        acc["keyboard_class_counts"].update(
            {str(k): int(v) for k, v in dict(part.get("keyboard_class_counts", {})).items()}
        )
        acc["button_class_counts"].update(
            {str(k): int(v) for k, v in dict(part.get("button_class_counts", {})).items()}
        )
        acc["sequence_counts"].update({tuple(k): int(v) for k, v in dict(part.get("sequence_counts", {})).items()})
        for key, value in dict(part.get("last_tokens_by_recording", {})).items():
            timestamp, tokens = value
            _latest_token_map_update(acc["last_tokens_by_recording"], str(key), timestamp, list(tokens))
        for key, value in dict(part.get("last_tokens_by_game", {})).items():
            timestamp, tokens = value
            _latest_token_map_update(acc["last_tokens_by_game"], str(key), timestamp, list(tokens))
        acc["source_ids"].update(str(value) for value in part.get("source_ids", set()))
        acc["resolution_tiers"].update(str(value) for value in part.get("resolution_tiers", set()))
        acc["split_names"].update(str(value) for value in part.get("split_names", set()))
        acc["eval_split_tags"].update(str(value) for value in part.get("eval_split_tags", set()))
        acc["fingerprint_parts"].append({"path": str(part.get("path", "")), "count": int(part.get("count", 0)), "fingerprint": str(part.get("fingerprint", ""))})

    count = int(acc["count"])
    if count == 0:
        raise ValueError(f"no training rows found in {train_records}")
    mean = [float(value) for value in acc["mean"]]
    m2 = [float(value) for value in acc["m2"]]
    std = [(m2[idx] / max(1, count - 1)) ** 0.5 or 1.0 for idx in range(len(mean))]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "feature_mode": feature_mode,
                "residual_mouse": bool(residual_mouse),
                "partitions": sorted(acc["fingerprint_parts"], key=lambda row: row["path"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    last_tokens_by_recording = {
        key: list(tokens)
        for key, (_timestamp, tokens) in sorted(dict(acc["last_tokens_by_recording"]).items())
    }
    last_tokens_by_game = {
        key: list(tokens)
        for key, (_timestamp, tokens) in sorted(dict(acc["last_tokens_by_game"]).items())
    }
    return {
        "schema": "streaming_idm_stats.v1",
        "train_records": str(train_records) if isinstance(train_records, (str, Path)) else [str(path) for path in train_records],
        "num_examples": count,
        "feature_mode": feature_mode,
        "input_dim": len(mean),
        "mean": mean,
        "std": std,
        "category_vocab": _category_vocab_from_counts(dict(acc["category_counts"]), categorical_min_count),
        "category_counts": dict(sorted(dict(acc["category_counts"]).items())),
        "keyboard_class_counts": dict(sorted(dict(acc["keyboard_class_counts"]).items())),
        "button_class_counts": dict(sorted(dict(acc["button_class_counts"]).items())),
        "global_majority_tokens": list(acc["sequence_counts"].most_common(1)[0][0]) if acc["sequence_counts"] else ["NOOP"],
        "last_tokens_by_recording": last_tokens_by_recording,
        "last_tokens_by_game": last_tokens_by_game,
        "source_ids": sorted(acc["source_ids"]),
        "resolution_tiers": sorted(acc["resolution_tiers"]),
        "split_names": sorted(acc["split_names"]),
        "eval_split_tags": sorted(acc["eval_split_tags"]),
        "dataset_fingerprint": fingerprint,
        "residual_mouse": bool(residual_mouse),
    }


def scan_streaming_idm_stats(
    train_records: str | Path | Sequence[str | Path],
    *,
    feature_mode: str,
    categorical_min_count: int = 1,
    num_workers: int = 1,
    action_history_len: int = 0,
    action_history_parallel_by_path: bool = False,
    residual_mouse: bool = False,
) -> dict[str, Any]:
    paths = _record_paths_from_value(train_records)
    if int(action_history_len) > 0:
        if len(paths) > 1 and int(num_workers) > 1 and action_history_parallel_by_path:
            return _scan_streaming_idm_stats_with_action_history_parallel(
                paths,
                train_records=train_records,
                feature_mode=feature_mode,
                categorical_min_count=categorical_min_count,
                action_history_len=int(action_history_len),
                num_workers=int(num_workers),
                residual_mouse=residual_mouse,
            )
        return _scan_streaming_idm_stats_with_action_history(
            paths,
            train_records=train_records,
            feature_mode=feature_mode,
            categorical_min_count=categorical_min_count,
            action_history_len=int(action_history_len),
            residual_mouse=residual_mouse,
        )
    if len(paths) > 1 and int(num_workers) > 1:
        workers = min(int(num_workers), len(paths))
        partitions: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_scan_stats_partition, path, feature_mode, residual_mouse=residual_mouse): path for path in paths}
            for future in as_completed(futures):
                partitions.append(future.result())
        return _merge_stats_partitions(
            partitions,
            train_records=train_records,
            feature_mode=feature_mode,
            categorical_min_count=categorical_min_count,
            residual_mouse=residual_mouse,
        )
    return _merge_stats_partitions(
        (_scan_stats_partition(path, feature_mode, residual_mouse=residual_mouse) for path in paths),
        train_records=train_records,
        feature_mode=feature_mode,
        categorical_min_count=categorical_min_count,
        residual_mouse=residual_mouse,
    )


def scan_streaming_idm_stats_from_config(config: dict[str, Any]) -> dict[str, Any]:
    train_record_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    return scan_streaming_idm_stats(
        train_record_paths if len(train_record_paths) > 1 else train_record_paths[0],
        feature_mode=str(config.get("feature_mode", "summary_compact_grid8_shift_surface_time")),
        categorical_min_count=int(config.get("categorical_min_count", 1)),
        num_workers=int(config.get("precompute_num_workers", config.get("stats_num_workers", 1))),
        action_history_len=_action_history_len_from_config(config),
        action_history_parallel_by_path=_action_history_parallel_by_path(config),
        residual_mouse=_residual_mouse_from_mode(_mouse_target_mode(config)),
    )



def _normalizer_tensors(torch, *, mean: list[float], std: list[float], device: str):
    return (
        torch.tensor(mean, dtype=torch.float32, device=device),
        torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6),
    )


def _batch_features(
    torch,
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    mean: list[float],
    std: list[float],
    device: str,
    mean_t=None,
    std_t=None,
):
    xs = [_base_record_features(row, feature_mode=feature_mode) for row in rows]
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    if mean_t is None or std_t is None:
        mean_t, std_t = _normalizer_tensors(torch, mean=mean, std=std, device=device)
    return (x - mean_t) / std_t


def _category_targets(torch, rows: list[dict[str, Any]], vocab: list[str], *, device: str, vocab_index: dict[str, int] | None = None):
    if vocab_index is None:
        vocab_index = {token: idx for idx, token in enumerate(vocab)}
    y = torch.zeros((len(rows), len(vocab)), dtype=torch.float32, device=device)
    for row_idx, row in enumerate(rows):
        for token in set(row.get("ground_truth_tokens", [])):
            idx = vocab_index.get(str(token))
            if idx is not None:
                y[row_idx, idx] = 1.0
    return y


def _mouse_target_mode(config: dict[str, Any]) -> str:
    return str(config.get("mouse_target_mode", "mean"))


def _residual_mouse_from_mode(mouse_target_mode: str) -> bool:
    return str(mouse_target_mode).replace("-", "_").lower() == "residual_last_seen"


def _base_mouse_target_mode(mouse_target_mode: str) -> str:
    return "sum" if _residual_mouse_from_mode(mouse_target_mode) else str(mouse_target_mode)


def _mouse_delta_from_tokens(tokens: Sequence[str]) -> tuple[float, float]:
    return target_mouse_delta({"ground_truth_tokens": list(tokens)}, mode="sum")


def _mouse_baseline_for_row(
    row: dict[str, Any],
    *,
    last_by_recording: dict[str, tuple[float, float]],
    last_by_game: dict[str, tuple[float, float]],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    recording_id = str(row.get("recording_id", ""))
    game = str(row.get("game", "unknown"))
    return last_by_recording.get(recording_id) or last_by_game.get(game) or fallback


def _update_mouse_baseline_state(
    row: dict[str, Any],
    delta: tuple[float, float],
    *,
    last_by_recording: dict[str, tuple[float, float]],
    last_by_game: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    value = (float(delta[0]), float(delta[1]))
    last_by_recording[str(row.get("recording_id", ""))] = value
    last_by_game[str(row.get("game", "unknown"))] = value
    return value


def _seed_streaming_mouse_delta_state(
    record_paths: str | Path | Sequence[str | Path],
    *,
    num_workers: int = 1,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float]]:
    paths = _record_paths_from_value(record_paths)
    if len(paths) > 1 and int(num_workers) > 1:
        return _seed_streaming_mouse_delta_state_parallel(paths, num_workers=int(num_workers))
    last_by_recording: dict[str, tuple[float, float]] = {}
    last_by_game: dict[str, tuple[float, float]] = {}
    fallback = (0.0, 0.0)
    for path in paths:
        for row in iter_jsonl(path):
            fallback = _update_mouse_baseline_state(
                row,
                target_mouse_delta(row, mode="sum"),
                last_by_recording=last_by_recording,
                last_by_game=last_by_game,
            )
    return last_by_recording, last_by_game, fallback


def _timestamp_for_seed_row(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("timestamp_ns", fallback))
    except Exception:
        return int(fallback)


def _seed_streaming_mouse_delta_state_partition(path: str | Path) -> dict[str, Any]:
    last_by_recording: dict[str, tuple[int, tuple[float, float]]] = {}
    last_by_game: dict[str, tuple[int, tuple[float, float]]] = {}
    fallback_ts = -1
    fallback = (0.0, 0.0)
    rows = 0
    for row in iter_jsonl(path):
        timestamp = _timestamp_for_seed_row(row, rows)
        value = target_mouse_delta(row, mode="sum")
        value = (float(value[0]), float(value[1]))
        recording_id = str(row.get("recording_id", ""))
        game = str(row.get("game", "unknown"))
        if recording_id not in last_by_recording or timestamp >= last_by_recording[recording_id][0]:
            last_by_recording[recording_id] = (timestamp, value)
        if game not in last_by_game or timestamp >= last_by_game[game][0]:
            last_by_game[game] = (timestamp, value)
        if timestamp >= fallback_ts:
            fallback_ts = timestamp
            fallback = value
        rows += 1
    return {
        "path": str(path),
        "rows": int(rows),
        "last_by_recording": {key: [ts, value[0], value[1]] for key, (ts, value) in last_by_recording.items()},
        "last_by_game": {key: [ts, value[0], value[1]] for key, (ts, value) in last_by_game.items()},
        "fallback": [fallback_ts, fallback[0], fallback[1]],
    }


def _seed_streaming_mouse_delta_state_parallel(
    record_paths: Sequence[str | Path],
    *,
    num_workers: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float]]:
    paths = list(record_paths)
    workers = min(max(1, int(num_workers)), len(paths))
    partitions: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_seed_streaming_mouse_delta_state_partition, path): path for path in paths}
        for future in as_completed(futures):
            partitions.append(future.result())
    merged_recording: dict[str, tuple[int, tuple[float, float]]] = {}
    merged_game: dict[str, tuple[int, tuple[float, float]]] = {}
    fallback_ts = -1
    fallback = (0.0, 0.0)
    for partition in partitions:
        for key, raw in dict(partition.get("last_by_recording", {})).items():
            timestamp, dx, dy = int(raw[0]), float(raw[1]), float(raw[2])
            if key not in merged_recording or timestamp >= merged_recording[key][0]:
                merged_recording[key] = (timestamp, (dx, dy))
        for key, raw in dict(partition.get("last_by_game", {})).items():
            timestamp, dx, dy = int(raw[0]), float(raw[1]), float(raw[2])
            if key not in merged_game or timestamp >= merged_game[key][0]:
                merged_game[key] = (timestamp, (dx, dy))
        raw_fallback = partition.get("fallback", [-1, 0.0, 0.0])
        timestamp, dx, dy = int(raw_fallback[0]), float(raw_fallback[1]), float(raw_fallback[2])
        if timestamp >= fallback_ts:
            fallback_ts = timestamp
            fallback = (dx, dy)
    return (
        {key: value for key, (_timestamp, value) in merged_recording.items()},
        {key: value for key, (_timestamp, value) in merged_game.items()},
        fallback,
    )


def _streaming_mouse_seed_state_payload(
    *,
    last_by_recording: dict[str, tuple[float, float]],
    last_by_game: dict[str, tuple[float, float]],
    fallback: tuple[float, float],
    source: str,
    workers: int,
) -> dict[str, Any]:
    return {
        "schema": "streaming_idm_mouse_seed_state.v1",
        "source": str(source),
        "workers": int(workers),
        "recordings": len(last_by_recording),
        "games": len(last_by_game),
        "last_by_recording": {key: [float(value[0]), float(value[1])] for key, value in last_by_recording.items()},
        "last_by_game": {key: [float(value[0]), float(value[1])] for key, value in last_by_game.items()},
        "fallback": [float(fallback[0]), float(fallback[1])],
    }


def _load_streaming_mouse_seed_state(
    seed_state: Any,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float]] | None:
    if not isinstance(seed_state, dict):
        return None
    raw_recording = seed_state.get("last_by_recording", {})
    raw_game = seed_state.get("last_by_game", {})
    raw_fallback = seed_state.get("fallback", [0.0, 0.0])
    if not isinstance(raw_recording, dict) or not isinstance(raw_game, dict):
        return None
    last_by_recording = {
        str(key): (float(value[0]), float(value[1]))
        for key, value in raw_recording.items()
        if isinstance(value, (list, tuple)) and len(value) >= 2
    }
    last_by_game = {
        str(key): (float(value[0]), float(value[1]))
        for key, value in raw_game.items()
        if isinstance(value, (list, tuple)) and len(value) >= 2
    }
    fallback = (
        float(raw_fallback[0]) if isinstance(raw_fallback, (list, tuple)) and len(raw_fallback) >= 2 else 0.0,
        float(raw_fallback[1]) if isinstance(raw_fallback, (list, tuple)) and len(raw_fallback) >= 2 else 0.0,
    )
    return last_by_recording, last_by_game, fallback


def _streaming_mouse_seed_state_for_parallel_prediction(
    config: dict[str, Any],
    *,
    train_paths: Sequence[str | Path],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(config.get("streaming_mouse_seed_state_mode", config.get("action_history_seed_state_mode", "train_scan"))).replace(
        "-",
        "_",
    ).lower()
    if mode in {"empty", "zero", "none", "target_start"}:
        return _streaming_mouse_seed_state_payload(
            last_by_recording={},
            last_by_game={},
            fallback=(0.0, 0.0),
            source="empty",
            workers=0,
        )
    if mode in {"stats_tail", "stats", "last_token"}:
        if stats is None:
            raise ValueError("streaming_mouse_seed_state_mode=stats_tail requires streaming stats")
        last_by_recording: dict[str, tuple[float, float]] = {}
        last_by_game: dict[str, tuple[float, float]] = {}
        fallback = (0.0, 0.0)
        for recording_id, tokens in dict(stats.get("last_tokens_by_recording", {})).items():
            value = _mouse_delta_from_tokens([str(token) for token in tokens])
            value = (float(value[0]), float(value[1]))
            last_by_recording[str(recording_id)] = value
            fallback = value
        for game, tokens in dict(stats.get("last_tokens_by_game", {})).items():
            value = _mouse_delta_from_tokens([str(token) for token in tokens])
            last_by_game[str(game)] = (float(value[0]), float(value[1]))
        return _streaming_mouse_seed_state_payload(
            last_by_recording=last_by_recording,
            last_by_game=last_by_game,
            fallback=fallback,
            source="stats_tail",
            workers=0,
        )
    workers = int(
        config.get(
            "streaming_mouse_seed_state_workers",
            config.get("action_history_seed_state_workers", config.get("prediction_workers", 1)),
        )
        or 1
    )
    parallel = bool(config.get("streaming_mouse_seed_state_parallel_by_path", workers > 1))
    last_by_recording, last_by_game, fallback = _seed_streaming_mouse_delta_state(
        train_paths,
        num_workers=workers if parallel else 1,
    )
    return _streaming_mouse_seed_state_payload(
        last_by_recording=last_by_recording,
        last_by_game=last_by_game,
        fallback=fallback,
        source="parent_train_scan_parallel" if parallel and workers > 1 else "parent_train_scan",
        workers=workers if parallel else 1,
    )


def _features_with_optional_mouse_baseline(
    row: dict[str, Any],
    features: list[float],
    *,
    residual_mouse: bool,
    baseline: tuple[float, float],
) -> list[float]:
    if not residual_mouse:
        return features
    return list(features) + [float(baseline[0]), float(baseline[1])]


def _row_with_mouse_baseline(
    row: dict[str, Any],
    *,
    features: list[float],
    residual_mouse: bool,
    baseline: tuple[float, float],
) -> dict[str, Any]:
    out = {**row, "__streaming_idm_features": features}
    if residual_mouse:
        out["__streaming_mouse_baseline"] = [float(baseline[0]), float(baseline[1])]
    return out


def _mouse_target_for_row(row: dict[str, Any], *, mouse_target_mode: str) -> tuple[float, float]:
    base_mode = _base_mouse_target_mode(mouse_target_mode)
    target_dx, target_dy = target_mouse_delta(row, mode=base_mode)
    if not _residual_mouse_from_mode(mouse_target_mode):
        return target_dx, target_dy
    baseline = row.get("__streaming_mouse_baseline")
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
        raise ValueError("residual_last_seen mouse targets require __streaming_mouse_baseline on each row")
    return float(target_dx) - float(baseline[0]), float(target_dy) - float(baseline[1])


def _mouse_targets(torch, rows: list[dict[str, Any]], *, device: str, mouse_target_mode: str = "mean"):
    return torch.tensor(
        [_mouse_target_for_row(row, mouse_target_mode=mouse_target_mode) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _axis_class_indices_with_index(
    records: list[dict[str, Any]],
    class_index: dict[str, int],
    *,
    mouse_target_mode: str = "mean",
) -> tuple[list[int], list[int]]:
    dx_indices: list[int] = []
    dy_indices: list[int] = []
    for row in records:
        dx, dy = _mouse_target_for_row(row, mouse_target_mode=mouse_target_mode)
        dx_indices.append(class_index[_axis_suffix_from_delta(dx, "MOUSE_DX_")])
        dy_indices.append(class_index[_axis_suffix_from_delta(dy, "MOUSE_DY_")])
    return dx_indices, dy_indices


def _axis_targets(
    torch,
    rows: list[dict[str, Any]],
    axis_classes: list[str],
    *,
    device: str,
    class_index: dict[str, int] | None = None,
    mouse_target_mode: str = "mean",
):
    if class_index is None:
        class_index = {label: idx for idx, label in enumerate(axis_classes)}
        dx, dy = _axis_class_indices_with_index(rows, class_index, mouse_target_mode=mouse_target_mode)
    else:
        dx, dy = _axis_class_indices_with_index(rows, class_index, mouse_target_mode=mouse_target_mode)
    return (
        torch.tensor(dx, dtype=torch.long, device=device),
        torch.tensor(dy, dtype=torch.long, device=device),
    )


def _iter_batches(
    path: str | Path | Sequence[str | Path],
    batch_size: int,
    max_examples: int | None = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    shard_by_path: bool = False,
    skip_examples: int = 0,
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    seen = 0
    skipped = 0
    for path_idx, record_path in enumerate(_record_paths_from_value(path)):
        if shard_by_path and world_size > 1 and (path_idx % world_size) != rank:
            continue
        for row in iter_jsonl(record_path):
            if skipped < int(skip_examples):
                skipped += 1
                continue
            if max_examples is not None and seen >= max_examples:
                break
            batch.append(row)
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if max_examples is not None and seen >= max_examples:
            break
    if batch:
        yield batch


def _iter_causal_feature_batches(
    path: str | Path | Sequence[str | Path],
    batch_size: int,
    max_examples: int | None,
    *,
    feature_mode: str,
    history_vocab: list[str],
    history_len: int,
    skip_examples: int = 0,
    residual_mouse: bool = False,
) -> Iterable[list[dict[str, Any]]]:
    if history_len <= 0 and not residual_mouse:
        yield from _iter_batches(path, batch_size, max_examples, skip_examples=skip_examples)
        return
    batch: list[dict[str, Any]] = []
    seen = 0
    skipped = 0
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    for record_path in _record_paths_from_value(path):
        for row in iter_jsonl(record_path):
            if history_len > 0:
                recording_id = str(row.get("recording_id", ""))
                history, button_state = _ensure_history_state(histories, button_states, recording_id)
                features = _record_features_with_history(
                    row,
                    feature_mode=feature_mode,
                    history_vocab=history_vocab,
                    history_len=history_len,
                    history=history,
                    button_state=button_state,
                )
            else:
                history = []
                button_state = {}
                features = _base_record_features(row, feature_mode=feature_mode)
            mouse_baseline = _mouse_baseline_for_row(
                row,
                last_by_recording=last_mouse_by_recording,
                last_by_game=last_mouse_by_game,
                fallback=last_mouse_fallback,
            )
            features = _features_with_optional_mouse_baseline(
                row,
                features,
                residual_mouse=residual_mouse,
                baseline=mouse_baseline,
            )
            tokens = _tokens(row)
            if history_len > 0:
                _append_history(history, button_state, tokens, history_len=history_len)
            if residual_mouse:
                last_mouse_fallback = _update_mouse_baseline_state(
                    row,
                    target_mouse_delta(row, mode="sum"),
                    last_by_recording=last_mouse_by_recording,
                    last_by_game=last_mouse_by_game,
                )
            if skipped < int(skip_examples):
                skipped += 1
                continue
            if max_examples is not None and seen >= max_examples:
                break
            batch.append(
                _row_with_mouse_baseline(
                    row,
                    features=features,
                    residual_mouse=residual_mouse,
                    baseline=mouse_baseline,
                )
            )
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if max_examples is not None and seen >= max_examples:
            break
    if batch:
        yield batch


def _cache_source_metadata(path: str | Path) -> dict[str, Any]:
    record_path = Path(path)
    stat = record_path.stat()
    return {
        "path": str(record_path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _training_cache_identity(
    path: str | Path,
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> dict[str, Any]:
    keyboard_classes = _keyboard_classes_for_heads(stats, config)
    button_classes = _button_classes_for_heads(stats, config)
    return {
        "schema": "streaming_idm_training_cache.v1",
        "source": _cache_source_metadata(path),
        "feature_mode": str(stats["feature_mode"]),
        "input_dim": int(stats["input_dim"]),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "category_vocab": list(category_vocab),
        "keyboard_head_mode": str(config.get("keyboard_head_mode", "multilabel")),
        "keyboard_classes": [list(tokens) for tokens in keyboard_classes],
        "button_head_mode": str(config.get("button_head_mode", "multilabel")),
        "button_classes": [list(tokens) for tokens in button_classes],
        "mouse_head_mode": str(config.get("mouse_head_mode", "axis_softmax")),
        "mouse_target_mode": _mouse_target_mode(config),
        "mouse_axis_classes": list(mouse_axis_classes),
        "action_history_len": int(stats.get("action_history_len", _action_history_len_from_config(config))),
        "action_history_vocab": list(stats.get("action_history_vocab", [])),
        "action_history_dim": int(stats.get("action_history_dim", 0)),
        "action_history_parallel_by_path": _action_history_parallel_by_path(config),
        "cache_version": int(config.get("training_cache_identity_version", 5)),
    }


def _training_cache_manifest_path(
    cache_dir: str | Path,
    path: str | Path,
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> Path:
    identity = _training_cache_identity(
        path,
        stats=stats,
        config=config,
        category_vocab=category_vocab,
        mouse_axis_classes=mouse_axis_classes,
    )
    key = stable_hash_json(identity)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(path).stem)[:64] or "records"
    return Path(cache_dir) / f"{safe_stem}-{key[:20]}.manifest.json"


def _cache_axis_indices(row: dict[str, Any], class_index: dict[str, int], *, mouse_target_mode: str = "mean") -> tuple[int, int]:
    dx, dy = _mouse_target_for_row(row, mouse_target_mode=mouse_target_mode)
    return (
        class_index[_axis_suffix_from_delta(dx, "MOUSE_DX_")],
        class_index[_axis_suffix_from_delta(dy, "MOUSE_DY_")],
    )


def _flush_training_cache_chunk(
    torch,
    *,
    chunk_path: Path,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    keyboard_classes: list[tuple[str, ...]],
    button_classes: list[tuple[str, ...]],
    vocab_index: dict[str, int],
    axis_class_index: dict[str, int],
    mouse_head_mode: str,
    mouse_target_mode: str,
) -> dict[str, Any]:
    feature_mode = str(stats["feature_mode"])
    x = torch.tensor(
        [_base_record_features(row, feature_mode=feature_mode) for row in rows],
        dtype=torch.float32,
    )
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device="cpu")
    x = (x - mean_t) / std_t
    mouse_y = torch.tensor([_mouse_target_for_row(row, mouse_target_mode=mouse_target_mode) for row in rows], dtype=torch.float32)
    cat_y = torch.zeros((len(rows), len(category_vocab)), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for token in set(row.get("ground_truth_tokens", [])):
            idx = vocab_index.get(str(token))
            if idx is not None:
                cat_y[row_idx, idx] = 1.0
    payload: dict[str, Any] = {
        "schema": "streaming_idm_training_cache_chunk.v1",
        "rows": len(rows),
        "x": x,
        "mouse_y": mouse_y,
        "cat_y": cat_y,
    }
    if str(config.get("keyboard_head_mode", "multilabel")) == "softmax":
        payload["keyboard_y"] = _exact_set_targets(torch, rows, keyboard_classes, _keyboard_label)
    if str(config.get("button_head_mode", "multilabel")) == "softmax":
        payload["button_y"] = _exact_set_targets(torch, rows, button_classes, _button_label)
    if mouse_head_mode == "axis_softmax":
        axis = [_cache_axis_indices(row, axis_class_index, mouse_target_mode=mouse_target_mode) for row in rows]
        payload["dx_y"] = torch.tensor([item[0] for item in axis], dtype=torch.long)
        payload["dy_y"] = torch.tensor([item[1] for item in axis], dtype=torch.long)
    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(chunk_path)
    return {"path": str(chunk_path), "rows": len(rows)}


def _build_training_cache_for_path(
    path: str | Path,
    *,
    manifest_path: str | Path,
    identity: dict[str, Any],
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
    chunk_size: int,
    force_rebuild: bool = False,
    histories: dict[str, list[list[str]]] | None = None,
    button_states: dict[str, dict[str, float]] | None = None,
    mouse_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    mouse_target_mode = _mouse_target_mode(config)
    residual_mouse = _residual_mouse_from_mode(mouse_target_mode)
    replay_existing_manifest_state = histories is not None or button_states is not None or mouse_state is not None
    if action_history_len > 0:
        histories = histories if histories is not None else {}
        button_states = button_states if button_states is not None else {}
    if residual_mouse:
        mouse_state = mouse_state if mouse_state is not None else {
            "last_by_recording": {},
            "last_by_game": {},
            "fallback": (0.0, 0.0),
        }
    if manifest_path.exists() and not force_rebuild:
        manifest = read_json(manifest_path)
        chunk_rows = manifest.get("chunks", [])
        if manifest.get("identity") == identity and chunk_rows and all(Path(row["path"]).exists() for row in chunk_rows):
            if replay_existing_manifest_state and (action_history_len > 0 or residual_mouse):
                for row in iter_jsonl(path):
                    if action_history_len > 0:
                        history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
                        _append_history(history, button_state, _tokens(row), history_len=action_history_len)
                    if residual_mouse:
                        mouse_state["fallback"] = _update_mouse_baseline_state(
                            row,
                            target_mouse_delta(row, mode="sum"),
                            last_by_recording=mouse_state["last_by_recording"],
                            last_by_game=mouse_state["last_by_game"],
                        )
            return manifest
    torch = require_torch()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = ensure_dir(manifest_path.with_suffix(""))
    for old_chunk in chunk_dir.glob("chunk_*.pt"):
        old_chunk.unlink()
    vocab_index = {token: idx for idx, token in enumerate(category_vocab)}
    axis_class_index = {label: idx for idx, label in enumerate(mouse_axis_classes)}
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    keyboard_classes = _keyboard_classes_for_heads(stats, config)
    button_classes = _button_classes_for_heads(stats, config)
    chunks: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    count = 0
    for row in iter_jsonl(path):
        if action_history_len > 0:
            recording_id = str(row.get("recording_id", ""))
            history, button_state = _ensure_history_state(histories, button_states, recording_id)
            features = _record_features_with_history(
                row,
                feature_mode=str(stats["feature_mode"]),
                history_vocab=history_vocab,
                history_len=action_history_len,
                history=history,
                button_state=button_state,
            )
            _append_history(history, button_state, _tokens(row), history_len=action_history_len)
        else:
            features = _base_record_features(row, feature_mode=str(stats["feature_mode"]))
        mouse_baseline = (
            _mouse_baseline_for_row(
                row,
                last_by_recording=mouse_state["last_by_recording"],
                last_by_game=mouse_state["last_by_game"],
                fallback=mouse_state["fallback"],
            )
            if residual_mouse
            else (0.0, 0.0)
        )
        features = _features_with_optional_mouse_baseline(
            row,
            features,
            residual_mouse=residual_mouse,
            baseline=mouse_baseline,
        )
        row = _row_with_mouse_baseline(
            row,
            features=features,
            residual_mouse=residual_mouse,
            baseline=mouse_baseline,
        )
        if residual_mouse:
            mouse_state["fallback"] = _update_mouse_baseline_state(
                row,
                target_mouse_delta(row, mode="sum"),
                last_by_recording=mouse_state["last_by_recording"],
                last_by_game=mouse_state["last_by_game"],
            )
        batch.append(row)
        count += 1
        if len(batch) >= chunk_size:
            chunks.append(
                _flush_training_cache_chunk(
                    torch,
                    chunk_path=chunk_dir / f"chunk_{len(chunks):06d}.pt",
                    rows=batch,
                    stats=stats,
                    config=config,
                    category_vocab=category_vocab,
                    keyboard_classes=keyboard_classes,
                    button_classes=button_classes,
                    vocab_index=vocab_index,
                    axis_class_index=axis_class_index,
                    mouse_head_mode=mouse_head_mode,
                    mouse_target_mode=mouse_target_mode,
                )
            )
            batch = []
    if batch:
        chunks.append(
            _flush_training_cache_chunk(
                torch,
                chunk_path=chunk_dir / f"chunk_{len(chunks):06d}.pt",
                rows=batch,
                stats=stats,
                config=config,
                category_vocab=category_vocab,
                keyboard_classes=keyboard_classes,
                button_classes=button_classes,
                vocab_index=vocab_index,
                axis_class_index=axis_class_index,
                mouse_head_mode=mouse_head_mode,
                mouse_target_mode=mouse_target_mode,
            )
        )
    manifest = {
        "schema": "streaming_idm_training_cache_manifest.v1",
        "identity": identity,
        "source_path": str(path),
        "manifest_path": str(manifest_path),
        "chunk_size": int(chunk_size),
        "rows": int(count),
        "chunks": chunks,
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    write_json(tmp_manifest, manifest)
    tmp_manifest.replace(manifest_path)
    return manifest


def _build_training_cache_manifests(
    record_paths: Sequence[str | Path],
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> list[dict[str, Any]]:
    cache_dir = config.get("training_cache_dir")
    if not cache_dir:
        return []
    chunk_size = int(config.get("training_cache_chunk_size", config.get("batch_size", 4096) * 2))
    if chunk_size <= 0:
        raise ValueError("training_cache_chunk_size must be positive")
    cache_workers = max(1, int(config.get("training_cache_num_workers", 1)))
    force_rebuild = bool(config.get("force_rebuild_training_cache", False))
    tasks = []
    for path in record_paths:
        identity = _training_cache_identity(
            path,
            stats=stats,
            config=config,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
        manifest_path = _training_cache_manifest_path(
            cache_dir,
            path,
            stats=stats,
            config=config,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
        tasks.append((path, manifest_path, identity))
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    residual_mouse = _residual_mouse_from_mode(_mouse_target_mode(config))
    if (action_history_len > 0 and not _action_history_parallel_by_path(config)) or (
        residual_mouse and len(tasks) > 1 and cache_workers <= 1
    ):
        histories: dict[str, list[list[str]]] = {}
        button_states: dict[str, dict[str, float]] = {}
        mouse_state: dict[str, Any] = {
            "last_by_recording": {},
            "last_by_game": {},
            "fallback": (0.0, 0.0),
        }
        return [
            _build_training_cache_for_path(
                path,
                manifest_path=manifest_path,
                identity=identity,
                stats=stats,
                config=config,
                category_vocab=category_vocab,
                mouse_axis_classes=mouse_axis_classes,
                chunk_size=chunk_size,
                force_rebuild=force_rebuild,
                histories=histories,
                button_states=button_states,
                mouse_state=mouse_state,
            )
            for path, manifest_path, identity in tasks
        ]
    if len(tasks) > 1 and cache_workers > 1:
        manifests: list[dict[str, Any]] = []
        workers = min(cache_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _build_training_cache_for_path,
                    path,
                    manifest_path=manifest_path,
                    identity=identity,
                    stats=stats,
                    config=config,
                    category_vocab=category_vocab,
                    mouse_axis_classes=mouse_axis_classes,
                    chunk_size=chunk_size,
                    force_rebuild=force_rebuild,
                ): manifest_path
                for path, manifest_path, identity in tasks
            }
            for future in as_completed(futures):
                manifests.append(future.result())
        return sorted(manifests, key=lambda row: str(row["source_path"]))
    return [
        _build_training_cache_for_path(
            path,
            manifest_path=manifest_path,
            identity=identity,
            stats=stats,
            config=config,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
            chunk_size=chunk_size,
            force_rebuild=force_rebuild,
        )
        for path, manifest_path, identity in tasks
    ]


def _load_training_cache_manifests(
    record_paths: Sequence[str | Path],
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> list[dict[str, Any]]:
    cache_dir = config.get("training_cache_dir")
    if not cache_dir:
        return []
    manifests: list[dict[str, Any]] = []
    for path in record_paths:
        identity = _training_cache_identity(
            path,
            stats=stats,
            config=config,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
        manifest_path = _training_cache_manifest_path(
            cache_dir,
            path,
            stats=stats,
            config=config,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing streaming IDM training cache manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale streaming IDM training cache manifest: {manifest_path}")
        manifests.append(manifest)
    return manifests


def _training_cache_manifest_row_count(manifest: dict[str, Any]) -> int:
    if manifest.get("rows") is not None:
        return int(manifest.get("rows") or 0)
    return sum(int(chunk.get("rows") or 0) for chunk in manifest.get("chunks", []))


def _training_cache_manifest_byte_count(manifest: dict[str, Any]) -> int:
    if manifest.get("bytes") is not None:
        return int(manifest.get("bytes") or 0)
    total = 0
    for chunk in manifest.get("chunks", []):
        if chunk.get("bytes") is not None:
            total += int(chunk.get("bytes") or 0)
            continue
        path = chunk.get("path")
        if path and Path(path).exists():
            total += Path(path).stat().st_size
    return total


def _training_cache_rank_assignment(
    cache_manifests: Sequence[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    mode: str = "greedy_rows",
) -> set[int]:
    """Return manifest indices assigned to a DDP rank.

    The legacy modulo-by-path scheme can strand one GPU when shard row counts
    differ.  Greedy assignment is deterministic and uses cache-manifest row
    counts, keeping DDP join time lower without changing record contents.
    """

    if world_size <= 1:
        return set(range(len(cache_manifests)))
    normalized = mode.replace("-", "_").lower()
    if normalized in {"round_robin", "path_modulo", "modulo"}:
        return {idx for idx in range(len(cache_manifests)) if (idx % world_size) == rank}
    if normalized not in {"greedy_rows", "greedy_bytes"}:
        raise ValueError(
            "training_cache_shard_assignment must be one of "
            "greedy_rows, greedy_bytes, round_robin/path_modulo"
        )
    load = [0 for _ in range(world_size)]
    assigned: list[list[int]] = [[] for _ in range(world_size)]
    weighted: list[tuple[int, int]] = []
    for idx, manifest in enumerate(cache_manifests):
        weight = (
            _training_cache_manifest_byte_count(manifest)
            if normalized == "greedy_bytes"
            else _training_cache_manifest_row_count(manifest)
        )
        weighted.append((idx, int(weight)))
    for idx, weight in sorted(weighted, key=lambda item: (-item[1], item[0])):
        target_rank = min(range(world_size), key=lambda candidate: (load[candidate], candidate))
        assigned[target_rank].append(idx)
        load[target_rank] += int(weight)
    return set(assigned[rank])


def _training_cache_assignment_plan(
    cache_manifests: Sequence[dict[str, Any]],
    *,
    world_size: int,
    mode: str = "greedy_rows",
) -> dict[str, Any]:
    normalized = mode.replace("-", "_").lower()
    ranks = []
    for rank in range(max(1, world_size)):
        assigned = sorted(
            _training_cache_rank_assignment(
                cache_manifests,
                rank=rank,
                world_size=world_size,
                mode=normalized,
            )
        )
        rows = sum(_training_cache_manifest_row_count(cache_manifests[idx]) for idx in assigned)
        bytes_ = sum(_training_cache_manifest_byte_count(cache_manifests[idx]) for idx in assigned)
        ranks.append({"rank": rank, "manifest_indices": assigned, "rows": rows, "bytes": bytes_})
    row_loads = [int(row["rows"]) for row in ranks]
    byte_loads = [int(row["bytes"]) for row in ranks]
    return {
        "mode": normalized,
        "world_size": int(world_size),
        "ranks": ranks,
        "row_load_min": min(row_loads) if row_loads else 0,
        "row_load_max": max(row_loads) if row_loads else 0,
        "byte_load_min": min(byte_loads) if byte_loads else 0,
        "byte_load_max": max(byte_loads) if byte_loads else 0,
    }


def _iter_training_cache_batches(
    torch,
    cache_manifests: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    max_examples: int | None,
    rank: int,
    world_size: int,
    shard_by_path: bool,
    shard_assignment: str = "greedy_rows",
) -> Iterable[tuple[Any, Any, Any, Any | None, Any | None, Any | None, Any | None, int]]:
    seen = 0
    source_batch_idx = 0
    assigned_indices = (
        _training_cache_rank_assignment(
            cache_manifests,
            rank=rank,
            world_size=world_size,
            mode=shard_assignment,
        )
        if shard_by_path and world_size > 1
        else None
    )
    for path_idx, manifest in enumerate(cache_manifests):
        if assigned_indices is not None and path_idx not in assigned_indices:
            continue
        for chunk in manifest.get("chunks", []):
            try:
                payload = torch.load(chunk["path"], map_location="cpu", weights_only=False)
            except TypeError:  # pragma: no cover - older torch releases.
                payload = torch.load(chunk["path"], map_location="cpu")
            rows = int(payload["rows"])
            for start in range(0, rows, batch_size):
                if max_examples is not None and seen >= max_examples:
                    break
                end = min(rows, start + batch_size)
                if max_examples is not None:
                    end = min(end, start + (max_examples - seen))
                batch_rows = int(end - start)
                current_source_batch_idx = source_batch_idx
                source_batch_idx += 1
                if not shard_by_path and world_size > 1 and (current_source_batch_idx % world_size) != rank:
                    continue
                x = payload["x"][start:end].to(device)
                mouse_y = payload["mouse_y"][start:end].to(device)
                cat_y = payload["cat_y"][start:end].to(device)
                keyboard_y = payload.get("keyboard_y")
                button_y = payload.get("button_y")
                if keyboard_y is not None:
                    keyboard_y = keyboard_y[start:end].to(device)
                if button_y is not None:
                    button_y = button_y[start:end].to(device)
                dx_y = payload.get("dx_y")
                dy_y = payload.get("dy_y")
                if dx_y is not None and dy_y is not None:
                    dx_y = dx_y[start:end].to(device)
                    dy_y = dy_y[start:end].to(device)
                seen += batch_rows
                yield x, mouse_y, cat_y, keyboard_y, button_y, dx_y, dy_y, batch_rows
            if max_examples is not None and seen >= max_examples:
                break
        if max_examples is not None and seen >= max_examples:
            break


def _soft_pos_weight(torch, category_counts: dict[str, int], vocab: list[str], total: int, *, cap: float, device: str):
    if not vocab:
        return None
    values = []
    for token in vocab:
        pos = max(1, int(category_counts.get(token, 0)))
        neg = max(1, total - pos)
        values.append(min(float(cap), neg / pos))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _train_one_epoch(
    torch,
    model,
    opt,
    *,
    train_records: str | Path,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    cat_pos_weight,
    keyboard_classes: list[tuple[str, ...]],
    keyboard_class_weight,
    button_classes: list[tuple[str, ...]],
    button_class_weight,
    mouse_axis_classes: list[str],
    rank: int = 0,
    world_size: int = 1,
    train_record_paths: Sequence[str | Path] | None = None,
    training_cache_manifests: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    batch_size = int(config.get("batch_size", 2048))
    feature_mode = str(stats["feature_mode"])
    keyboard_head_mode = str(config.get("keyboard_head_mode", "multilabel"))
    button_head_mode = str(config.get("button_head_mode", "multilabel"))
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    mouse_target_mode = _mouse_target_mode(config)
    losses: list[float] = []
    batches = 0
    examples = 0
    loss_sum = 0.0
    record_paths = list(train_record_paths or _record_paths_from_value(train_records))
    shard_by_path = len(record_paths) > 1
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    residual_mouse = _residual_mouse_from_mode(mouse_target_mode)
    if training_cache_manifests:
        cache_shard_by_path = bool(config.get("training_cache_shard_by_path", shard_by_path))
        progress_interval = int(config.get("training_progress_interval_batches", 0) or 0)
        progress_dir = ensure_dir(Path(config.get("output_dir", "outputs/idm_streaming_full")) / "rank_progress")
        heartbeat_path = progress_dir / f"train_rank{rank}.json"
        for batch_idx, (x, mouse_y, cat_y, keyboard_y, button_y, dx_y, dy_y, batch_rows) in enumerate(
            _iter_training_cache_batches(
                torch,
                training_cache_manifests,
                batch_size=batch_size,
                device=device,
                max_examples=config.get("max_train_examples"),
                rank=rank,
                world_size=world_size,
                shard_by_path=cache_shard_by_path,
                shard_assignment=str(config.get("training_cache_shard_assignment", "greedy_rows")),
            )
        ):
            pred = model(x)
            mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y)
            category_end = 2 + len(category_vocab)
            if category_vocab:
                cat_loss = _categorical_loss(torch, pred[:, 2:category_end], cat_y, cat_pos_weight, config)
            else:
                cat_loss = torch.tensor(0.0, device=device)
            keyboard_end = category_end
            if keyboard_head_mode == "softmax":
                keyboard_end = category_end + len(keyboard_classes)
                if keyboard_y is None:
                    raise ValueError("training cache is missing keyboard targets for keyboard_head_mode=softmax")
                keyboard_loss = torch.nn.functional.cross_entropy(
                    pred[:, category_end:keyboard_end],
                    keyboard_y,
                    weight=keyboard_class_weight,
                )
            else:
                keyboard_loss = torch.tensor(0.0, device=device)
            button_end = keyboard_end
            if button_head_mode == "softmax":
                button_end = keyboard_end + len(button_classes)
                if button_y is None:
                    raise ValueError("training cache is missing button targets for button_head_mode=softmax")
                button_loss = torch.nn.functional.cross_entropy(
                    pred[:, keyboard_end:button_end],
                    button_y,
                    weight=button_class_weight,
                )
            else:
                button_loss = torch.tensor(0.0, device=device)
            if mouse_head_mode == "axis_softmax":
                if dx_y is None or dy_y is None:
                    raise ValueError("training cache is missing axis targets for mouse_head_mode=axis_softmax")
                axis_count = len(mouse_axis_classes)
                dx_logits = pred[:, button_end : button_end + axis_count]
                dy_logits = pred[:, button_end + axis_count : button_end + (2 * axis_count)]
                axis_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(dx_logits, dx_y)
                    + torch.nn.functional.cross_entropy(dy_logits, dy_y)
                )
            else:
                axis_loss = torch.tensor(0.0, device=device)
            loss = (
                float(config.get("mouse_regression_loss_weight", 1.0)) * mouse_loss
                + float(config.get("categorical_loss_weight", 0.5)) * cat_loss
                + float(config.get("keyboard_softmax_loss_weight", 1.0)) * keyboard_loss
                + float(config.get("button_softmax_loss_weight", 1.0)) * button_loss
                + float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)) * axis_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            opt.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            loss_sum += loss_value * batch_rows
            batches += 1
            examples += batch_rows
            if progress_interval > 0 and (batches == 1 or batches % progress_interval == 0):
                write_json(
                    heartbeat_path,
                    {
                        "schema": "streaming_idm_train_rank_progress.v1",
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "batches": int(batches),
                        "examples": int(examples),
                        "last_local_batch_index": int(batch_idx),
                        "loss": loss_value,
                        "updated_at_epoch": time.time(),
                        "training_cache": True,
                        "training_cache_shard_by_path": cache_shard_by_path,
                    },
                )
        return {
            "loss": sum(losses) / len(losses) if losses else None,
            "loss_sum": loss_sum,
            "batches": batches,
            "examples": examples,
            "training_cache": True,
            "training_cache_shard_by_path": cache_shard_by_path,
        }
    if (action_history_len > 0 or residual_mouse) and world_size > 1:
        raise ValueError("distributed streaming IDM training with causal action/mouse state requires training_cache_dir")
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device=device)
    vocab_index = {token: idx for idx, token in enumerate(category_vocab)}
    axis_class_index = {label: idx for idx, label in enumerate(mouse_axis_classes)}
    for batch_idx, rows in enumerate(
        _iter_causal_feature_batches(
            record_paths,
            batch_size,
            config.get("max_train_examples"),
            feature_mode=feature_mode,
            history_vocab=history_vocab,
            history_len=action_history_len,
            residual_mouse=residual_mouse,
        )
    ):
        if world_size > 1 and not shard_by_path and (batch_idx % world_size) != rank:
            continue
        x = _batch_features(
            torch,
            rows,
            feature_mode=feature_mode,
            mean=stats["mean"],
            std=stats["std"],
            device=device,
            mean_t=mean_t,
            std_t=std_t,
        )
        mouse_y = _mouse_targets(torch, rows, device=device, mouse_target_mode=mouse_target_mode)
        cat_y = _category_targets(torch, rows, category_vocab, device=device, vocab_index=vocab_index)
        pred = model(x)
        mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y)
        category_end = 2 + len(category_vocab)
        if category_vocab:
            cat_loss = _categorical_loss(torch, pred[:, 2:category_end], cat_y, cat_pos_weight, config)
        else:
            cat_loss = torch.tensor(0.0, device=device)
        keyboard_end = category_end
        if keyboard_head_mode == "softmax":
            keyboard_end = category_end + len(keyboard_classes)
            keyboard_y = _exact_set_targets(torch, rows, keyboard_classes, _keyboard_label, device=device)
            keyboard_loss = torch.nn.functional.cross_entropy(
                pred[:, category_end:keyboard_end],
                keyboard_y,
                weight=keyboard_class_weight,
            )
        else:
            keyboard_loss = torch.tensor(0.0, device=device)
        button_end = keyboard_end
        if button_head_mode == "softmax":
            button_end = keyboard_end + len(button_classes)
            button_y = _exact_set_targets(torch, rows, button_classes, _button_label, device=device)
            button_loss = torch.nn.functional.cross_entropy(
                pred[:, keyboard_end:button_end],
                button_y,
                weight=button_class_weight,
            )
        else:
            button_loss = torch.tensor(0.0, device=device)
        if mouse_head_mode == "axis_softmax":
            dx_y, dy_y = _axis_targets(
                torch,
                rows,
                mouse_axis_classes,
                device=device,
                class_index=axis_class_index,
                mouse_target_mode=mouse_target_mode,
            )
            axis_count = len(mouse_axis_classes)
            dx_logits = pred[:, button_end : button_end + axis_count]
            dy_logits = pred[:, button_end + axis_count : button_end + (2 * axis_count)]
            axis_loss = 0.5 * (
                torch.nn.functional.cross_entropy(dx_logits, dx_y)
                + torch.nn.functional.cross_entropy(dy_logits, dy_y)
            )
        else:
            axis_loss = torch.tensor(0.0, device=device)
        loss = (
            float(config.get("mouse_regression_loss_weight", 1.0)) * mouse_loss
            + float(config.get("categorical_loss_weight", 0.5)) * cat_loss
            + float(config.get("keyboard_softmax_loss_weight", 1.0)) * keyboard_loss
            + float(config.get("button_softmax_loss_weight", 1.0)) * button_loss
            + float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)) * axis_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
        opt.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        loss_sum += loss_value * len(rows)
        batches += 1
        examples += len(rows)
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "loss_sum": loss_sum,
        "batches": batches,
        "examples": examples,
        "training_cache": False,
    }


class StreamingActionMetrics:
    def __init__(self) -> None:
        self.matched = 0
        self.keyboard_total = 0
        self.keyboard_correct = 0
        self.button_total = 0
        self.button_correct = 0
        self.button_predicted_total = 0
        self.button_exact_tp = 0
        self.button_fp = 0
        self.button_fn = 0
        self.button_no_gt = 0
        self.button_no_gt_fp = 0
        self.mouse_n = 0
        self.sum_pred = 0.0
        self.sum_gt = 0.0
        self.sum_pred_sq = 0.0
        self.sum_gt_sq = 0.0
        self.sum_cross = 0.0
        self.sum_abs_pred = 0.0
        self.sum_abs_gt = 0.0
        self.failure_count = 0

    @staticmethod
    def _category(tokens: list[str], prefixes: tuple[str, ...]) -> list[str]:
        return sorted(token for token in tokens if token.startswith(prefixes))

    @staticmethod
    def _axis_mean(tokens: list[str], prefix: str) -> float | None:
        from fdm_d2e.tokenization.actions import token_to_delta_class

        values = [token_to_delta_class(token) for token in tokens if token.startswith(prefix)]
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    def update(self, predicted_tokens: list[str], row: dict[str, Any]) -> None:
        self.matched += 1
        gtokens = list(row.get("ground_truth_tokens", []))
        pk = self._category(predicted_tokens, ("KEY_",))
        gk = self._category(gtokens, ("KEY_",))
        if gk:
            self.keyboard_total += 1
            self.keyboard_correct += int(pk == gk)
        pb = self._category(predicted_tokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        gb = self._category(gtokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        if pb:
            self.button_predicted_total += 1
        if gb:
            self.button_total += 1
            self.button_correct += int(pb == gb)
            if pb == gb:
                self.button_exact_tp += 1
            else:
                self.button_fn += 1
                if pb:
                    self.button_fp += 1
        else:
            self.button_no_gt += 1
            if pb:
                self.button_fp += 1
                self.button_no_gt_fp += 1
        for axis_prefix in ("MOUSE_DX_", "MOUSE_DY_"):
            pred_axis = self._axis_mean(predicted_tokens, axis_prefix)
            gt_axis = self._axis_mean(gtokens, axis_prefix)
            if pred_axis is not None and gt_axis is not None:
                self.mouse_n += 1
                self.sum_pred += pred_axis
                self.sum_gt += gt_axis
                self.sum_pred_sq += pred_axis * pred_axis
                self.sum_gt_sq += gt_axis * gt_axis
                self.sum_cross += pred_axis * gt_axis
                self.sum_abs_pred += abs(pred_axis)
                self.sum_abs_gt += abs(gt_axis)
        if predicted_tokens != gtokens:
            self.failure_count += 1

    def payload(self) -> dict[str, Any]:
        if self.mouse_n >= 2:
            n = float(self.mouse_n)
            cov = self.sum_cross - (self.sum_pred * self.sum_gt / n)
            var_pred = self.sum_pred_sq - (self.sum_pred * self.sum_pred / n)
            var_gt = self.sum_gt_sq - (self.sum_gt * self.sum_gt / n)
            pearson = cov / ((var_pred * var_gt) ** 0.5) if var_pred > 0 and var_gt > 0 else None
            mean_abs_pred = self.sum_abs_pred / n
            mean_abs_gt = self.sum_abs_gt / n
            scale_ratio = max(mean_abs_pred, mean_abs_gt) / min(mean_abs_pred, mean_abs_gt) if mean_abs_pred > 0 and mean_abs_gt > 0 else None
        else:
            pearson = None
            scale_ratio = None
        metrics = {
            "schema": "metrics.v1",
            "stage": "fdm_eval",
            "num_examples": self.matched,
            "keyboard": {
                "status": "computed" if self.keyboard_total else "absent",
                "accuracy": self.keyboard_correct / self.keyboard_total if self.keyboard_total else None,
                "num_examples": self.keyboard_total,
            },
            "mouse_button": {
                "status": "computed" if self.button_total else "absent",
                "accuracy": self.button_correct / self.button_total if self.button_total else None,
                "num_examples": self.button_total,
                "predicted_examples": self.button_predicted_total,
                "exact_true_positive_examples": self.button_exact_tp,
                "false_positive_examples": self.button_fp,
                "false_negative_examples": self.button_fn,
                "precision": self.button_exact_tp / (self.button_exact_tp + self.button_fp) if (self.button_exact_tp + self.button_fp) else None,
                "recall": self.button_exact_tp / (self.button_exact_tp + self.button_fn) if (self.button_exact_tp + self.button_fn) else None,
                "f1": (
                    (2 * self.button_exact_tp) / ((2 * self.button_exact_tp) + self.button_fp + self.button_fn)
                    if ((2 * self.button_exact_tp) + self.button_fp + self.button_fn)
                    else None
                ),
                "no_button_examples": self.button_no_gt,
                "no_button_false_positive_examples": self.button_no_gt_fp,
                "no_button_false_positive_rate": self.button_no_gt_fp / self.button_no_gt if self.button_no_gt else None,
            },
            "mouse_move": {
                "status": "computed" if self.mouse_n else "absent",
                "pearson": pearson,
                "scale_ratio": scale_ratio,
                "num_values": self.mouse_n,
            },
            "failure_count": self.failure_count,
        }
        validate_named(metrics, "metrics.schema.json")
        return metrics


_STREAMING_ACTION_METRIC_FIELDS = (
    "matched",
    "keyboard_total",
    "keyboard_correct",
    "button_total",
    "button_correct",
    "button_predicted_total",
    "button_exact_tp",
    "button_fp",
    "button_fn",
    "button_no_gt",
    "button_no_gt_fp",
    "mouse_n",
    "sum_pred",
    "sum_gt",
    "sum_pred_sq",
    "sum_gt_sq",
    "sum_cross",
    "sum_abs_pred",
    "sum_abs_gt",
    "failure_count",
)


def _metric_state(metric: StreamingActionMetrics) -> dict[str, int | float]:
    return {field: getattr(metric, field) for field in _STREAMING_ACTION_METRIC_FIELDS}


def _metric_from_state(state: dict[str, Any]) -> StreamingActionMetrics:
    metric = StreamingActionMetrics()
    for field in _STREAMING_ACTION_METRIC_FIELDS:
        if field in state:
            setattr(metric, field, state[field])
    return metric


def _merge_metric_state(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    merged = {field: (left or {}).get(field, 0) for field in _STREAMING_ACTION_METRIC_FIELDS}
    for field in _STREAMING_ACTION_METRIC_FIELDS:
        merged[field] = merged.get(field, 0) + right.get(field, 0)
    return merged


def _metric_state_map(metrics: dict[str, StreamingActionMetrics]) -> dict[str, dict[str, Any]]:
    return {name: _metric_state(metric) for name, metric in metrics.items()}


def _nested_metric_state_map(metrics: dict[str, dict[str, StreamingActionMetrics]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        name: {key: _metric_state(metric) for key, metric in metric_map.items()}
        for name, metric_map in metrics.items()
    }


def _merge_named_metric_states(items: Iterable[dict[str, dict[str, Any]]]) -> dict[str, StreamingActionMetrics]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        for name, state in item.items():
            merged[name] = _merge_metric_state(merged.get(name), state)
    return {name: _metric_from_state(state) for name, state in merged.items()}


def _merge_nested_metric_states(items: Iterable[dict[str, dict[str, dict[str, Any]]]]) -> dict[str, dict[str, StreamingActionMetrics]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for item in items:
        for name, metric_map in item.items():
            target_map = merged.setdefault(name, {})
            for key, state in metric_map.items():
                target_map[key] = _merge_metric_state(target_map.get(key), state)
    return {
        name: {key: _metric_from_state(state) for key, state in metric_map.items()}
        for name, metric_map in merged.items()
    }


def _group_keys(row: dict[str, Any]) -> list[str]:
    keys = ["all"]
    for tag in row.get("eval_split_tags", []) or []:
        keys.append(f"eval_tag:{tag}")
    for field in ("game", "resolution_tier", "source_id"):
        value = row.get(field)
        if value is not None:
            keys.append(f"{field}:{value}")
    for field in ("split_temporal", "split_heldout_recording", "split_heldout_game"):
        value = row.get(field)
        if value is not None:
            keys.append(f"{field}:{value}")
    return keys


def _ensure_metric(metrics: dict[str, StreamingActionMetrics], key: str) -> StreamingActionMetrics:
    if key not in metrics:
        metrics[key] = StreamingActionMetrics()
    return metrics[key]


def _baseline_tokens(name: str, row: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    if name == "noop":
        return ["NOOP"]
    majority = list(stats.get("global_majority_tokens") or ["NOOP"])
    if name == "global_majority":
        return majority
    if name == "last_seen_train":
        by_recording = stats.get("last_tokens_by_recording", {})
        by_game = stats.get("last_tokens_by_game", {})
        return list(
            by_recording.get(str(row.get("recording_id", "")))
            or by_game.get(str(row.get("game", "unknown")))
            or majority
        )
    raise ValueError(f"unsupported streaming baseline: {name}")


def _metric_payloads(metrics: dict[str, StreamingActionMetrics]) -> dict[str, Any]:
    return {name: metric.payload() for name, metric in sorted(metrics.items())}


def _streaming_statistical_comparison(
    metrics_by_model_cluster: dict[str, dict[str, StreamingActionMetrics]],
    endpoints_config: dict[str, Any],
) -> dict[str, Any]:
    default_reference_name = str(endpoints_config.get("reference_baseline", "noop"))
    bootstrap_cfg = dict(endpoints_config.get("bootstrap", {}))
    comparisons: list[dict[str, Any]] = []
    payload_by_model_cluster = {
        model: {cluster: metric.payload() for cluster, metric in cluster_metrics.items()}
        for model, cluster_metrics in metrics_by_model_cluster.items()
    }
    for endpoint in endpoints_config.get("endpoints", []):
        reference_name = str(endpoint.get("reference_baseline", default_reference_name))
        if reference_name not in payload_by_model_cluster:
            continue
        reference_values = {
            cluster: value
            for cluster, metrics in payload_by_model_cluster[reference_name].items()
            if (value := endpoint_value(metrics, endpoint)) is not None
        }
        for name, cluster_payloads in payload_by_model_cluster.items():
            if name == reference_name:
                continue
            candidate_values = {
                cluster: value
                for cluster, metrics in cluster_payloads.items()
                if (value := endpoint_value(metrics, endpoint)) is not None
            }
            stats = cluster_bootstrap_delta(
                candidate_values,
                reference_values,
                direction=str(endpoint.get("direction", "higher")),
                n_resamples=int(bootstrap_cfg.get("n_resamples", 2000)),
                confidence=float(bootstrap_cfg.get("confidence", 0.95)),
                seed=int(bootstrap_cfg.get("seed", 0)) + len(comparisons),
            )
            comparisons.append(
                {
                    "model": name,
                    "reference": reference_name,
                    "endpoint": endpoint["name"],
                    "direction": endpoint.get("direction", "higher"),
                    "min_effect": endpoint.get("min_effect"),
                    **stats,
                }
            )
    payload = {
        "schema": "stat_comparison.v1",
        "reference_baseline": default_reference_name,
        "correction": str(endpoints_config.get("correction", "holm_bonferroni")),
        "cluster_key": str(endpoints_config.get("cluster_key", "recording_id")),
        "comparisons": holm_bonferroni(comparisons),
    }
    validate_named(payload, "stat_comparison.schema.json")
    return payload


def _distributed_runtime(torch, config: dict[str, Any]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    force_cpu = bool(config.get("force_cpu", False))
    enabled = world_size > 1
    backend = None
    if enabled:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is required for WORLD_SIZE>1 streaming training")
        backend = str(config.get("distributed_backend") or ("nccl" if torch.cuda.is_available() and not force_cpu else "gloo"))
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("distributed_backend=nccl requires CUDA")
        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.set_device(local_rank)
        timeout_seconds = config.get("distributed_timeout_seconds")
        if timeout_seconds is None:
            timeout_seconds = os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS") or os.environ.get("TORCH_DIST_TIMEOUT_SECONDS")
        if not torch.distributed.is_initialized():
            init_kwargs: dict[str, Any] = {"backend": backend}
            if timeout_seconds is not None:
                timeout = float(timeout_seconds)
                if timeout <= 0:
                    raise ValueError("distributed_timeout_seconds must be positive")
                init_kwargs["timeout"] = timedelta(seconds=timeout)
            torch.distributed.init_process_group(**init_kwargs)
    else:
        timeout_seconds = None
    if torch.cuda.is_available() and not force_cpu:
        device = f"cuda:{local_rank}" if enabled else "cuda"
    else:
        device = "cpu"
    return {
        "enabled": enabled,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_rank0": rank == 0,
        "backend": backend,
        "device": device,
        "timeout_seconds": float(timeout_seconds) if enabled and timeout_seconds is not None else None,
    }


def _streaming_resume_epochs_to_run(*, total_epochs: int, completed_epochs: int, resume_training: bool) -> int:
    """Return how many epochs this streaming-IDM invocation should execute.

    ``resume_training_checkpoint`` is recovery-oriented: ``epochs`` names the
    desired total epoch count for the run, not additional epochs to append.
    ``initial_checkpoint_path`` remains warm-start-oriented and keeps the
    historical behavior of executing the configured epoch count.
    """

    total = max(0, int(total_epochs))
    completed = max(0, int(completed_epochs))
    if not resume_training:
        return total
    return max(0, total - completed)


def _barrier(torch, dist: dict[str, Any]) -> None:
    if dist["enabled"] and torch.distributed.is_initialized():
        if str(dist.get("backend")) == "nccl" and str(dist.get("device", "")).startswith("cuda"):
            torch.distributed.barrier(device_ids=[int(dist["local_rank"])])
        else:
            torch.distributed.barrier()


def _aggregate_epoch_stats(torch, stats: dict[str, Any], *, device: str, dist: dict[str, Any]) -> dict[str, Any]:
    if not dist["enabled"]:
        return stats
    tensor = torch.tensor(
        [
            float(stats.get("loss_sum") or 0.0),
            float(stats.get("examples") or 0),
            float(stats.get("batches") or 0),
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    loss_sum = float(tensor[0].item())
    examples = int(tensor[1].item())
    batches = int(tensor[2].item())
    return {
        "loss": loss_sum / examples if examples else None,
        "loss_sum": loss_sum,
        "batches": batches,
        "examples": examples,
    }


def _predicted_tokens_from_output(
    output: list[float],
    *,
    config: dict[str, Any],
    category_vocab: list[str],
    mouse_axis_classes: list[str],
    base_dx: float = 0.0,
    base_dy: float = 0.0,
    residual_mouse: bool = False,
) -> list[str]:
    category_threshold = float(config.get("category_threshold", 0.35))
    configured_thresholds = config.get("category_thresholds", {})
    category_thresholds = {
        token: float(configured_thresholds.get(token, category_threshold))
        for token in category_vocab
    } if isinstance(configured_thresholds, dict) else {token: category_threshold for token in category_vocab}
    _dx, _dy, tokens = _prediction_from_output(
        output,
        base_dx=float(base_dx),
        base_dy=float(base_dy),
        residual_mouse=bool(residual_mouse),
        category_vocab=category_vocab,
        category_thresholds=category_thresholds,
        category_threshold=category_threshold,
        keyboard_head_mode=str(config.get("keyboard_head_mode", "multilabel")),
        keyboard_classes=_keyboard_classes_for_heads({"keyboard_class_counts": {}}, config),
        keyboard_softmax_threshold=float(config.get("keyboard_softmax_threshold", 0.5)),
        button_head_mode=str(config.get("button_head_mode", "multilabel")),
        button_classes=_button_classes_for_heads({"button_class_counts": {}}, config),
        button_softmax_threshold=float(config.get("button_softmax_threshold", 0.5)),
        mouse_head_mode=str(config.get("mouse_head_mode", "axis_softmax")),
        mouse_axis_classes=mouse_axis_classes,
        mouse_axis_decode_mode=str(config.get("mouse_axis_decode_mode", "expected")),
        mouse_axis_temperature=float(config.get("mouse_axis_temperature", 1.0)),
        mouse_output_gain=float(config.get("mouse_output_gain", 1.0)),
        mouse_output_gain_x=float(config["mouse_output_gain_x"]) if config.get("mouse_output_gain_x") is not None else None,
        mouse_output_gain_y=float(config["mouse_output_gain_y"]) if config.get("mouse_output_gain_y") is not None else None,
        mouse_emit_mode=str(config.get("mouse_emit_mode", "single")),
        mouse_max_tokens_per_axis=int(config.get("mouse_max_tokens_per_axis", 8)),
    )
    return tokens


def _stream_category_group(token: str) -> str:
    if token.startswith("KEY_"):
        return "keyboard"
    if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
        return "mouse_button"
    return "other"


def _default_calibration_grid(config: dict[str, Any]) -> list[float]:
    raw_grid = config.get("category_calibration_grid")
    if isinstance(raw_grid, list) and raw_grid:
        grid = [float(value) for value in raw_grid]
    else:
        grid = [round(value / 100.0, 4) for value in range(5, 96, 5)]
    return sorted({min(0.99, max(0.01, value)) for value in grid})


def _fbeta_score(tp: float, fp: float, fn: float, beta: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = beta * beta
    score = (
        (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
        if precision or recall
        else 0.0
    )
    return score, precision, recall


def _group_no_event_fpr_cap(config: dict[str, Any], group: str) -> float | None:
    keys = {
        "keyboard": ("category_calibration_max_no_key_fpr", "keyboard_category_calibration_max_no_key_fpr"),
        "mouse_button": ("category_calibration_max_no_button_fpr", "button_category_calibration_max_no_button_fpr"),
    }.get(group, ())
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        cap = float(value)
        if cap < 0.0 or cap > 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
        return cap
    return None


def _select_group_fbeta_threshold(
    counts_by_threshold: dict[float, dict[str, int]],
    *,
    default_threshold: float,
    beta: float,
    max_no_event_fpr: float | None = None,
) -> tuple[float, dict[str, Any]]:
    best_threshold = float(default_threshold)
    best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, float(default_threshold))
    best_stats: dict[str, Any] = {}
    fallback_threshold = float(default_threshold)
    fallback_key: tuple[float, float, float, float, float] = (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    fallback_stats: dict[str, Any] = {}
    eligible_count = 0
    for threshold in sorted(counts_by_threshold):
        counts = dict(counts_by_threshold[threshold])
        score, precision, recall = _fbeta_score(counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0), beta)
        no_event_examples = int(counts.get("no_event_examples", 0))
        no_event_fp = int(counts.get("no_event_false_positive_examples", 0))
        no_event_fpr = no_event_fp / no_event_examples if no_event_examples else None
        stats = {
            **counts,
            "score": score,
            "precision": precision,
            "recall": recall,
            "no_event_false_positive_rate": no_event_fpr,
            "max_no_event_false_positive_rate": max_no_event_fpr,
        }
        constraint_satisfied = max_no_event_fpr is None or no_event_fpr is None or no_event_fpr <= max_no_event_fpr
        stats["constraint_satisfied"] = bool(constraint_satisfied)
        fallback_sort_fpr = no_event_fpr if no_event_fpr is not None else 0.0
        fallback_candidate_key = (fallback_sort_fpr, -score, -precision, -recall, -float(threshold))
        if fallback_candidate_key < fallback_key:
            fallback_key = fallback_candidate_key
            fallback_threshold = float(threshold)
            fallback_stats = stats
        if not constraint_satisfied:
            continue
        eligible_count += 1
        key = (score, precision, recall, float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats
    if not best_stats and fallback_stats:
        best_threshold = fallback_threshold
        best_stats = fallback_stats
    best_stats["eligible_thresholds"] = eligible_count
    return best_threshold, best_stats


def _group_threshold_label(group: str) -> str:
    if group == "keyboard":
        return "no_key"
    if group == "mouse_button":
        return "no_button"
    return "no_event"


def _softmax_threshold_label(head: str) -> str:
    if head == "keyboard":
        return "no_key"
    if head == "button":
        return "no_button"
    return "no_event"


def _select_softmax_threshold(
    counts_by_threshold: dict[float, dict[str, int]],
    *,
    default_threshold: float,
    beta: float,
    max_no_event_fpr: float | None,
) -> tuple[float, dict[str, Any]]:
    best_threshold = float(default_threshold)
    best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, float(default_threshold))
    best_stats: dict[str, Any] = {}
    fallback_threshold = float(default_threshold)
    fallback_key: tuple[float, float, float, float, float] = (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    fallback_stats: dict[str, Any] = {}
    eligible_count = 0
    for threshold in sorted(counts_by_threshold):
        counts = dict(counts_by_threshold[threshold])
        score, precision, recall = _fbeta_score(counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0), beta)
        no_event_examples = int(counts.get("no_event_examples", 0))
        no_event_fp = int(counts.get("no_event_false_positive_examples", 0))
        no_event_fpr = no_event_fp / no_event_examples if no_event_examples else None
        stats = {
            **counts,
            "score": score,
            "precision": precision,
            "recall": recall,
            "no_event_false_positive_rate": no_event_fpr,
            "max_no_event_false_positive_rate": max_no_event_fpr,
        }
        constraint_satisfied = max_no_event_fpr is None or no_event_fpr is None or no_event_fpr <= max_no_event_fpr
        stats["constraint_satisfied"] = bool(constraint_satisfied)
        fallback_sort_fpr = no_event_fpr if no_event_fpr is not None else 0.0
        fallback_candidate_key = (fallback_sort_fpr, -score, -precision, -recall, -float(threshold))
        if fallback_candidate_key < fallback_key:
            fallback_key = fallback_candidate_key
            fallback_threshold = float(threshold)
            fallback_stats = stats
        if not constraint_satisfied:
            continue
        eligible_count += 1
        key = (score, precision, recall, float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats
    if not best_stats and fallback_stats:
        best_threshold = fallback_threshold
        best_stats = fallback_stats
    best_stats["eligible_thresholds"] = eligible_count
    return best_threshold, best_stats


def _softmax_calibration_grid(config: dict[str, Any], *, head: str) -> list[float]:
    raw_grid = config.get(f"{head}_softmax_calibration_grid", config.get("softmax_calibration_grid"))
    if isinstance(raw_grid, list) and raw_grid:
        grid = [float(value) for value in raw_grid]
    else:
        grid = [0.0] + [round(value / 100.0, 4) for value in range(5, 100, 5)]
    return sorted({min(0.999, max(0.0, value)) for value in grid})


def _softmax_calibration_fpr_cap(config: dict[str, Any], *, head: str) -> float | None:
    if head == "keyboard":
        keys = ("keyboard_softmax_calibration_max_no_key_fpr", "softmax_calibration_max_no_key_fpr")
    elif head == "button":
        keys = ("button_softmax_calibration_max_no_button_fpr", "softmax_calibration_max_no_button_fpr")
    else:
        keys = ()
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        cap = float(value)
        if cap < 0.0 or cap > 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
        return cap
    return None


def _calibrate_streaming_softmax_thresholds_from_cache(
    torch,
    model,
    *,
    training_cache_manifests: Sequence[dict[str, Any]],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    keyboard_classes: list[tuple[str, ...]],
    button_classes: list[tuple[str, ...]],
) -> dict[str, Any]:
    """Choose exact-set softmax emission thresholds from train-cache labels.

    Cross-entropy can rank the correct non-empty exact-set class while still
    assigning too many low-confidence positive classes at inference time. This
    calibrates the emit/no-emit threshold without touching heldout labels.
    """

    keyboard_enabled = str(config.get("keyboard_head_mode", "multilabel")) == "softmax" and bool(keyboard_classes)
    button_enabled = str(config.get("button_head_mode", "multilabel")) == "softmax" and bool(button_classes)
    if not keyboard_enabled and not button_enabled:
        return {"status": "skipped_no_softmax_heads"}
    batch_size = int(config.get("softmax_calibration_batch_size", config.get("category_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048)))))
    max_examples = config.get("softmax_calibration_max_examples", config.get("category_calibration_max_examples"))
    beta = float(config.get("softmax_calibration_beta", config.get("category_calibration_beta", 1.0)))
    grids = {
        "keyboard": _softmax_calibration_grid(config, head="keyboard"),
        "button": _softmax_calibration_grid(config, head="button"),
    }
    counts: dict[str, dict[float, dict[str, int]]] = {
        "keyboard": {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "predicted_positive": 0, "positive_examples": 0, "no_event_examples": 0, "no_event_false_positive_examples": 0}
            for threshold in grids["keyboard"]
        },
        "button": {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "predicted_positive": 0, "positive_examples": 0, "no_event_examples": 0, "no_event_false_positive_examples": 0}
            for threshold in grids["button"]
        },
    }
    observed_examples = 0
    model.eval()
    with torch.no_grad():
        for x, _mouse_y, _cat_y, keyboard_y, button_y, _dx_y, _dy_y, batch_rows in _iter_training_cache_batches(
            torch,
            training_cache_manifests,
            batch_size=batch_size,
            device=device,
            max_examples=max_examples,
            rank=0,
            world_size=1,
            shard_by_path=False,
            shard_assignment=str(config.get("training_cache_shard_assignment", "greedy_rows")),
        ):
            outputs = model(x)
            category_end = 2 + len(category_vocab)
            keyboard_end = category_end + (len(keyboard_classes) if keyboard_enabled else 0)
            if keyboard_enabled:
                if keyboard_y is None:
                    raise ValueError("training cache is missing keyboard targets for keyboard softmax calibration")
                logits = outputs[:, category_end:keyboard_end]
                probs = torch.softmax(logits, dim=1)
                best_prob, best_idx = probs.max(dim=1)
                labels = keyboard_y
                for threshold in grids["keyboard"]:
                    pred_idx = torch.where(
                        (best_idx != 0) & (best_prob >= float(threshold)),
                        best_idx,
                        torch.zeros_like(best_idx),
                    )
                    label_positive = labels != 0
                    pred_positive = pred_idx != 0
                    exact = pred_idx == labels
                    row = counts["keyboard"][threshold]
                    row["tp"] += int((label_positive & exact).sum().detach().cpu().item())
                    row["fn"] += int((label_positive & ~exact).sum().detach().cpu().item())
                    row["fp"] += int(((~label_positive & pred_positive) | (label_positive & pred_positive & ~exact)).sum().detach().cpu().item())
                    row["predicted_positive"] += int(pred_positive.sum().detach().cpu().item())
                    row["positive_examples"] += int(label_positive.sum().detach().cpu().item())
                    row["no_event_examples"] += int((~label_positive).sum().detach().cpu().item())
                    row["no_event_false_positive_examples"] += int((~label_positive & pred_positive).sum().detach().cpu().item())
            button_start = keyboard_end
            button_end = button_start + (len(button_classes) if button_enabled else 0)
            if button_enabled:
                if button_y is None:
                    raise ValueError("training cache is missing button targets for button softmax calibration")
                logits = outputs[:, button_start:button_end]
                probs = torch.softmax(logits, dim=1)
                best_prob, best_idx = probs.max(dim=1)
                labels = button_y
                for threshold in grids["button"]:
                    pred_idx = torch.where(
                        (best_idx != 0) & (best_prob >= float(threshold)),
                        best_idx,
                        torch.zeros_like(best_idx),
                    )
                    label_positive = labels != 0
                    pred_positive = pred_idx != 0
                    exact = pred_idx == labels
                    row = counts["button"][threshold]
                    row["tp"] += int((label_positive & exact).sum().detach().cpu().item())
                    row["fn"] += int((label_positive & ~exact).sum().detach().cpu().item())
                    row["fp"] += int(((~label_positive & pred_positive) | (label_positive & pred_positive & ~exact)).sum().detach().cpu().item())
                    row["predicted_positive"] += int(pred_positive.sum().detach().cpu().item())
                    row["positive_examples"] += int(label_positive.sum().detach().cpu().item())
                    row["no_event_examples"] += int((~label_positive).sum().detach().cpu().item())
                    row["no_event_false_positive_examples"] += int((~label_positive & pred_positive).sum().detach().cpu().item())
            observed_examples += int(batch_rows)
    result: dict[str, Any] = {
        "status": "computed" if observed_examples else "no_calibration_examples",
        "source": "training_cache",
        "observed_examples": int(observed_examples),
        "max_examples": max_examples,
        "beta": beta,
        "heads": {},
    }
    for head, enabled, default_key in [
        ("keyboard", keyboard_enabled, "keyboard_softmax_threshold"),
        ("button", button_enabled, "button_softmax_threshold"),
    ]:
        if not enabled:
            continue
        threshold, stats = _select_softmax_threshold(
            counts[head],
            default_threshold=float(config.get(default_key, 0.5)),
            beta=beta,
            max_no_event_fpr=_softmax_calibration_fpr_cap(config, head=head),
        )
        config[default_key] = float(threshold)
        label = _softmax_threshold_label(head)
        result["heads"][head] = {
            "threshold": float(threshold),
            "grid": grids[head],
            **stats,
            f"{label}_examples": stats.get("no_event_examples"),
            f"{label}_false_positive_examples": stats.get("no_event_false_positive_examples"),
            f"{label}_false_positive_rate": stats.get("no_event_false_positive_rate"),
        }
    return result


def _calibrate_streaming_category_thresholds(
    torch,
    model,
    *,
    train_records: str | Path | Sequence[str | Path],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    default_threshold = float(config.get("category_threshold", 0.35))
    thresholds = {token: default_threshold for token in category_vocab}
    mode = str(config.get("category_threshold_mode", "global"))
    base_info: dict[str, Any] = {
        "mode": "global_threshold_streaming" if mode == "global" else mode,
        "category_threshold": default_threshold,
        "category_thresholds": thresholds,
    }
    if mode == "global" or not category_vocab:
        return thresholds, base_info
    if mode not in {"per_token_fbeta_calibrated", "group_fbeta_calibrated"}:
        raise ValueError(f"unsupported streaming category_threshold_mode: {mode}")

    grid = _default_calibration_grid(config)
    beta = float(config.get("category_calibration_beta", 1.0))
    batch_size = int(config.get("category_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048))))
    max_examples = config.get("category_calibration_max_examples")
    vocab_index = {token: idx for idx, token in enumerate(category_vocab)}
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device=device)
    groups = {
        group: [idx for idx, token in enumerate(category_vocab) if _stream_category_group(token) == group]
        for group in sorted({_stream_category_group(token) for token in category_vocab})
    }
    per_token_counts = {
        token: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in grid}
        for token in category_vocab
    }
    per_group_counts = {
        group: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in grid}
        for group in groups
    }
    per_group_exact_counts = {
        group: {
            threshold: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "predicted_positive": 0,
                "positive_examples": 0,
                "no_event_examples": 0,
                "no_event_false_positive_examples": 0,
            }
            for threshold in grid
        }
        for group in groups
    }
    observed_examples = 0
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    residual_mouse = _residual_mouse_from_mode(_mouse_target_mode(config))
    model.eval()
    with torch.no_grad():
        for rows in _iter_causal_feature_batches(
            train_records,
            batch_size,
            max_examples,
            feature_mode=str(stats["feature_mode"]),
            history_vocab=history_vocab,
            history_len=action_history_len,
            residual_mouse=residual_mouse,
        ):
            if not rows:
                continue
            x = _batch_features(
                torch,
                rows,
                feature_mode=str(stats["feature_mode"]),
                mean=stats["mean"],
                std=stats["std"],
                device=device,
                mean_t=mean_t,
                std_t=std_t,
            )
            outputs = model(x)
            category_end = 2 + len(category_vocab)
            probs = torch.sigmoid(outputs[:, 2:category_end]).detach()
            labels = _category_targets(torch, rows, category_vocab, device=device, vocab_index=vocab_index).bool()
            observed_examples += len(rows)
            for threshold in grid:
                pred = probs >= float(threshold)
                tp_vec = (pred & labels).sum(dim=0).detach().cpu().tolist()
                fp_vec = (pred & ~labels).sum(dim=0).detach().cpu().tolist()
                fn_vec = (~pred & labels).sum(dim=0).detach().cpu().tolist()
                for idx, token in enumerate(category_vocab):
                    counts = per_token_counts[token][threshold]
                    counts["tp"] += int(tp_vec[idx])
                    counts["fp"] += int(fp_vec[idx])
                    counts["fn"] += int(fn_vec[idx])
                for group, indices in groups.items():
                    counts = per_group_counts[group][threshold]
                    counts["tp"] += sum(int(tp_vec[idx]) for idx in indices)
                    counts["fp"] += sum(int(fp_vec[idx]) for idx in indices)
                    counts["fn"] += sum(int(fn_vec[idx]) for idx in indices)
                    pred_group = pred[:, indices]
                    label_group = labels[:, indices]
                    pred_any = pred_group.any(dim=1)
                    label_any = label_group.any(dim=1)
                    exact_match = (pred_group == label_group).all(dim=1)
                    exact_counts = per_group_exact_counts[group][threshold]
                    exact_counts["tp"] += int((label_any & exact_match).sum().detach().cpu().item())
                    exact_counts["fn"] += int((label_any & ~exact_match).sum().detach().cpu().item())
                    exact_counts["fp"] += int(((~label_any & pred_any) | (label_any & pred_any & ~exact_match)).sum().detach().cpu().item())
                    exact_counts["predicted_positive"] += int(pred_any.sum().detach().cpu().item())
                    exact_counts["positive_examples"] += int(label_any.sum().detach().cpu().item())
                    exact_counts["no_event_examples"] += int((~label_any).sum().detach().cpu().item())
                    exact_counts["no_event_false_positive_examples"] += int((~label_any & pred_any).sum().detach().cpu().item())

    diagnostics: dict[str, Any] = {
        "mode": mode,
        "category_threshold": default_threshold,
        "grid": grid,
        "beta": beta,
        "observed_examples": observed_examples,
        "max_examples": max_examples,
    }
    if observed_examples == 0:
        diagnostics["status"] = "no_calibration_examples"
        diagnostics["category_thresholds"] = thresholds
        return thresholds, diagnostics

    if mode == "per_token_fbeta_calibrated":
        per_token: dict[str, Any] = {}
        for token in category_vocab:
            best_key = (-1.0, -1.0, -1.0, default_threshold)
            best_threshold = default_threshold
            best_counts: dict[str, int] = {}
            for threshold in grid:
                counts = per_token_counts[token][threshold]
                score, precision, recall = _fbeta_score(counts["tp"], counts["fp"], counts["fn"], beta)
                key = (score, precision, recall, threshold)
                if key > best_key:
                    best_key = key
                    best_threshold = float(threshold)
                    best_counts = dict(counts)
                    best_counts.update({"score": score, "precision": precision, "recall": recall})
            thresholds[token] = best_threshold
            per_token[token] = {"threshold": best_threshold, **best_counts}
        diagnostics["per_token"] = per_token
    else:
        per_group: dict[str, Any] = {}
        for group, indices in groups.items():
            max_no_event_fpr = _group_no_event_fpr_cap(config, group)
            best_threshold, best_counts = _select_group_fbeta_threshold(
                per_group_exact_counts[group],
                default_threshold=default_threshold,
                beta=beta,
                max_no_event_fpr=max_no_event_fpr,
            )
            legacy_counts = per_group_counts[group].get(best_threshold, {})
            for idx in indices:
                thresholds[category_vocab[idx]] = best_threshold
            no_event_label = _group_threshold_label(group)
            per_group[group] = {
                "threshold": best_threshold,
                "token_count": len(indices),
                **best_counts,
                "token_level_counts": dict(legacy_counts),
                f"{no_event_label}_examples": best_counts.get("no_event_examples"),
                f"{no_event_label}_false_positive_examples": best_counts.get("no_event_false_positive_examples"),
                f"{no_event_label}_false_positive_rate": best_counts.get("no_event_false_positive_rate"),
                f"max_{no_event_label}_false_positive_rate": max_no_event_fpr,
            }
        diagnostics["per_group"] = per_group
    diagnostics["status"] = "computed"
    diagnostics["category_thresholds"] = thresholds
    return thresholds, diagnostics


def _calibrate_streaming_mouse_output_gain(
    torch,
    model,
    *,
    train_records: str | Path | Sequence[str | Path],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> tuple[float, dict[str, Any]]:
    configured_gain = float(config.get("mouse_output_gain", 1.0))
    mode = str(config.get("mouse_output_gain_mode", "fixed"))
    min_gain = float(config.get("mouse_output_gain_min", 0.25))
    max_gain = float(config.get("mouse_output_gain_max", 4.0))
    if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
        raise ValueError("mouse_output_gain_min/max must be positive and ordered")
    if mode == "fixed":
        return configured_gain, {
            "mode": "fixed",
            "configured_gain": configured_gain,
            "gain": configured_gain,
            "min_gain": min_gain,
            "max_gain": max_gain,
        }
    if mode not in {"train_abs_ratio", "train_abs_ratio_per_axis"}:
        raise ValueError(f"unsupported streaming mouse_output_gain_mode: {mode}")
    batch_size = int(config.get("mouse_gain_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048))))
    max_examples = config.get("mouse_gain_calibration_max_examples", config.get("category_calibration_max_examples"))
    category_threshold = float(config.get("category_threshold", 0.35))
    configured_thresholds = config.get("category_thresholds", {})
    category_thresholds = {
        token: float(configured_thresholds.get(token, category_threshold))
        for token in category_vocab
    } if isinstance(configured_thresholds, dict) else {token: category_threshold for token in category_vocab}
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device=device)
    predicted_abs_sum = 0.0
    target_abs_sum = 0.0
    predicted_abs_x_sum = 0.0
    predicted_abs_y_sum = 0.0
    target_abs_x_sum = 0.0
    target_abs_y_sum = 0.0
    value_count = 0
    mouse_target_mode = _mouse_target_mode(config)
    residual_mouse = _residual_mouse_from_mode(mouse_target_mode)
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    model.eval()
    with torch.no_grad():
        for rows in _iter_causal_feature_batches(
            train_records,
            batch_size,
            max_examples,
            feature_mode=str(stats["feature_mode"]),
            history_vocab=history_vocab,
            history_len=action_history_len,
            residual_mouse=residual_mouse,
        ):
            x = _batch_features(
                torch,
                rows,
                feature_mode=str(stats["feature_mode"]),
                mean=stats["mean"],
                std=stats["std"],
                device=device,
                mean_t=mean_t,
                std_t=std_t,
            )
            output_tensor = model(x).detach()
            if mouse_head_mode == "regression":
                predicted_mouse = output_tensor[:, :2]
                if residual_mouse:
                    baselines = torch.tensor(
                        [
                            [
                                float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[0]),
                                float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[1]),
                            ]
                            for row in rows
                        ],
                        dtype=torch.float32,
                        device=predicted_mouse.device,
                    )
                    predicted_mouse = predicted_mouse + baselines
                targets = torch.tensor(
                    [target_mouse_delta(row, mode=_base_mouse_target_mode(mouse_target_mode)) for row in rows],
                    dtype=torch.float32,
                    device=predicted_mouse.device,
                )
                batch_predicted_abs_x = float(predicted_mouse[:, 0].abs().sum().cpu())
                batch_predicted_abs_y = float(predicted_mouse[:, 1].abs().sum().cpu())
                batch_target_abs_x = float(targets[:, 0].abs().sum().cpu())
                batch_target_abs_y = float(targets[:, 1].abs().sum().cpu())
                predicted_abs_x_sum += batch_predicted_abs_x
                predicted_abs_y_sum += batch_predicted_abs_y
                target_abs_x_sum += batch_target_abs_x
                target_abs_y_sum += batch_target_abs_y
                predicted_abs_sum += batch_predicted_abs_x + batch_predicted_abs_y
                target_abs_sum += batch_target_abs_x + batch_target_abs_y
                value_count += int(predicted_mouse.numel())
                continue
            outputs = output_tensor.cpu().tolist()
            for row, output in zip(rows, outputs):
                dx, dy, _tokens = _prediction_from_output(
                    output,
                    base_dx=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[0]),
                    base_dy=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[1]),
                    residual_mouse=residual_mouse,
                    category_vocab=category_vocab,
                    category_thresholds=category_thresholds,
                    category_threshold=category_threshold,
                    keyboard_head_mode=str(config.get("keyboard_head_mode", "multilabel")),
                    keyboard_classes=_keyboard_classes_for_heads({"keyboard_class_counts": {}}, config),
                    keyboard_softmax_threshold=float(config.get("keyboard_softmax_threshold", 0.5)),
                    button_head_mode=str(config.get("button_head_mode", "multilabel")),
                    button_classes=_button_classes_for_heads({"button_class_counts": {}}, config),
                    button_softmax_threshold=float(config.get("button_softmax_threshold", 0.5)),
                    mouse_head_mode=mouse_head_mode,
                    mouse_axis_classes=mouse_axis_classes,
                    mouse_axis_decode_mode=str(config.get("mouse_axis_decode_mode", "expected")),
                    mouse_axis_temperature=float(config.get("mouse_axis_temperature", 1.0)),
                    mouse_output_gain=1.0,
                    mouse_output_gain_x=1.0,
                    mouse_output_gain_y=1.0,
                )
                target_dx, target_dy = target_mouse_delta(row, mode=_base_mouse_target_mode(mouse_target_mode))
                predicted_abs_sum += abs(float(dx)) + abs(float(dy))
                target_abs_sum += abs(float(target_dx)) + abs(float(target_dy))
                predicted_abs_x_sum += abs(float(dx))
                predicted_abs_y_sum += abs(float(dy))
                target_abs_x_sum += abs(float(target_dx))
                target_abs_y_sum += abs(float(target_dy))
                value_count += 2
    predicted_abs_mean = predicted_abs_sum / value_count if value_count else None
    target_abs_mean = target_abs_sum / value_count if value_count else None
    if not predicted_abs_mean or not target_abs_mean:
        return configured_gain, {
            "mode": mode,
            "status": "insufficient_nonzero_mouse",
            "configured_gain": configured_gain,
            "gain": configured_gain,
            "min_gain": min_gain,
            "max_gain": max_gain,
            "predicted_abs_mean": predicted_abs_mean,
            "target_abs_mean": target_abs_mean,
            "value_count": value_count,
        }
    raw_ratio = target_abs_mean / max(predicted_abs_mean, 1e-9)
    unclipped_gain = configured_gain * raw_ratio
    gain = min(max_gain, max(min_gain, unclipped_gain))
    raw_ratio_x = target_abs_x_sum / max(predicted_abs_x_sum, 1e-9)
    raw_ratio_y = target_abs_y_sum / max(predicted_abs_y_sum, 1e-9)
    gain_x = min(max_gain, max(min_gain, configured_gain * raw_ratio_x))
    gain_y = min(max_gain, max(min_gain, configured_gain * raw_ratio_y))
    if mode == "train_abs_ratio_per_axis":
        gain = (gain_x + gain_y) / 2.0
    return gain, {
        "mode": mode,
        "status": "computed",
        "configured_gain": configured_gain,
        "raw_ratio": raw_ratio,
        "unclipped_gain": unclipped_gain,
        "gain": gain,
        "raw_ratio_x": raw_ratio_x,
        "raw_ratio_y": raw_ratio_y,
        "gain_x": gain_x if mode == "train_abs_ratio_per_axis" else gain,
        "gain_y": gain_y if mode == "train_abs_ratio_per_axis" else gain,
        "min_gain": min_gain,
        "max_gain": max_gain,
        "predicted_abs_mean": predicted_abs_mean,
        "target_abs_mean": target_abs_mean,
        "predicted_abs_x_mean": predicted_abs_x_sum / max(1, value_count // 2),
        "predicted_abs_y_mean": predicted_abs_y_sum / max(1, value_count // 2),
        "target_abs_x_mean": target_abs_x_sum / max(1, value_count // 2),
        "target_abs_y_mean": target_abs_y_sum / max(1, value_count // 2),
        "value_count": value_count,
        "max_examples": max_examples,
    }


def _calibrate_streaming_category_thresholds_from_cache(
    torch,
    model,
    *,
    training_cache_manifests: Sequence[dict[str, Any]],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    default_threshold = float(config.get("category_threshold", 0.35))
    thresholds = {token: default_threshold for token in category_vocab}
    mode = str(config.get("category_threshold_mode", "global"))
    base_info: dict[str, Any] = {
        "mode": "global_threshold_streaming_cache" if mode == "global" else f"{mode}_cache",
        "category_threshold": default_threshold,
        "category_thresholds": thresholds,
        "source": "training_cache",
    }
    if mode == "global" or not category_vocab:
        return thresholds, base_info
    if mode not in {"per_token_fbeta_calibrated", "group_fbeta_calibrated"}:
        raise ValueError(f"unsupported streaming category_threshold_mode: {mode}")

    grid = _default_calibration_grid(config)
    beta = float(config.get("category_calibration_beta", 1.0))
    batch_size = int(config.get("category_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048))))
    max_examples = config.get("category_calibration_max_examples")
    groups = {
        group: [idx for idx, token in enumerate(category_vocab) if _stream_category_group(token) == group]
        for group in sorted({_stream_category_group(token) for token in category_vocab})
    }
    per_token_counts = {
        token: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in grid}
        for token in category_vocab
    }
    per_group_counts = {
        group: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in grid}
        for group in groups
    }
    per_group_exact_counts = {
        group: {
            threshold: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "predicted_positive": 0,
                "positive_examples": 0,
                "no_event_examples": 0,
                "no_event_false_positive_examples": 0,
            }
            for threshold in grid
        }
        for group in groups
    }
    observed_examples = 0
    model.eval()
    with torch.no_grad():
        for x, _mouse_y, cat_y, _keyboard_y, _button_y, _dx_y, _dy_y, batch_rows in _iter_training_cache_batches(
            torch,
            training_cache_manifests,
            batch_size=batch_size,
            device=device,
            max_examples=max_examples,
            rank=0,
            world_size=1,
            shard_by_path=False,
            shard_assignment=str(config.get("training_cache_shard_assignment", "greedy_rows")),
        ):
            outputs = model(x)
            category_end = 2 + len(category_vocab)
            probs = torch.sigmoid(outputs[:, 2:category_end]).detach()
            labels = cat_y.bool()
            observed_examples += int(batch_rows)
            for threshold in grid:
                pred = probs >= float(threshold)
                tp_vec = (pred & labels).sum(dim=0).detach().cpu().tolist()
                fp_vec = (pred & ~labels).sum(dim=0).detach().cpu().tolist()
                fn_vec = (~pred & labels).sum(dim=0).detach().cpu().tolist()
                for idx, token in enumerate(category_vocab):
                    counts = per_token_counts[token][threshold]
                    counts["tp"] += int(tp_vec[idx])
                    counts["fp"] += int(fp_vec[idx])
                    counts["fn"] += int(fn_vec[idx])
                for group, indices in groups.items():
                    counts = per_group_counts[group][threshold]
                    counts["tp"] += sum(int(tp_vec[idx]) for idx in indices)
                    counts["fp"] += sum(int(fp_vec[idx]) for idx in indices)
                    counts["fn"] += sum(int(fn_vec[idx]) for idx in indices)
                    pred_group = pred[:, indices]
                    label_group = labels[:, indices]
                    pred_any = pred_group.any(dim=1)
                    label_any = label_group.any(dim=1)
                    exact_match = (pred_group == label_group).all(dim=1)
                    exact_counts = per_group_exact_counts[group][threshold]
                    exact_counts["tp"] += int((label_any & exact_match).sum().detach().cpu().item())
                    exact_counts["fn"] += int((label_any & ~exact_match).sum().detach().cpu().item())
                    exact_counts["fp"] += int(((~label_any & pred_any) | (label_any & pred_any & ~exact_match)).sum().detach().cpu().item())
                    exact_counts["predicted_positive"] += int(pred_any.sum().detach().cpu().item())
                    exact_counts["positive_examples"] += int(label_any.sum().detach().cpu().item())
                    exact_counts["no_event_examples"] += int((~label_any).sum().detach().cpu().item())
                    exact_counts["no_event_false_positive_examples"] += int((~label_any & pred_any).sum().detach().cpu().item())

    diagnostics: dict[str, Any] = {
        "mode": mode,
        "source": "training_cache",
        "category_threshold": default_threshold,
        "grid": grid,
        "beta": beta,
        "observed_examples": observed_examples,
        "max_examples": max_examples,
    }
    if observed_examples == 0:
        diagnostics["status"] = "no_calibration_examples"
        diagnostics["category_thresholds"] = thresholds
        return thresholds, diagnostics
    if mode == "per_token_fbeta_calibrated":
        per_token: dict[str, Any] = {}
        for token in category_vocab:
            best_key = (-1.0, -1.0, -1.0, default_threshold)
            best_threshold = default_threshold
            best_counts: dict[str, int] = {}
            for threshold in grid:
                counts = per_token_counts[token][threshold]
                score, precision, recall = _fbeta_score(counts["tp"], counts["fp"], counts["fn"], beta)
                key = (score, precision, recall, threshold)
                if key > best_key:
                    best_key = key
                    best_threshold = float(threshold)
                    best_counts = dict(counts)
                    best_counts.update({"score": score, "precision": precision, "recall": recall})
            thresholds[token] = best_threshold
            per_token[token] = {"threshold": best_threshold, **best_counts}
        diagnostics["per_token"] = per_token
    else:
        per_group: dict[str, Any] = {}
        for group, indices in groups.items():
            max_no_event_fpr = _group_no_event_fpr_cap(config, group)
            best_threshold, best_counts = _select_group_fbeta_threshold(
                per_group_exact_counts[group],
                default_threshold=default_threshold,
                beta=beta,
                max_no_event_fpr=max_no_event_fpr,
            )
            legacy_counts = per_group_counts[group].get(best_threshold, {})
            for idx in indices:
                thresholds[category_vocab[idx]] = best_threshold
            no_event_label = _group_threshold_label(group)
            per_group[group] = {
                "threshold": best_threshold,
                "token_count": len(indices),
                **best_counts,
                "token_level_counts": dict(legacy_counts),
                f"{no_event_label}_examples": best_counts.get("no_event_examples"),
                f"{no_event_label}_false_positive_examples": best_counts.get("no_event_false_positive_examples"),
                f"{no_event_label}_false_positive_rate": best_counts.get("no_event_false_positive_rate"),
                f"max_{no_event_label}_false_positive_rate": max_no_event_fpr,
            }
        diagnostics["per_group"] = per_group
    diagnostics["status"] = "computed"
    diagnostics["category_thresholds"] = thresholds
    return thresholds, diagnostics


def _calibrate_streaming_mouse_output_gain_from_cache(
    torch,
    model,
    *,
    training_cache_manifests: Sequence[dict[str, Any]],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> tuple[float, dict[str, Any]]:
    configured_gain = float(config.get("mouse_output_gain", 1.0))
    mode = str(config.get("mouse_output_gain_mode", "fixed"))
    min_gain = float(config.get("mouse_output_gain_min", 0.25))
    max_gain = float(config.get("mouse_output_gain_max", 4.0))
    if mode == "fixed":
        return configured_gain, {
            "mode": "fixed",
            "source": "training_cache",
            "configured_gain": configured_gain,
            "gain": configured_gain,
            "min_gain": min_gain,
            "max_gain": max_gain,
        }
    if mode not in {"train_abs_ratio", "train_abs_ratio_per_axis"}:
        raise ValueError(f"unsupported streaming mouse_output_gain_mode: {mode}")
    batch_size = int(config.get("mouse_gain_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048))))
    max_examples = config.get("mouse_gain_calibration_max_examples", config.get("category_calibration_max_examples"))
    category_threshold = float(config.get("category_threshold", 0.35))
    configured_thresholds = config.get("category_thresholds", {})
    category_thresholds = {
        token: float(configured_thresholds.get(token, category_threshold))
        for token in category_vocab
    } if isinstance(configured_thresholds, dict) else {token: category_threshold for token in category_vocab}
    predicted_abs_sum = 0.0
    target_abs_sum = 0.0
    predicted_abs_x_sum = 0.0
    predicted_abs_y_sum = 0.0
    target_abs_x_sum = 0.0
    target_abs_y_sum = 0.0
    value_count = 0
    model.eval()
    with torch.no_grad():
        for x, mouse_y, _cat_y, _keyboard_y, _button_y, _dx_y, _dy_y, _batch_rows in _iter_training_cache_batches(
            torch,
            training_cache_manifests,
            batch_size=batch_size,
            device=device,
            max_examples=max_examples,
            rank=0,
            world_size=1,
            shard_by_path=False,
            shard_assignment=str(config.get("training_cache_shard_assignment", "greedy_rows")),
        ):
            output_tensor = model(x).detach()
            mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
            if mouse_head_mode == "regression":
                predicted_mouse = output_tensor[:, :2]
                targets_tensor = mouse_y.detach()
                batch_predicted_abs_x = float(predicted_mouse[:, 0].abs().sum().cpu())
                batch_predicted_abs_y = float(predicted_mouse[:, 1].abs().sum().cpu())
                batch_target_abs_x = float(targets_tensor[:, 0].abs().sum().cpu())
                batch_target_abs_y = float(targets_tensor[:, 1].abs().sum().cpu())
                predicted_abs_x_sum += batch_predicted_abs_x
                predicted_abs_y_sum += batch_predicted_abs_y
                target_abs_x_sum += batch_target_abs_x
                target_abs_y_sum += batch_target_abs_y
                predicted_abs_sum += batch_predicted_abs_x + batch_predicted_abs_y
                target_abs_sum += batch_target_abs_x + batch_target_abs_y
                value_count += int(predicted_mouse.numel())
                continue
            outputs = output_tensor.cpu().tolist()
            targets = mouse_y.detach().cpu().tolist()
            for target, output in zip(targets, outputs):
                dx, dy, _tokens = _prediction_from_output(
                    output,
                    base_dx=0.0,
                    base_dy=0.0,
                    residual_mouse=False,
                    category_vocab=category_vocab,
                    category_thresholds=category_thresholds,
                    category_threshold=category_threshold,
                    keyboard_head_mode=str(config.get("keyboard_head_mode", "multilabel")),
                    keyboard_classes=_keyboard_classes_for_heads({"keyboard_class_counts": {}}, config),
                    keyboard_softmax_threshold=float(config.get("keyboard_softmax_threshold", 0.5)),
                    button_head_mode=str(config.get("button_head_mode", "multilabel")),
                    button_classes=_button_classes_for_heads({"button_class_counts": {}}, config),
                    button_softmax_threshold=float(config.get("button_softmax_threshold", 0.5)),
                    mouse_head_mode=mouse_head_mode,
                    mouse_axis_classes=mouse_axis_classes,
                    mouse_axis_decode_mode=str(config.get("mouse_axis_decode_mode", "expected")),
                    mouse_axis_temperature=float(config.get("mouse_axis_temperature", 1.0)),
                    mouse_output_gain=1.0,
                    mouse_output_gain_x=1.0,
                    mouse_output_gain_y=1.0,
                )
                target_dx, target_dy = target
                predicted_abs_sum += abs(float(dx)) + abs(float(dy))
                target_abs_sum += abs(float(target_dx)) + abs(float(target_dy))
                predicted_abs_x_sum += abs(float(dx))
                predicted_abs_y_sum += abs(float(dy))
                target_abs_x_sum += abs(float(target_dx))
                target_abs_y_sum += abs(float(target_dy))
                value_count += 2
    predicted_abs_mean = predicted_abs_sum / value_count if value_count else None
    target_abs_mean = target_abs_sum / value_count if value_count else None
    if not predicted_abs_mean or not target_abs_mean:
        return configured_gain, {
            "mode": mode,
            "source": "training_cache",
            "status": "insufficient_nonzero_mouse",
            "configured_gain": configured_gain,
            "gain": configured_gain,
            "min_gain": min_gain,
            "max_gain": max_gain,
            "predicted_abs_mean": predicted_abs_mean,
            "target_abs_mean": target_abs_mean,
            "value_count": value_count,
        }
    raw_ratio = target_abs_mean / max(predicted_abs_mean, 1e-9)
    unclipped_gain = configured_gain * raw_ratio
    gain = min(max_gain, max(min_gain, unclipped_gain))
    raw_ratio_x = target_abs_x_sum / max(predicted_abs_x_sum, 1e-9)
    raw_ratio_y = target_abs_y_sum / max(predicted_abs_y_sum, 1e-9)
    gain_x = min(max_gain, max(min_gain, configured_gain * raw_ratio_x))
    gain_y = min(max_gain, max(min_gain, configured_gain * raw_ratio_y))
    if mode == "train_abs_ratio_per_axis":
        gain = (gain_x + gain_y) / 2.0
    return gain, {
        "mode": mode,
        "source": "training_cache",
        "status": "computed",
        "configured_gain": configured_gain,
        "raw_ratio": raw_ratio,
        "unclipped_gain": unclipped_gain,
        "gain": gain,
        "raw_ratio_x": raw_ratio_x,
        "raw_ratio_y": raw_ratio_y,
        "gain_x": gain_x if mode == "train_abs_ratio_per_axis" else gain,
        "gain_y": gain_y if mode == "train_abs_ratio_per_axis" else gain,
        "min_gain": min_gain,
        "max_gain": max_gain,
        "predicted_abs_mean": predicted_abs_mean,
        "target_abs_mean": target_abs_mean,
        "predicted_abs_x_mean": predicted_abs_x_sum / max(1, value_count // 2),
        "predicted_abs_y_mean": predicted_abs_y_sum / max(1, value_count // 2),
        "target_abs_x_mean": target_abs_x_sum / max(1, value_count // 2),
        "target_abs_y_mean": target_abs_y_sum / max(1, value_count // 2),
        "value_count": value_count,
        "max_examples": max_examples,
    }


def _metric_path_value(metrics: dict[str, Any], path: str) -> float | None:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    return float(current)


def _count_jsonl_rows(path: str | Path) -> int:
    with Path(path).open() as handle:
        return sum(1 for line in handle if line.strip())


def _iter_records(path: str | Path | Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    for record_path in _record_paths_from_value(path):
        yield from iter_jsonl(record_path)


def _observe_prediction_metrics(
    *,
    row: dict[str, Any],
    tokens: list[str],
    stats: dict[str, Any],
    model_name: str,
    baseline_names: list[str],
    metrics_by_model: dict[str, StreamingActionMetrics],
    group_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]],
    cluster_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]],
) -> None:
    model_tokens = {model_name: tokens}
    for baseline_name in baseline_names:
        model_tokens[baseline_name] = _baseline_tokens(baseline_name, row, stats)
    cluster = str(row.get("recording_id") or row.get("cross_resolution_key") or row.get("sequence_id"))
    for name, pred_tokens in model_tokens.items():
        metrics_by_model[name].update(pred_tokens, row)
        _ensure_metric(cluster_metrics_by_model[name], cluster).update(pred_tokens, row)
        for group_key in _group_keys(row):
            _ensure_metric(group_metrics_by_model[name], group_key).update(pred_tokens, row)


def _file_artifact_metadata(path_text: str | Path | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": 0}
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _json_fingerprint(path_text: str | Path | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = read_json(path)
        return str(payload.get("dataset_fingerprint") or payload.get("split_contract_fingerprint") or stable_hash_json(payload))
    except Exception:
        return None


def _convergence_score(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode and mode != "composite_primary":
        value = _metric_path_value(metrics, mode)
        return {"mode": mode, "value": value, "components": {mode: value}}
    components = {
        "keyboard_accuracy": _metric_path_value(metrics, "keyboard.accuracy"),
        "mouse_button_f1": _metric_path_value(metrics, "mouse_button.f1"),
        "mouse_button_accuracy": _metric_path_value(metrics, "mouse_button.accuracy"),
        "mouse_move_pearson": _metric_path_value(metrics, "mouse_move.pearson"),
    }
    values = [
        value
        for key, value in components.items()
        if value is not None and key != "mouse_button_accuracy"
    ]
    if components["mouse_button_f1"] is None and components["mouse_button_accuracy"] is not None:
        values.append(components["mouse_button_accuracy"])
    return {
        "mode": "composite_primary",
        "value": sum(values) / len(values) if values else None,
        "components": components,
    }


def _evaluate_stream_metrics(
    torch,
    model,
    *,
    target_records: str | Path | Sequence[str | Path],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
) -> dict[str, Any]:
    metric = StreamingActionMetrics()
    batch_size = int(config.get("convergence_eval_batch_size", config.get("eval_batch_size", config.get("batch_size", 2048))))
    max_examples = config.get("convergence_eval_max_examples")
    count = 0
    model.eval()
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device=device)
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    mouse_target_mode = _mouse_target_mode(config)
    residual_mouse = _residual_mouse_from_mode(mouse_target_mode)
    histories: dict[str, list[list[str]]] = {}
    button_states: dict[str, dict[str, float]] = {}
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    if action_history_len > 0 and config.get("train_records"):
        train_paths = _record_paths_from_config(
            config,
            primary_key="train_records",
            paths_key="train_record_paths",
            glob_key="train_records_glob",
        )
        for train_row in _iter_records(train_paths):
            history, button_state = _ensure_history_state(histories, button_states, str(train_row.get("recording_id", "")))
            _append_history(history, button_state, _tokens(train_row), history_len=action_history_len)
    if residual_mouse:
        mouse_seed_state = _load_streaming_mouse_seed_state(config.get("_streaming_mouse_seed_state"))
        if mouse_seed_state is not None:
            last_mouse_by_recording, last_mouse_by_game, last_mouse_fallback = mouse_seed_state
        elif config.get("train_records") or config.get("train_record_paths") or config.get("train_records_glob"):
            train_paths = _record_paths_from_config(
                config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            last_mouse_by_recording, last_mouse_by_game, last_mouse_fallback = _seed_streaming_mouse_delta_state(
                train_paths,
                num_workers=int(config.get("streaming_mouse_seed_state_workers", 1) or 1),
            )
    with torch.no_grad():
        if action_history_len > 0 or residual_mouse:
            def causal_batches() -> Iterable[list[dict[str, Any]]]:
                emitted = 0
                for row in _iter_records(target_records):
                    if max_examples is not None and emitted >= int(max_examples):
                        break
                    if action_history_len > 0:
                        history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
                        features = _record_features_with_history(
                            row,
                            feature_mode=str(stats["feature_mode"]),
                            history_vocab=history_vocab,
                            history_len=action_history_len,
                            history=history,
                            button_state=button_state,
                        )
                    else:
                        features = _base_record_features(row, feature_mode=str(stats["feature_mode"]))
                    mouse_baseline = _mouse_baseline_for_row(
                        row,
                        last_by_recording=last_mouse_by_recording,
                        last_by_game=last_mouse_by_game,
                        fallback=last_mouse_fallback,
                    )
                    features = _features_with_optional_mouse_baseline(
                        row,
                        features,
                        residual_mouse=residual_mouse,
                        baseline=mouse_baseline,
                    )
                    emitted += 1
                    yield [
                        _row_with_mouse_baseline(
                            row,
                            features=features,
                            residual_mouse=residual_mouse,
                            baseline=mouse_baseline,
                        )
                    ]

            batches = causal_batches()
        else:
            batches = _iter_batches(target_records, batch_size, max_examples)
        for rows in batches:
            if max_examples is not None and count >= int(max_examples):
                break
            x = _batch_features(
                torch,
                rows,
                feature_mode=str(stats["feature_mode"]),
                mean=stats["mean"],
                std=stats["std"],
                device=device,
                mean_t=mean_t,
                std_t=std_t,
            )
            outputs = model(x).detach().cpu().tolist()
            for row, output in zip(rows, outputs):
                tokens = _predicted_tokens_from_output(
                    output,
                    config=config,
                    category_vocab=category_vocab,
                    mouse_axis_classes=mouse_axis_classes,
                    base_dx=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[0]),
                    base_dy=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[1]),
                    residual_mouse=residual_mouse,
                )
                if action_history_len > 0:
                    history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
                    _append_history(history, button_state, tokens, history_len=action_history_len)
                if residual_mouse:
                    last_mouse_fallback = _update_mouse_baseline_state(
                        row,
                        _mouse_delta_from_tokens(tokens),
                        last_by_recording=last_mouse_by_recording,
                        last_by_game=last_mouse_by_game,
                    )
                metric.update(tokens, row)
                count += 1
    metrics = metric.payload()
    score = _convergence_score(metrics, str(config.get("convergence_score", "composite_primary")))
    return {"target_records": count, "metrics": metrics, "score": score}


def _convergence_report(history: list[dict[str, Any]], config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    patience = int(config.get("plateau_patience", 3))
    min_relative_improvement = float(config.get("plateau_min_relative_improvement", 0.01))
    validation_rows = [
        row
        for row in history
        if isinstance(row.get("validation"), dict)
        and row["validation"].get("score", {}).get("value") is not None
    ]
    values = [float(row["validation"]["score"]["value"]) for row in validation_rows]
    recent_relative_improvements: list[float] = []
    plateau_met = False
    if len(values) >= patience + 1:
        recent = values[-(patience + 1) :]
        for prev, curr in zip(recent, recent[1:]):
            recent_relative_improvements.append((curr - prev) / max(abs(prev), 1e-9))
        plateau_met = all(value < min_relative_improvement for value in recent_relative_improvements)
    report = {
        "schema": "streaming_convergence_report.v1",
        "score_mode": str(config.get("convergence_score", "composite_primary")),
        "direction": "higher",
        "eval_interval_epochs": int(config.get("eval_interval_epochs", 0)),
        "patience": patience,
        "min_relative_improvement": min_relative_improvement,
        "plateau_met": plateau_met,
        "num_validation_checkpoints": len(validation_rows),
        "recent_relative_improvements": recent_relative_improvements,
        "history": [
            {
                "epoch": row["epoch"],
                "train_loss": row.get("loss"),
                "validation_score": row.get("validation", {}).get("score"),
                "validation_examples": row.get("validation", {}).get("target_records"),
            }
            for row in history
        ],
        "report_path": str(output_dir / "convergence_report.json"),
    }
    return report


def _predict_stream(
    torch,
    model,
    *,
    target_records: str | Path | Sequence[str | Path],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model_name = str(config.get("model_name", "streaming_compact_idm"))
    baseline_names = [str(name) for name in config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])]
    all_model_names = [model_name, *baseline_names]
    metrics_by_model = {name: StreamingActionMetrics() for name in all_model_names}
    group_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    cluster_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 2048)))
    max_target_examples = config.get("max_target_examples")
    model.eval()
    target_count = 0
    target_source_ids: set[str] = set()
    target_resolution_tiers: set[str] = set()
    target_eval_split_tags: set[str] = set()
    validate_pseudolabels = bool(config.get("validate_pseudolabels", True))
    sequence_fingerprint = hashlib.sha256()
    pseudo_path = output_dir / "pseudolabels.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    resume_predictions = bool(config.get("resume_predictions", False))
    resume_existing_rows = 0
    action_history_len = int(stats.get("action_history_len", _action_history_len_from_config(config)))
    history_vocab = [str(token) for token in stats.get("action_history_vocab", category_vocab)]
    mouse_target_mode = _mouse_target_mode(config)
    residual_mouse = _residual_mouse_from_mode(mouse_target_mode)
    seed_state_provided = isinstance(config.get("_action_history_seed_state"), dict)
    histories, button_states = _load_action_history_seed_state(config.get("_action_history_seed_state"))
    last_mouse_by_recording: dict[str, tuple[float, float]] = {}
    last_mouse_by_game: dict[str, tuple[float, float]] = {}
    last_mouse_fallback = (0.0, 0.0)
    if action_history_len > 0:
        if not histories and not seed_state_provided:
            train_paths = _record_paths_from_config(
                config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            seed_state = _action_history_seed_state_from_records(train_paths, history_len=action_history_len)
            histories, button_states = _load_action_history_seed_state(seed_state)
    if residual_mouse:
        mouse_seed_state = _load_streaming_mouse_seed_state(config.get("_streaming_mouse_seed_state"))
        if mouse_seed_state is not None:
            last_mouse_by_recording, last_mouse_by_game, last_mouse_fallback = mouse_seed_state
        elif config.get("train_records") or config.get("train_record_paths") or config.get("train_records_glob"):
            train_paths = _record_paths_from_config(
                config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            last_mouse_by_recording, last_mouse_by_game, last_mouse_fallback = _seed_streaming_mouse_delta_state(train_paths)
    closed_loop_state_context = _closed_loop_state_context_enabled(config, stats)
    closed_loop_tracker: _ClosedLoopStateContextTracker | None = None
    closed_loop_seed_rows = 0
    if closed_loop_state_context:
        seed_from_train = bool(config.get("closed_loop_state_context_seed_from_train", True))
        if seed_from_train and (
            config.get("train_records") or config.get("train_record_paths") or config.get("train_records_glob")
        ):
            train_paths = _record_paths_from_config(
                config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            seed_max = config.get("closed_loop_state_context_seed_max_examples")
            closed_loop_tracker = _seed_closed_loop_state_context_from_records(
                train_paths,
                max_examples=None if seed_max is None else int(seed_max),
            )
            closed_loop_seed_rows = int(closed_loop_tracker.seed_rows)
        else:
            closed_loop_tracker = _ClosedLoopStateContextTracker()
    if resume_predictions:
        pseudo_exists = pseudo_path.exists()
        predictions_exists = predictions_path.exists()
        if pseudo_exists != predictions_exists:
            raise ValueError(
                "resume_predictions requires pseudolabels and predictions to either both exist or both be absent: "
                f"pseudolabels={pseudo_exists} predictions={predictions_exists}"
            )
        if pseudo_exists and predictions_exists:
            pseudo_rows = _count_jsonl_rows(pseudo_path)
            prediction_rows = _count_jsonl_rows(predictions_path)
            if pseudo_rows != prediction_rows:
                raise ValueError(
                    f"resume_predictions found mismatched output row counts: "
                    f"pseudolabels={pseudo_rows} predictions={prediction_rows}"
                )
            if max_target_examples is not None and prediction_rows > int(max_target_examples):
                raise ValueError(
                    f"resume_predictions found {prediction_rows} existing rows, exceeding max_target_examples={max_target_examples}"
                )
            resume_existing_rows = prediction_rows
            if resume_existing_rows:
                if closed_loop_state_context and not bool(config.get("resume_closed_loop_state_context", False)):
                    raise ValueError(
                        "closed_loop_state_context requires fresh target inference by default; "
                        "rerun with resume_predictions=false/--no-resume-predictions or set "
                        "resume_closed_loop_state_context=true only for a confirmed closed-loop partial output"
                    )
                target_iter = _iter_records(target_records)
                for row_idx, (pseudo_row, pred_row) in enumerate(zip(iter_jsonl(pseudo_path), iter_jsonl(predictions_path)), 1):
                    try:
                        row = next(target_iter)
                    except StopIteration as exc:
                        raise ValueError(
                            f"resume_predictions has more predictions than target records at row {row_idx}"
                        ) from exc
                    if str(row.get("sequence_id")) != str(pred_row.get("sequence_id")):
                        raise ValueError(
                            f"resume_predictions sequence_id mismatch at row {row_idx}: "
                            f"{row.get('sequence_id')!r} != {pred_row.get('sequence_id')!r}"
                        )
                    if str(pseudo_row.get("sequence_id")) != str(pred_row.get("sequence_id")):
                        raise ValueError(
                            f"resume_predictions pseudolabel/prediction sequence_id mismatch at row {row_idx}: "
                            f"{pseudo_row.get('sequence_id')!r} != {pred_row.get('sequence_id')!r}"
                        )
                    if str(pseudo_row.get("model")) != model_name:
                        raise ValueError(
                            f"resume_predictions model mismatch at row {row_idx}: "
                            f"{pseudo_row.get('model')!r} != {model_name!r}"
                        )
                    if str(pseudo_row.get("training_split_hash")) != str(stats["dataset_fingerprint"]):
                        raise ValueError(
                            f"resume_predictions training_split_hash mismatch at row {row_idx}: "
                            f"{pseudo_row.get('training_split_hash')!r} != {stats['dataset_fingerprint']!r}"
                        )
                    if row.get("source_id") is not None:
                        target_source_ids.add(str(row["source_id"]))
                    if row.get("resolution_tier") is not None:
                        target_resolution_tiers.add(str(row["resolution_tier"]))
                    for tag in row.get("eval_split_tags", []) or []:
                        target_eval_split_tags.add(str(tag))
                    tokens = [str(token) for token in pred_row.get("predicted_tokens", [])]
                    if [str(token) for token in pseudo_row.get("predicted_tokens", [])] != tokens:
                        raise ValueError(f"resume_predictions token mismatch at row {row_idx}")
                    if closed_loop_tracker is not None:
                        closed_loop_tracker.observe_tokens(row, tokens)
                    if action_history_len > 0:
                        history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
                        _append_history(history, button_state, tokens, history_len=action_history_len)
                    if residual_mouse:
                        last_mouse_fallback = _update_mouse_baseline_state(
                            row,
                            _mouse_delta_from_tokens(tokens),
                            last_by_recording=last_mouse_by_recording,
                            last_by_game=last_mouse_by_game,
                        )
                    _observe_prediction_metrics(
                        row=row,
                        tokens=tokens,
                        stats=stats,
                        model_name=model_name,
                        baseline_names=baseline_names,
                        metrics_by_model=metrics_by_model,
                        group_metrics_by_model=group_metrics_by_model,
                        cluster_metrics_by_model=cluster_metrics_by_model,
                    )
                    sequence_fingerprint.update(json.dumps({"id": row["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                    sequence_fingerprint.update(b"\n")
                target_count = resume_existing_rows
    write_mode = "a" if resume_existing_rows else "w"
    mean_t, std_t = _normalizer_tensors(torch, mean=stats["mean"], std=stats["std"], device=device)
    remaining_max_examples = None
    if max_target_examples is not None:
        remaining_max_examples = max(0, int(max_target_examples) - int(resume_existing_rows))
    category_threshold = float(config.get("category_threshold", 0.35))
    configured_category_thresholds = config.get("category_thresholds", {})
    category_thresholds = (
        {
            token: float(configured_category_thresholds.get(token, category_threshold))
            for token in category_vocab
        }
        if isinstance(configured_category_thresholds, dict)
        else {token: category_threshold for token in category_vocab}
    )
    keyboard_head_mode = str(config.get("keyboard_head_mode", "multilabel"))
    button_head_mode = str(config.get("button_head_mode", "multilabel"))
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    keyboard_classes = _keyboard_classes_for_heads({"keyboard_class_counts": {}}, config)
    button_classes = _button_classes_for_heads({"button_class_counts": {}}, config)
    fast_regression_softmax_decode = (
        mouse_head_mode == "regression"
        and keyboard_head_mode == "softmax"
        and button_head_mode == "softmax"
        and bool(keyboard_classes)
        and bool(button_classes)
    )
    category_end = 2 + len(category_vocab)
    keyboard_end = category_end + (len(keyboard_classes) if keyboard_head_mode == "softmax" else 0)
    button_end = keyboard_end + (len(button_classes) if button_head_mode == "softmax" else 0)
    keyboard_softmax_threshold = float(config.get("keyboard_softmax_threshold", 0.5))
    button_softmax_threshold = float(config.get("button_softmax_threshold", 0.5))
    mouse_output_gain = float(config.get("mouse_output_gain", 1.0))
    mouse_output_gain_x = float(config["mouse_output_gain_x"]) if config.get("mouse_output_gain_x") is not None else mouse_output_gain
    mouse_output_gain_y = float(config["mouse_output_gain_y"]) if config.get("mouse_output_gain_y") is not None else mouse_output_gain
    mouse_emit_mode = str(config.get("mouse_emit_mode", "single"))
    mouse_max_tokens_per_axis = int(config.get("mouse_max_tokens_per_axis", 8))

    def prediction_batches() -> Iterable[list[dict[str, Any]]]:
        if not closed_loop_state_context and action_history_len <= 0 and not residual_mouse:
            yield from _iter_batches(
                target_records,
                batch_size,
                remaining_max_examples,
                skip_examples=resume_existing_rows,
            )
            return
        row_iter = _iter_records(target_records)
        for _ in range(resume_existing_rows):
            next(row_iter, None)
        emitted = 0
        for row in row_iter:
            if remaining_max_examples is not None and emitted >= remaining_max_examples:
                break
            feature_row = closed_loop_tracker.row_with_prior_context(row) if closed_loop_tracker is not None else row
            if action_history_len > 0:
                history, button_state = _ensure_history_state(histories, button_states, str(feature_row.get("recording_id", "")))
                features = _record_features_with_history(
                    feature_row,
                    feature_mode=str(stats["feature_mode"]),
                    history_vocab=history_vocab,
                    history_len=action_history_len,
                    history=history,
                    button_state=button_state,
                )
            else:
                features = _base_record_features(feature_row, feature_mode=str(stats["feature_mode"]))
            mouse_baseline = _mouse_baseline_for_row(
                feature_row,
                last_by_recording=last_mouse_by_recording,
                last_by_game=last_mouse_by_game,
                fallback=last_mouse_fallback,
            )
            features = _features_with_optional_mouse_baseline(
                feature_row,
                features,
                residual_mouse=residual_mouse,
                baseline=mouse_baseline,
            )
            emitted += 1
            yield [
                _row_with_mouse_baseline(
                    feature_row,
                    features=features,
                    residual_mouse=residual_mouse,
                    baseline=mouse_baseline,
                )
            ]

    with pseudo_path.open(write_mode) as pseudo_f, predictions_path.open(write_mode) as pred_f, torch.no_grad():
        for rows in prediction_batches():
            x = _batch_features(
                torch,
                rows,
                feature_mode=str(stats["feature_mode"]),
                mean=stats["mean"],
                std=stats["std"],
                device=device,
                mean_t=mean_t,
                std_t=std_t,
            )
            output_tensor = model(x).detach()
            fast_batch: dict[str, Any] | None = None
            if fast_regression_softmax_decode:
                dx_values = (output_tensor[:, 0] * mouse_output_gain_x).detach()
                dy_values = (output_tensor[:, 1] * mouse_output_gain_y).detach()
                if residual_mouse:
                    baselines = torch.tensor(
                        [
                            [
                                float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[0]),
                                float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[1]),
                            ]
                            for row in rows
                        ],
                        dtype=torch.float32,
                        device=output_tensor.device,
                    )
                    dx_values = dx_values + baselines[:, 0]
                    dy_values = dy_values + baselines[:, 1]
                keyboard_logits = output_tensor[:, category_end:keyboard_end]
                keyboard_probs, keyboard_indices = torch.nn.functional.softmax(keyboard_logits, dim=1).max(dim=1)
                button_logits = output_tensor[:, keyboard_end:button_end]
                button_probs, button_indices = torch.nn.functional.softmax(button_logits, dim=1).max(dim=1)
                if category_vocab:
                    category_probs = torch.sigmoid(output_tensor[:, 2:category_end]).detach().cpu().tolist()
                else:
                    category_probs = [[] for _ in rows]
                fast_batch = {
                    "dx": dx_values.cpu().tolist(),
                    "dy": dy_values.cpu().tolist(),
                    "keyboard_indices": keyboard_indices.cpu().tolist(),
                    "keyboard_probs": keyboard_probs.cpu().tolist(),
                    "button_indices": button_indices.cpu().tolist(),
                    "button_probs": button_probs.cpu().tolist(),
                    "category_probs": category_probs,
                }
                outputs: list[list[float]] = []
            else:
                outputs = output_tensor.cpu().tolist()
            for row_index, row in enumerate(rows):
                if row.get("source_id") is not None:
                    target_source_ids.add(str(row["source_id"]))
                if row.get("resolution_tier") is not None:
                    target_resolution_tiers.add(str(row["resolution_tier"]))
                for tag in row.get("eval_split_tags", []) or []:
                    target_eval_split_tags.add(str(tag))
                if fast_batch is not None:
                    tokens = tokens_from_delta(
                        float(fast_batch["dx"][row_index]),
                        float(fast_batch["dy"][row_index]),
                        emit_mode=mouse_emit_mode,
                        max_tokens_per_axis=mouse_max_tokens_per_axis,
                    )
                    for token, prob in zip(category_vocab, fast_batch["category_probs"][row_index]):
                        if float(prob) >= float(category_thresholds.get(token, category_threshold)):
                            tokens.append(token)
                    keyboard_idx = int(fast_batch["keyboard_indices"][row_index])
                    if keyboard_idx != 0 and float(fast_batch["keyboard_probs"][row_index]) >= keyboard_softmax_threshold:
                        tokens.extend(keyboard_classes[keyboard_idx])
                    button_idx = int(fast_batch["button_indices"][row_index])
                    if button_idx != 0 and float(fast_batch["button_probs"][row_index]) >= button_softmax_threshold:
                        tokens.extend(button_classes[button_idx])
                else:
                    output = outputs[row_index]
                    tokens = _predicted_tokens_from_output(
                        output,
                        config=config,
                        category_vocab=category_vocab,
                        mouse_axis_classes=mouse_axis_classes,
                        base_dx=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[0]),
                        base_dy=float((row.get("__streaming_mouse_baseline") or [0.0, 0.0])[1]),
                        residual_mouse=residual_mouse,
                    )
                if closed_loop_tracker is not None:
                    closed_loop_tracker.observe_tokens(row, tokens)
                if action_history_len > 0:
                    history, button_state = _ensure_history_state(histories, button_states, str(row.get("recording_id", "")))
                    _append_history(history, button_state, tokens, history_len=action_history_len)
                if residual_mouse:
                    last_mouse_fallback = _update_mouse_baseline_state(
                        row,
                        _mouse_delta_from_tokens(tokens),
                        last_by_recording=last_mouse_by_recording,
                        last_by_game=last_mouse_by_game,
                    )
                confidence = max(0.05, min(0.99, 1.0 / (1.0 + len(tokens))))
                pseudo = {
                    "schema": "idm_pseudolabel.v1",
                    "sequence_id": row["sequence_id"],
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "predicted_tokens": tokens,
                    "label_source": "idm_generated",
                    "confidence": confidence,
                    "model": model_name,
                    "training_split_hash": str(stats["dataset_fingerprint"]),
                    "input_window": {"frame_ref": row.get("frame", {}).get("path", ""), "frame_index": int(row.get("frame", {}).get("index", 0))},
                }
                if closed_loop_state_context:
                    pseudo["state_context_source"] = _ClosedLoopStateContextTracker.source
                if validate_pseudolabels:
                    validate_named(pseudo, "idm_pseudolabel.schema.json")
                pred = {
                    "sequence_id": row["sequence_id"],
                    "recording_id": row.get("recording_id"),
                    "cross_resolution_key": row.get("cross_resolution_key"),
                    "game": row.get("game"),
                    "timestamp_ns": row["timestamp_ns"],
                    "predicted_tokens": tokens,
                }
                if closed_loop_state_context:
                    pred["state_context_source"] = _ClosedLoopStateContextTracker.source
                pseudo_f.write(json.dumps(pseudo, ensure_ascii=False, sort_keys=True) + "\n")
                pred_f.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
                _observe_prediction_metrics(
                    row=row,
                    tokens=tokens,
                    stats=stats,
                    model_name=model_name,
                    baseline_names=baseline_names,
                    metrics_by_model=metrics_by_model,
                    group_metrics_by_model=group_metrics_by_model,
                    cluster_metrics_by_model=cluster_metrics_by_model,
                )
                sequence_fingerprint.update(json.dumps({"id": row["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                sequence_fingerprint.update(b"\n")
                target_count += 1
    metrics_path = output_dir / "metrics.json"
    metrics_payload = metrics_by_model[model_name].payload()
    write_json(metrics_path, metrics_payload)
    label_quality_report = {
        "schema": "idm_label_quality_report.v1",
        "model": model_name,
        "target_records": target_count,
        "model_metrics": metrics_payload,
        "baseline_metrics": {name: metrics_by_model[name].payload() for name in baseline_names},
        "groups_by_model": {
            name: _metric_payloads(group_metrics)
            for name, group_metrics in group_metrics_by_model.items()
        },
        "cluster_count": len(cluster_metrics_by_model[model_name]),
    }
    label_quality_report_path = output_dir / "label_quality_report.json"
    write_json(label_quality_report_path, label_quality_report)
    statistical_comparison = None
    statistical_comparison_path = None
    if config.get("endpoints"):
        statistical_comparison = _streaming_statistical_comparison(
            cluster_metrics_by_model,
            load_config(config["endpoints"]),
        )
        statistical_comparison_path = output_dir / "statistical_comparison.json"
        write_json(statistical_comparison_path, statistical_comparison)
    return {
        "pseudo_label_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
        "label_quality_report_path": str(label_quality_report_path),
        "label_quality_report": label_quality_report,
        "statistical_comparison_path": str(statistical_comparison_path) if statistical_comparison_path else None,
        "statistical_comparison": statistical_comparison,
        "target_records": target_count,
        "prediction_fingerprint": sequence_fingerprint.hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "target_source_ids": sorted(target_source_ids),
        "target_resolution_tiers": sorted(target_resolution_tiers),
        "target_eval_split_tags": sorted(target_eval_split_tags),
        "prediction_resume": {
            "enabled": resume_predictions,
            "existing_rows": resume_existing_rows,
            "write_mode": write_mode,
            "pseudolabel_validation": validate_pseudolabels,
            "closed_loop_state_context": {
                "enabled": closed_loop_state_context,
                "source": _ClosedLoopStateContextTracker.source if closed_loop_state_context else None,
                "seed_from_train": bool(config.get("closed_loop_state_context_seed_from_train", True))
                if closed_loop_state_context
                else False,
                "seed_rows": closed_loop_seed_rows,
            },
        },
        "metrics_state": _metric_state_map(metrics_by_model),
        "group_metrics_state": _nested_metric_state_map(group_metrics_by_model),
        "cluster_metrics_state": _nested_metric_state_map(cluster_metrics_by_model),
    }


def predict_streaming_idm_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    """Run a trained streaming IDM checkpoint over an arbitrary record JSONL.

    G003 trains/evaluates the IDM on the predeclared D2E target split.  G004
    also needs IDM pseudo-labels for the D2E train-core records so the FDM can
    train on train-core pseudo-actions while evaluating on heldout target
    records.  This prediction-only path preserves the original checkpoint,
    writes a separate pseudo-label artifact, and avoids retraining or
    overwriting the G003 target-eval pseudo-labels.
    """

    torch = require_torch()
    checkpoint_path = Path(config["checkpoint_path"])
    records_path = Path(config["records_path"])
    record_paths = _record_paths_from_config(
        config,
        primary_key="records_path",
        paths_key="record_paths",
        glob_key="records_glob",
    )
    output_dir = ensure_dir(config.get("output_dir", checkpoint_path.parent / "prediction"))
    force_cpu = bool(config.get("force_cpu", False))
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    prediction_workers = int(config.get("prediction_workers", 1))
    parallel_prediction = prediction_workers > 1 and len(record_paths) > 1
    checkpoint_device = "cpu" if parallel_prediction else device
    try:
        checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device)
    checkpoint_config = dict(checkpoint.get("config", {}))
    action_history_len = int(checkpoint.get("stats", {}).get("action_history_len", checkpoint_config.get("action_history_len", 0)) or 0)
    residual_mouse = _residual_mouse_from_mode(str(checkpoint_config.get("mouse_target_mode", config.get("mouse_target_mode", "mean"))))
    if parallel_prediction and action_history_len > 0 and not _action_history_parallel_by_path({**checkpoint_config, **config}):
        raise ValueError(
            "parallel prediction with action_history_len>0 requires action_history_parallel_by_path=true "
            "and target record paths that preserve complete recording order"
        )
    if parallel_prediction and residual_mouse and not _action_history_parallel_by_path({**checkpoint_config, **config}):
        raise ValueError(
            "parallel prediction with residual_last_seen mouse state requires action_history_parallel_by_path=true "
            "and target record paths that preserve complete recording order"
        )
    prediction_config = dict(checkpoint_config)
    for key, value in config.get("prediction_overrides", {}).items():
        prediction_config[key] = value
    for key in (
        "model_name",
        "endpoints",
        "baseline_names",
        "eval_batch_size",
        "batch_size",
        "max_target_examples",
        "category_threshold",
        "category_thresholds",
        "category_threshold_mode",
        "keyboard_softmax_threshold",
        "button_softmax_threshold",
        "mouse_axis_decode_mode",
        "mouse_axis_temperature",
        "mouse_output_gain",
        "mouse_output_gain_x",
        "mouse_output_gain_y",
        "mouse_output_gain_mode",
        "mouse_target_mode",
        "mouse_emit_mode",
        "mouse_max_tokens_per_axis",
        "resume_predictions",
        "closed_loop_state_context",
        "state_context_source",
        "closed_loop_state_context_seed_from_train",
        "closed_loop_state_context_seed_max_examples",
        "resume_closed_loop_state_context",
        "force_cpu",
        "train_records",
        "train_record_paths",
        "train_records_glob",
        "action_history_seed_state_mode",
        "action_history_seed_state_workers",
        "action_history_seed_state_parallel_by_path",
    ):
        if key in config:
            prediction_config[key] = config[key]
    if parallel_prediction and _closed_loop_state_context_enabled(prediction_config, dict(checkpoint.get("stats", {}))):
        raise ValueError(
            "closed_loop_state_context target inference is sequential and currently requires prediction_workers=1"
        )
    if parallel_prediction:
        if action_history_len > 0:
            train_paths = _record_paths_from_config(
                prediction_config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            seed_state, seed_source = _action_history_seed_state_for_parallel_prediction(
                {**prediction_config, **config},
                train_paths=train_paths,
                history_len=action_history_len,
            )
            prediction_config["_action_history_seed_state"] = seed_state
            prediction_config["action_history_seed_state_summary"] = {
                "source": seed_source,
                "recordings": int(seed_state.get("recordings", 0)),
                "history_len": int(seed_state.get("history_len", action_history_len)),
            }
        prediction = _predict_streaming_idm_checkpoint_parallel(
            config,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            target_record_paths=record_paths,
            prediction_config_base=prediction_config,
        )
    else:
        stats = dict(checkpoint["stats"])
        category_vocab = [str(token) for token in checkpoint.get("category_vocab", [])]
        keyboard_head_mode = str(checkpoint.get("keyboard_head_mode", prediction_config.get("keyboard_head_mode", "multilabel")))
        keyboard_classes = [tuple(str(token) for token in row) for row in checkpoint.get("keyboard_classes", prediction_config.get("keyboard_classes", []))]
        button_head_mode = str(checkpoint.get("button_head_mode", prediction_config.get("button_head_mode", "multilabel")))
        button_classes = [tuple(str(token) for token in row) for row in checkpoint.get("button_classes", prediction_config.get("button_classes", []))]
        mouse_head_mode = str(checkpoint.get("mouse_head_mode", prediction_config.get("mouse_head_mode", "axis_softmax")))
        mouse_axis_classes = [str(value) for value in checkpoint.get("mouse_axis_classes", prediction_config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
        prediction_config["keyboard_head_mode"] = keyboard_head_mode
        prediction_config["keyboard_classes"] = [list(tokens) for tokens in keyboard_classes]
        prediction_config["button_head_mode"] = button_head_mode
        prediction_config["button_classes"] = [list(tokens) for tokens in button_classes]
        prediction_config["mouse_head_mode"] = mouse_head_mode
        model = _build_model(
            torch,
            input_dim=int(stats["input_dim"]),
            output_dim=_streaming_output_dim(
                category_vocab=category_vocab,
                keyboard_head_mode=keyboard_head_mode,
                keyboard_classes=keyboard_classes,
                button_head_mode=button_head_mode,
                button_classes=button_classes,
                mouse_head_mode=mouse_head_mode,
                mouse_axis_classes=mouse_axis_classes,
            ),
            hidden_dim=int(checkpoint_config.get("hidden_dim", prediction_config.get("hidden_dim", 512))),
            depth=int(checkpoint_config.get("depth", prediction_config.get("depth", 3))),
            dropout=float(checkpoint_config.get("dropout", prediction_config.get("dropout", 0.05))),
            config=prediction_config,
            feature_mode=str(stats["feature_mode"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction = _predict_stream(
            torch,
            model,
            target_records=record_paths if len(record_paths) > 1 else records_path,
            stats=stats,
            config=prediction_config,
            device=device,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
    summary_prediction_config = dict(prediction_config)
    summary_prediction_config.pop("_action_history_seed_state", None)
    summary = {
        "schema": "streaming_idm_predict_summary.v1",
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_artifact": _file_artifact_metadata(checkpoint_path),
        "source_checkpoint_metadata": _file_artifact_metadata(config.get("checkpoint_metadata_path")),
        "records_path": str(records_path),
        "record_paths": [str(path) for path in record_paths],
        "records_glob": config.get("records_glob"),
        "output_dir": str(output_dir),
        "prediction_config": summary_prediction_config,
        "records": int(prediction["target_records"]),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "predictions_path": prediction["predictions_path"],
        "metrics_path": prediction["metrics_path"],
        "label_quality_report_path": prediction["label_quality_report_path"],
        "statistical_comparison_path": prediction["statistical_comparison_path"],
        "target_source_ids": prediction["target_source_ids"],
        "target_resolution_tiers": prediction["target_resolution_tiers"],
        "target_eval_split_tags": prediction["target_eval_split_tags"],
        "prediction_fingerprint": prediction["prediction_fingerprint"],
        "prediction_resume": prediction["prediction_resume"],
        "claim_boundary": "Prediction-only IDM pseudo-label artifact; it does not retrain or modify the source G003 checkpoint.",
    }
    if config.get("summary_out"):
        write_json(config["summary_out"], summary)
    else:
        write_json(output_dir / "summary.json", summary)
    return summary


def _chunk_sequence(items: Sequence[Path], chunks: int) -> list[list[Path]]:
    chunks = max(1, min(int(chunks), len(items)))
    base = len(items) // chunks
    extra = len(items) % chunks
    out: list[list[Path]] = []
    start = 0
    for idx in range(chunks):
        size = base + (1 if idx < extra else 0)
        out.append(list(items[start : start + size]))
        start += size
    return [chunk for chunk in out if chunk]


def _predict_stream_for_parallel_part(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point returning raw metric states for aggregation.

    This intentionally calls the lower-level prediction path instead of the
    public summary wrapper so the parent can aggregate exact counters across
    workers and still emit the normal monolithic G003 recovery contract.
    """

    if payload.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_visible_devices"])
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    torch = require_torch()
    checkpoint_path = Path(payload["checkpoint_path"])
    force_cpu = bool(payload.get("force_cpu", False))
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = dict(checkpoint["config"])
    prediction_config = dict(payload["prediction_config"])
    stats = dict(checkpoint["stats"])
    category_vocab = [str(token) for token in checkpoint.get("category_vocab", [])]
    keyboard_head_mode = str(checkpoint.get("keyboard_head_mode", prediction_config.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in checkpoint.get("keyboard_classes", prediction_config.get("keyboard_classes", []))]
    button_head_mode = str(checkpoint.get("button_head_mode", prediction_config.get("button_head_mode", "multilabel")))
    button_classes = [tuple(str(token) for token in row) for row in checkpoint.get("button_classes", prediction_config.get("button_classes", []))]
    mouse_head_mode = str(checkpoint.get("mouse_head_mode", prediction_config.get("mouse_head_mode", "axis_softmax")))
    mouse_axis_classes = [str(value) for value in checkpoint.get("mouse_axis_classes", prediction_config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
    prediction_config["keyboard_head_mode"] = keyboard_head_mode
    prediction_config["keyboard_classes"] = [list(tokens) for tokens in keyboard_classes]
    prediction_config["button_head_mode"] = button_head_mode
    prediction_config["button_classes"] = [list(tokens) for tokens in button_classes]
    prediction_config["mouse_head_mode"] = mouse_head_mode
    model = _build_model(
        torch,
        input_dim=int(stats["input_dim"]),
        output_dim=_streaming_output_dim(
            category_vocab=category_vocab,
            keyboard_head_mode=keyboard_head_mode,
            keyboard_classes=keyboard_classes,
            button_head_mode=button_head_mode,
            button_classes=button_classes,
            mouse_head_mode=mouse_head_mode,
            mouse_axis_classes=mouse_axis_classes,
        ),
        hidden_dim=int(checkpoint_config.get("hidden_dim", prediction_config.get("hidden_dim", 512))),
        depth=int(checkpoint_config.get("depth", prediction_config.get("depth", 3))),
        dropout=float(checkpoint_config.get("dropout", prediction_config.get("dropout", 0.05))),
        config=prediction_config,
        feature_mode=str(stats["feature_mode"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    record_paths = [Path(path) for path in payload["record_paths"]]
    output_dir = ensure_dir(payload["output_dir"])
    prediction = _predict_stream(
        torch,
        model,
        target_records=record_paths,
        stats=stats,
        config=prediction_config,
        device=device,
        category_vocab=category_vocab,
        mouse_axis_classes=mouse_axis_classes,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    return {
        "part_index": int(payload["part_index"]),
        "record_paths": [str(path) for path in record_paths],
        "output_dir": str(output_dir),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "predictions_path": prediction["predictions_path"],
        "records": int(prediction["target_records"]),
        "target_source_ids": prediction["target_source_ids"],
        "target_resolution_tiers": prediction["target_resolution_tiers"],
        "target_eval_split_tags": prediction["target_eval_split_tags"],
        "metrics_state": prediction["metrics_state"],
        "group_metrics_state": prediction["group_metrics_state"],
        "cluster_metrics_state": prediction["cluster_metrics_state"],
        "prediction_resume": prediction.get("prediction_resume", {}),
    }


def _concatenate_prediction_parts(parts: list[dict[str, Any]], *, output_dir: Path) -> dict[str, Any]:
    pseudo_path = output_dir / "pseudolabels.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    sequence_fingerprint = hashlib.sha256()
    rows = 0
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    with pseudo_path.open("w") as pseudo_out, predictions_path.open("w") as pred_out:
        for part in sorted(parts, key=lambda item: int(item["part_index"])):
            with Path(part["pseudo_label_path"]).open() as pseudo_in, Path(part["predictions_path"]).open() as pred_in:
                for pseudo_line, pred_line in zip(pseudo_in, pred_in):
                    if not pseudo_line.strip() and not pred_line.strip():
                        continue
                    if not pseudo_line.strip() or not pred_line.strip():
                        raise ValueError(f"mismatched blank prediction lines in part {part['part_index']}")
                    pseudo = json.loads(pseudo_line)
                    pred = json.loads(pred_line)
                    if str(pseudo.get("sequence_id")) != str(pred.get("sequence_id")):
                        raise ValueError(f"sequence_id mismatch while merging part {part['part_index']}: {pseudo.get('sequence_id')} != {pred.get('sequence_id')}")
                    tokens = [str(token) for token in pred.get("predicted_tokens", [])]
                    if [str(token) for token in pseudo.get("predicted_tokens", [])] != tokens:
                        raise ValueError(f"token mismatch while merging part {part['part_index']}: {pred.get('sequence_id')}")
                    pseudo_out.write(pseudo_line if pseudo_line.endswith("\n") else pseudo_line + "\n")
                    pred_out.write(pred_line if pred_line.endswith("\n") else pred_line + "\n")
                    sequence_fingerprint.update(json.dumps({"id": pred["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                    sequence_fingerprint.update(b"\n")
                    rows += 1
                remaining_pseudo = [line for line in pseudo_in if line.strip()]
                remaining_pred = [line for line in pred_in if line.strip()]
                if remaining_pseudo or remaining_pred:
                    raise ValueError(f"mismatched prediction row counts while merging part {part['part_index']}")
    return {
        "pseudo_label_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "target_records": rows,
        "prediction_fingerprint": sequence_fingerprint.hexdigest(),
    }


def _predict_streaming_idm_checkpoint_parallel(
    config: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    target_record_paths: Sequence[Path] | None = None,
    prediction_config_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_config = dict(checkpoint.get("config", {}))
    if target_record_paths is None:
        target_record_paths = _record_paths_from_config(
            checkpoint_config,
            primary_key="target_records",
            paths_key="target_record_paths",
            glob_key="target_records_glob",
        )
    else:
        target_record_paths = list(target_record_paths)
    if not target_record_paths:
        raise ValueError("parallel checkpoint recovery requires at least one target record path")
    workers = int(config.get("prediction_workers", 1))
    if workers <= 1 or len(target_record_paths) == 1:
        raise ValueError("parallel prediction requested without multiple workers/record paths")
    chunks = _chunk_sequence(target_record_paths, workers)
    prediction_config = dict(prediction_config_base or checkpoint_config)
    for key in (
        "model_name",
        "endpoints",
        "baseline_names",
        "eval_batch_size",
        "batch_size",
        "max_target_examples",
        "category_threshold",
        "category_thresholds",
        "category_threshold_mode",
        "keyboard_softmax_threshold",
        "button_softmax_threshold",
        "mouse_axis_decode_mode",
        "mouse_axis_temperature",
        "mouse_output_gain",
        "mouse_output_gain_x",
        "mouse_output_gain_y",
        "mouse_output_gain_mode",
        "mouse_target_mode",
        "mouse_emit_mode",
        "mouse_max_tokens_per_axis",
        "closed_loop_state_context",
        "state_context_source",
        "closed_loop_state_context_seed_from_train",
        "closed_loop_state_context_seed_max_examples",
        "resume_closed_loop_state_context",
        "force_cpu",
        "validate_pseudolabels",
    ):
        if key in config:
            prediction_config[key] = config[key]
    if _closed_loop_state_context_enabled(prediction_config, dict(checkpoint.get("stats", {}))):
        raise ValueError(
            "closed_loop_state_context target inference is sequential and currently requires prediction_workers=1"
        )
    prediction_config["resume_predictions"] = bool(
        config.get("resume_prediction_parts", config.get("resume_predictions", True))
    )
    parts_root = ensure_dir(config.get("prediction_parts_dir", output_dir / "prediction_recovery_parts"))
    cuda_devices = config.get("prediction_cuda_devices")
    if cuda_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            cuda_devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        else:
            cuda_devices = list(range(len(chunks)))
    cuda_devices = list(cuda_devices)
    if not cuda_devices and not bool(config.get("force_cpu", False)):
        raise ValueError("prediction_cuda_devices resolved to an empty list")
    payloads = []
    for part_index, record_paths in enumerate(chunks):
        payloads.append(
            {
                "part_index": part_index,
                "record_paths": [str(path) for path in record_paths],
                "output_dir": str(parts_root / f"part_{part_index:03d}"),
                "checkpoint_path": str(checkpoint_path),
                "prediction_config": prediction_config,
                "force_cpu": bool(config.get("force_cpu", False)),
                "cuda_visible_devices": None if bool(config.get("force_cpu", False)) else cuda_devices[part_index % len(cuda_devices)],
            }
        )
    parts: list[dict[str, Any]] = []
    # Use spawn rather than fork because the parent may have already imported
    # torch/CUDA during training or tests. Forking after torch init can hang
    # before workers return their prediction summaries.
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_predict_stream_for_parallel_part, payload) for payload in payloads]
        for future in as_completed(futures):
            parts.append(future.result())
    parts = sorted(parts, key=lambda item: int(item["part_index"]))
    merged_paths = _concatenate_prediction_parts(parts, output_dir=output_dir)
    metrics_by_model = _merge_named_metric_states(part["metrics_state"] for part in parts)
    group_metrics_by_model = _merge_nested_metric_states(part["group_metrics_state"] for part in parts)
    cluster_metrics_by_model = _merge_nested_metric_states(part["cluster_metrics_state"] for part in parts)
    model_name = str(prediction_config.get("model_name", "streaming_compact_idm"))
    baseline_names = [str(name) for name in prediction_config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])]
    metrics_path = output_dir / "metrics.json"
    metrics_payload = metrics_by_model[model_name].payload()
    write_json(metrics_path, metrics_payload)
    label_quality_report = {
        "schema": "idm_label_quality_report.v1",
        "model": model_name,
        "target_records": int(merged_paths["target_records"]),
        "model_metrics": metrics_payload,
        "baseline_metrics": {name: metrics_by_model[name].payload() for name in baseline_names if name in metrics_by_model},
        "groups_by_model": {
            name: _metric_payloads(group_metrics)
            for name, group_metrics in group_metrics_by_model.items()
        },
        "cluster_count": len(cluster_metrics_by_model.get(model_name, {})),
    }
    label_quality_report_path = output_dir / "label_quality_report.json"
    write_json(label_quality_report_path, label_quality_report)
    statistical_comparison = None
    statistical_comparison_path = None
    if prediction_config.get("endpoints"):
        statistical_comparison = _streaming_statistical_comparison(
            cluster_metrics_by_model,
            load_config(prediction_config["endpoints"]),
        )
        statistical_comparison_path = output_dir / "statistical_comparison.json"
        write_json(statistical_comparison_path, statistical_comparison)
    return {
        **merged_paths,
        "records": int(merged_paths["target_records"]),
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
        "label_quality_report_path": str(label_quality_report_path),
        "label_quality_report": label_quality_report,
        "statistical_comparison_path": str(statistical_comparison_path) if statistical_comparison_path else None,
        "statistical_comparison": statistical_comparison,
        "target_source_ids": sorted({source for part in parts for source in part.get("target_source_ids", [])}),
        "target_resolution_tiers": sorted({tier for part in parts for tier in part.get("target_resolution_tiers", [])}),
        "target_eval_split_tags": sorted({tag for part in parts for tag in part.get("target_eval_split_tags", [])}),
        "prediction_resume": {
            "enabled": bool(prediction_config.get("resume_predictions", False)),
            "existing_rows": sum(
                int((part.get("prediction_resume") or {}).get("existing_rows") or 0)
                for part in parts
            ),
            "write_mode": "parallel_parts",
            "workers": len(parts),
            "pseudolabel_validation": bool(prediction_config.get("validate_pseudolabels", True)),
            "action_history_seed_state_summary": dict(prediction_config.get("action_history_seed_state_summary", {})),
            "parts": [
                {
                    "part_index": part["part_index"],
                    "records": part["records"],
                    "record_paths": part["record_paths"],
                    "pseudo_label_path": part["pseudo_label_path"],
                    "predictions_path": part["predictions_path"],
                    "resume_existing_rows": int((part.get("prediction_resume") or {}).get("existing_rows") or 0),
                }
                for part in parts
            ],
        },
    }


def recover_streaming_idm_outputs_from_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild streaming IDM prediction/metadata artifacts from an existing checkpoint.

    Full-corpus G003/G004 jobs save the model checkpoint before running the long
    target prediction pass.  If that prediction or metadata write is
    interrupted, this recovery path avoids retraining: it reruns/resumes
    checkpoint inference over the target records, then reconstructs the same
    checkpoint metadata and train-summary contract produced by
    ``train_streaming_idm``.
    """

    torch = require_torch()
    checkpoint_path = Path(config["checkpoint_path"])
    force_cpu = bool(config.get("force_cpu", False))
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = dict(checkpoint.get("config", {}))
    if not checkpoint_config:
        raise ValueError(f"checkpoint does not contain a training config: {checkpoint_path}")
    output_dir = ensure_dir(config.get("output_dir", checkpoint_config.get("output_dir", checkpoint_path.parent)))
    target_record_paths = _record_paths_from_config(
        checkpoint_config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    prediction_config: dict[str, Any] = dict(checkpoint_config)
    prediction_config.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata_path": str(output_dir / "checkpoint_metadata.json"),
            "records_path": str(checkpoint_config["target_records"]),
            "output_dir": str(output_dir),
            "resume_predictions": bool(config.get("resume_predictions", True)),
            "force_cpu": force_cpu,
            "summary_out": str(config.get("prediction_summary_out", output_dir / "prediction_recovery_summary.json")),
        }
    )
    if len(target_record_paths) > 1:
        prediction_config["record_paths"] = [str(path) for path in target_record_paths]
        if checkpoint_config.get("target_records_glob"):
            prediction_config["records_glob"] = checkpoint_config["target_records_glob"]
    for key in (
        "model_name",
        "endpoints",
        "baseline_names",
        "eval_batch_size",
        "batch_size",
        "max_target_examples",
        "category_threshold",
        "category_thresholds",
        "category_threshold_mode",
        "keyboard_softmax_threshold",
        "button_softmax_threshold",
        "mouse_axis_decode_mode",
        "mouse_axis_temperature",
        "mouse_output_gain",
        "mouse_output_gain_x",
        "mouse_output_gain_y",
        "mouse_output_gain_mode",
        "mouse_target_mode",
        "mouse_emit_mode",
        "mouse_max_tokens_per_axis",
        "validate_pseudolabels",
        "prediction_workers",
        "prediction_cuda_devices",
        "resume_prediction_parts",
        "closed_loop_state_context",
        "state_context_source",
        "closed_loop_state_context_seed_from_train",
        "closed_loop_state_context_seed_max_examples",
        "resume_closed_loop_state_context",
        "action_history_parallel_by_path",
        "action_history_parallel_prediction_by_path",
        "action_history_seed_state_mode",
        "parallel_action_history_seed_state_mode",
    ):
        if key in config:
            prediction_config[key] = config[key]
        elif key in checkpoint_config:
            prediction_config[key] = checkpoint_config[key]
    prediction_workers = int(config.get("prediction_workers", 1))
    if prediction_workers > 1 and _closed_loop_state_context_enabled(prediction_config, dict(checkpoint.get("stats", {}))):
        raise ValueError(
            "closed_loop_state_context target inference is sequential and currently requires prediction_workers=1"
        )
    if prediction_workers > 1:
        action_history_len = int(
            checkpoint.get("stats", {}).get("action_history_len", checkpoint_config.get("action_history_len", 0)) or 0
        )
        residual_mouse = _residual_mouse_from_mode(str(checkpoint_config.get("mouse_target_mode", config.get("mouse_target_mode", "mean"))))
        if action_history_len > 0 and not _action_history_parallel_by_path({**checkpoint_config, **config}):
            raise ValueError(
                "parallel prediction with action_history_len>0 requires action_history_parallel_by_path=true "
                "and target record paths that preserve complete recording order"
            )
        if residual_mouse and not _action_history_parallel_by_path({**checkpoint_config, **config}):
            raise ValueError(
                "parallel prediction with residual_last_seen mouse state requires action_history_parallel_by_path=true "
                "and target record paths that preserve complete recording order"
            )
        if action_history_len > 0:
            train_paths = _record_paths_from_config(
                checkpoint_config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            seed_state, seed_source = _action_history_seed_state_for_parallel_prediction(
                {**checkpoint_config, **config},
                train_paths=train_paths,
                history_len=action_history_len,
                stats=dict(checkpoint.get("stats", {})),
            )
            prediction_config["_action_history_seed_state"] = seed_state
            prediction_config["action_history_seed_state_summary"] = {
                "source": seed_source,
                "recordings": int(seed_state.get("recordings", 0)),
                "history_len": int(seed_state.get("history_len", action_history_len)),
            }
        if residual_mouse:
            train_paths = _record_paths_from_config(
                checkpoint_config,
                primary_key="train_records",
                paths_key="train_record_paths",
                glob_key="train_records_glob",
            )
            mouse_seed_state = _streaming_mouse_seed_state_for_parallel_prediction(
                {**checkpoint_config, **config},
                train_paths=train_paths,
                stats=dict(checkpoint.get("stats", {})),
            )
            prediction_config["_streaming_mouse_seed_state"] = mouse_seed_state
            prediction_config["streaming_mouse_seed_state_summary"] = {
                "source": mouse_seed_state.get("source"),
                "recordings": int(mouse_seed_state.get("recordings", 0)),
                "games": int(mouse_seed_state.get("games", 0)),
                "workers": int(mouse_seed_state.get("workers", 1)),
            }
        prediction = _predict_streaming_idm_checkpoint_parallel(
            config,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            target_record_paths=target_record_paths,
            prediction_config_base=prediction_config,
        )
    else:
        prediction = predict_streaming_idm_checkpoint(prediction_config)
    stats = dict(checkpoint["stats"])
    category_vocab = [str(token) for token in checkpoint.get("category_vocab", [])]
    keyboard_head_mode = str(checkpoint.get("keyboard_head_mode", checkpoint_config.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in checkpoint.get("keyboard_classes", checkpoint_config.get("keyboard_classes", []))]
    button_head_mode = str(checkpoint.get("button_head_mode", checkpoint_config.get("button_head_mode", "multilabel")))
    button_classes = [tuple(str(token) for token in row) for row in checkpoint.get("button_classes", checkpoint_config.get("button_classes", []))]
    mouse_head_mode = str(checkpoint.get("mouse_head_mode", checkpoint_config.get("mouse_head_mode", "axis_softmax")))
    mouse_axis_classes = [str(value) for value in checkpoint.get("mouse_axis_classes", checkpoint_config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
    history = list(checkpoint.get("history", []))
    config_fingerprint = stable_hash_json(checkpoint_config)
    resolved_config_path = output_dir / "resolved_config.json"
    write_json(
        resolved_config_path,
        {
            "schema": "streaming_idm_resolved_config.v1",
            "model": str(checkpoint_config.get("model_name", "streaming_compact_idm")),
            "config": checkpoint_config,
            "config_fingerprint": config_fingerprint,
            "recovered_from_checkpoint": str(checkpoint_path),
        },
    )
    convergence_report_path = output_dir / "convergence_report.json"
    convergence_report = read_json(convergence_report_path) if convergence_report_path.exists() else _convergence_report(history, checkpoint_config, output_dir=output_dir)
    if not convergence_report_path.exists():
        write_json(convergence_report_path, convergence_report)
    data_universe_path = checkpoint_config.get("data_universe")
    split_contract_path = checkpoint_config.get("split_contract")
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(checkpoint_config.get("model_name", "streaming_compact_idm")),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "config_fingerprint": config_fingerprint,
        "config_path": str(checkpoint_config.get("config_path", config.get("config_path", ""))),
        "resolved_config_path": str(resolved_config_path),
        "train_records": int(stats["num_examples"]),
        "target_records": int(prediction["records"]),
        "train_records_path": str(checkpoint_config["train_records"]),
        "target_records_path": str(checkpoint_config["target_records"]),
        "data_universe": _file_artifact_metadata(data_universe_path),
        "data_universe_fingerprint": _json_fingerprint(data_universe_path),
        "split_contract": _file_artifact_metadata(split_contract_path),
        "split_contract_fingerprint": _json_fingerprint(split_contract_path),
        "split_id": str(checkpoint_config.get("split_id") or _json_fingerprint(split_contract_path) or "d2e_full_split_contract"),
        "source_namespace": str(checkpoint_config.get("source_namespace", "d2e_full_corpus")),
        "source_ids": list(stats.get("source_ids", [])),
        "resolution_tiers": list(stats.get("resolution_tiers", [])),
        "target_source_ids": list(prediction.get("target_source_ids", [])),
        "target_resolution_tiers": list(prediction.get("target_resolution_tiers", [])),
        "split_names": list(stats.get("split_names", [])),
        "eval_split_tags": list(stats.get("eval_split_tags", [])),
        "target_eval_split_tags": list(prediction.get("target_eval_split_tags", [])),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "filtered_pseudo_label_path": prediction["pseudo_label_path"],
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": prediction["metrics_path"],
        "label_quality_report_path": prediction["label_quality_report_path"],
        "statistical_comparison_path": prediction["statistical_comparison_path"],
        "convergence_report_path": str(convergence_report_path),
        "convergence_plateau_met": bool(convergence_report.get("plateau_met", False)),
        "calibration": {
            "mode": (
                "global_threshold_streaming"
                if str(checkpoint_config.get("category_threshold_mode", "global")) == "global"
                else str(checkpoint_config.get("category_threshold_mode", "global_threshold_streaming"))
            ),
            "category_threshold": float(checkpoint_config.get("category_threshold", 0.35)),
            "category_thresholds": dict(checkpoint_config.get("category_thresholds", {})),
            "keyboard_softmax_threshold": float(checkpoint_config.get("keyboard_softmax_threshold", 0.5)),
            "button_softmax_threshold": float(checkpoint_config.get("button_softmax_threshold", 0.5)),
            "mouse_output_gain": float(checkpoint_config.get("mouse_output_gain", 1.0)),
            "mouse_output_gain_x": (
                float(checkpoint_config["mouse_output_gain_x"])
                if checkpoint_config.get("mouse_output_gain_x") is not None
                else None
            ),
            "mouse_output_gain_y": (
                float(checkpoint_config["mouse_output_gain_y"])
                if checkpoint_config.get("mouse_output_gain_y") is not None
                else None
            ),
            "mouse_output_gain_mode": str(checkpoint_config.get("mouse_output_gain_mode", "fixed")),
            "last_train_loss": history[-1]["loss"] if history else None,
            "prediction_fingerprint": prediction["prediction_fingerprint"],
        },
        "feature_mode": str(stats["feature_mode"]),
        "input_dim": int(stats["input_dim"]),
        "action_history_len": int(stats.get("action_history_len", checkpoint_config.get("action_history_len", 0)) or 0),
        "action_history_dim": int(stats.get("action_history_dim", 0) or 0),
        "action_history_feedback": "autoregressive_predicted" if int(stats.get("action_history_len", checkpoint_config.get("action_history_len", 0)) or 0) > 0 else "none",
        "categorical_vocab": category_vocab,
        "keyboard_head_mode": keyboard_head_mode,
        "keyboard_classes": [list(tokens) for tokens in keyboard_classes],
        "button_head_mode": button_head_mode,
        "button_classes": [list(tokens) for tokens in button_classes],
        "mouse_head_mode": mouse_head_mode,
        "mouse_target_mode": str(checkpoint_config.get("mouse_target_mode", "mean")),
        "mouse_emit_mode": str(checkpoint_config.get("mouse_emit_mode", "single")),
        "mouse_max_tokens_per_axis": int(checkpoint_config.get("mouse_max_tokens_per_axis", 8)),
        "mouse_axis_classes": mouse_axis_classes if mouse_head_mode == "axis_softmax" else [],
        "distributed": {
            "enabled": bool(checkpoint_config.get("distributed", {}).get("enabled", False)),
            "world_size": int(checkpoint_config.get("distributed", {}).get("world_size", 1)),
            "backend": checkpoint_config.get("distributed", {}).get("backend"),
            "rank0_device": device,
        },
        "recovery": {
            "schema": "streaming_idm_checkpoint_recovery.v1",
            "source_checkpoint_path": str(checkpoint_path),
            "prediction_summary_path": str(prediction_config["summary_out"]),
            "prediction_resume": prediction.get("prediction_resume", {}),
        },
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    metadata_path = output_dir / "checkpoint_metadata.json"
    write_json(metadata_path, metadata)
    metrics = read_json(prediction["metrics_path"])
    label_quality_report = read_json(prediction["label_quality_report_path"])
    statistical_comparison = read_json(prediction["statistical_comparison_path"]) if prediction.get("statistical_comparison_path") else None
    summary = {
        "schema": "streaming_idm_train_summary.v1",
        "metadata": metadata,
        "metrics": metrics,
        "label_quality_report": label_quality_report,
        "statistical_comparison": statistical_comparison,
        "convergence_report": convergence_report,
        "history_tail": history[-5:],
        "device": device,
        "stats_path": str(output_dir / "streaming_stats.json"),
        "predictions_path": prediction["predictions_path"],
        "prediction_resume": prediction.get("prediction_resume", {}),
        "recovered_from_checkpoint": str(checkpoint_path),
    }
    summary_path = Path(config.get("summary_out", checkpoint_config.get("summary_out", output_dir / "summary.json")))
    write_json(summary_path, summary)
    recovery_summary = {
        "schema": "streaming_idm_checkpoint_recovery_summary.v1",
        "status": "pass",
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "prediction_summary_path": str(prediction_config["summary_out"]),
        "target_records": int(prediction["records"]),
        "prediction_resume": prediction.get("prediction_resume", {}),
    }
    write_json(Path(prediction_config["summary_out"]), recovery_summary)
    return recovery_summary


def train_streaming_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    dist = _distributed_runtime(torch, config)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = str(dist["device"])
    train_records = Path(config["train_records"])
    target_records = Path(config["target_records"])
    train_record_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    target_record_paths = _record_paths_from_config(
        config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    feature_mode = str(config.get("feature_mode", "summary_compact_grid8_shift_surface_time"))
    out_dir = ensure_dir(config.get("output_dir", "outputs/idm_streaming_full"))
    stats_path = out_dir / "streaming_stats.json"
    if dist["enabled"] and not dist["is_rank0"]:
        _barrier(torch, dist)
        stats = read_json(stats_path)
    else:
        if stats_path.exists() and not bool(config.get("rescan_stats", False)):
            stats = read_json(stats_path)
        else:
            stats = scan_streaming_idm_stats(
                train_record_paths if len(train_record_paths) > 1 else train_record_paths[0],
                feature_mode=feature_mode,
                categorical_min_count=int(config.get("categorical_min_count", 1)),
                num_workers=int(config.get("precompute_num_workers", config.get("stats_num_workers", 1))),
                action_history_len=_action_history_len_from_config(config),
                action_history_parallel_by_path=_action_history_parallel_by_path(config),
                residual_mouse=_residual_mouse_from_mode(_mouse_target_mode(config)),
            )
            write_json(stats_path, stats)
        if dist["enabled"]:
            _barrier(torch, dist)
    category_vocab = _category_vocab_for_heads(stats, config)
    keyboard_head_mode = str(config.get("keyboard_head_mode", "multilabel"))
    if keyboard_head_mode not in {"multilabel", "softmax"}:
        raise ValueError(f"unsupported keyboard_head_mode: {keyboard_head_mode}")
    button_head_mode = str(config.get("button_head_mode", "multilabel"))
    if button_head_mode not in {"multilabel", "softmax"}:
        raise ValueError(f"unsupported button_head_mode: {button_head_mode}")
    keyboard_classes = _keyboard_classes_for_heads(stats, config)
    button_classes = _button_classes_for_heads(stats, config)
    if keyboard_classes:
        config["keyboard_classes"] = [list(tokens) for tokens in keyboard_classes]
    if button_classes:
        config["button_classes"] = [list(tokens) for tokens in button_classes]
    mouse_axis_classes = [str(value) for value in config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)]
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    if mouse_head_mode not in {"regression", "axis_softmax"}:
        raise ValueError(f"unsupported mouse_head_mode: {mouse_head_mode}")
    training_cache_manifests: list[dict[str, Any]] = []
    if config.get("training_cache_dir"):
        if dist["enabled"] and not dist["is_rank0"]:
            _barrier(torch, dist)
            training_cache_manifests = _load_training_cache_manifests(
                train_record_paths,
                stats=stats,
                config=config,
                category_vocab=category_vocab,
                mouse_axis_classes=mouse_axis_classes,
            )
        else:
            training_cache_manifests = _build_training_cache_manifests(
                train_record_paths,
                stats=stats,
                config=config,
                category_vocab=category_vocab,
                mouse_axis_classes=mouse_axis_classes,
            )
            if dist["enabled"]:
                _barrier(torch, dist)
    output_dim = _streaming_output_dim(
        category_vocab=category_vocab,
        keyboard_head_mode=keyboard_head_mode,
        keyboard_classes=keyboard_classes,
        button_head_mode=button_head_mode,
        button_classes=button_classes,
        mouse_head_mode=mouse_head_mode,
        mouse_axis_classes=mouse_axis_classes,
    )
    model = _build_model(
        torch,
        input_dim=int(stats["input_dim"]),
        output_dim=output_dim,
        hidden_dim=int(config.get("hidden_dim", 512)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.05)),
        config=config,
        feature_mode=feature_mode,
    ).to(device)
    resume_training_checkpoint_path = config.get("resume_training_checkpoint")
    resume_checkpoint_path = resume_training_checkpoint_path or config.get("initial_checkpoint_path")
    resume_training = bool(resume_training_checkpoint_path)
    resumed_history: list[dict[str, Any]] = []
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_training_checkpoint does not exist: {resume_path}")
        try:
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch releases.
            resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if bool(config.get("resume_training_preserve_history", True)):
            resumed_history = [dict(row) for row in resume_checkpoint.get("history", []) if isinstance(row, dict)]
        config["resumed_from_checkpoint"] = str(resume_path)
    train_model = model
    if dist["enabled"]:
        ddp_kwargs = {"device_ids": [int(dist["local_rank"])]} if str(device).startswith("cuda") else {}
        train_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    opt = torch.optim.AdamW(train_model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    cat_pos_weight = _soft_pos_weight(
        torch,
        {str(k): int(v) for k, v in stats.get("category_counts", {}).items()},
        category_vocab,
        int(stats["num_examples"]),
        cap=float(config.get("categorical_pos_weight_cap", 20.0)),
        device=device,
    )
    keyboard_class_weight = _exact_set_class_weights(
        torch,
        {str(k): int(v) for k, v in stats.get("keyboard_class_counts", {}).items()},
        keyboard_classes,
        cap=float(config.get("keyboard_softmax_class_weight_cap", 20.0)),
        empty_weight=float(config.get("keyboard_softmax_no_key_weight", 1.0)),
        positive_weight=float(config.get("keyboard_softmax_positive_weight", 1.0)),
        device=device,
    ) if keyboard_head_mode == "softmax" else None
    button_class_weight = _exact_set_class_weights(
        torch,
        {str(k): int(v) for k, v in stats.get("button_class_counts", {}).items()},
        button_classes,
        cap=float(config.get("button_softmax_class_weight_cap", 20.0)),
        empty_weight=float(config.get("button_softmax_no_button_weight", 1.0)),
        positive_weight=float(config.get("button_softmax_positive_weight", 1.0)),
        device=device,
    ) if button_head_mode == "softmax" else None
    history = resumed_history
    start_epoch = len(history)
    total_epochs = int(config.get("epochs", 3))
    epochs_to_run = _streaming_resume_epochs_to_run(
        total_epochs=total_epochs,
        completed_epochs=start_epoch,
        resume_training=resume_training,
    )
    config["resume_training_completed_epochs"] = start_epoch if resume_training else 0
    config["resume_training_epochs_to_run"] = epochs_to_run
    convergence_report = {
        "schema": "streaming_convergence_report.v1",
        "score_mode": str(config.get("convergence_score", "composite_primary")),
        "direction": "higher",
        "eval_interval_epochs": int(config.get("eval_interval_epochs", 0)),
        "patience": int(config.get("plateau_patience", 3)),
        "min_relative_improvement": float(config.get("plateau_min_relative_improvement", 0.01)),
        "plateau_met": False,
        "num_validation_checkpoints": 0,
        "recent_relative_improvements": [],
        "history": [],
        "report_path": str(out_dir / "convergence_report.json"),
    }
    epoch_checkpoint_path = out_dir / "checkpoint_epoch_latest.pt"
    epoch_checkpoint_metadata_path = out_dir / "checkpoint_epoch_latest_metadata.json"
    save_epoch_checkpoints = bool(config.get("save_epoch_checkpoints", False))

    def build_checkpoint_payload(stage: str) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "stats": stats,
            "category_vocab": category_vocab,
            "keyboard_head_mode": keyboard_head_mode,
            "keyboard_classes": [list(tokens) for tokens in keyboard_classes],
            "button_head_mode": button_head_mode,
            "button_classes": [list(tokens) for tokens in button_classes],
            "mouse_head_mode": mouse_head_mode,
            "mouse_target_mode": _mouse_target_mode(config),
            "mouse_emit_mode": str(config.get("mouse_emit_mode", "single")),
            "mouse_max_tokens_per_axis": int(config.get("mouse_max_tokens_per_axis", 8)),
            "mouse_axis_classes": mouse_axis_classes,
            "history": history,
            "calibration_stage": stage,
        }

    eval_interval_epochs = int(config.get("eval_interval_epochs", 0))
    for epoch in range(epochs_to_run):
        join_context = train_model.join() if dist["enabled"] else nullcontext()
        with join_context:
            epoch_stats = _train_one_epoch(
                torch,
                train_model,
                opt,
                train_records=train_records,
                stats=stats,
                config=config,
                device=device,
                category_vocab=category_vocab,
                cat_pos_weight=cat_pos_weight,
                keyboard_classes=keyboard_classes,
                keyboard_class_weight=keyboard_class_weight,
                button_classes=button_classes,
                button_class_weight=button_class_weight,
                mouse_axis_classes=mouse_axis_classes,
                rank=int(dist["rank"]),
                world_size=int(dist["world_size"]),
                train_record_paths=train_record_paths,
                training_cache_manifests=training_cache_manifests,
            )
        epoch_stats = _aggregate_epoch_stats(torch, epoch_stats, device=device, dist=dist)
        if dist["is_rank0"]:
            epoch_number = start_epoch + epoch + 1
            row = {"epoch": epoch_number, **epoch_stats}
            if eval_interval_epochs > 0 and ((epoch + 1) % eval_interval_epochs == 0 or (epoch + 1) == epochs_to_run):
                row["validation"] = _evaluate_stream_metrics(
                    torch,
                    model,
                    target_records=target_record_paths if len(target_record_paths) > 1 else target_record_paths[0],
                    stats=stats,
                    config=config,
                    device=device,
                    category_vocab=category_vocab,
                    mouse_axis_classes=mouse_axis_classes,
                )
            history.append(row)
            convergence_report = _convergence_report(history, config, output_dir=out_dir)
            write_json(out_dir / "train_history.json", {"schema": "streaming_idm_train_history.v1", "history": history})
            write_json(out_dir / "convergence_report.json", convergence_report)
            if save_epoch_checkpoints:
                tmp_epoch_checkpoint_path = epoch_checkpoint_path.with_suffix(epoch_checkpoint_path.suffix + ".tmp")
                torch.save(build_checkpoint_payload(f"epoch_{epoch_number}"), tmp_epoch_checkpoint_path)
                tmp_epoch_checkpoint_path.replace(epoch_checkpoint_path)
                write_json(
                    epoch_checkpoint_metadata_path,
                    {
                        "schema": "streaming_idm_epoch_checkpoint_metadata.v1",
                        "status": "pass",
                        "checkpoint_path": str(epoch_checkpoint_path),
                        "epoch": epoch_number,
                        "history_path": str(out_dir / "train_history.json"),
                        "updated_at_epoch": time.time(),
                    },
                )
        _barrier(torch, dist)
    if dist["enabled"] and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not dist["is_rank0"]:
        return {
            "schema": "streaming_idm_worker_summary.v1",
            "rank": int(dist["rank"]),
            "world_size": int(dist["world_size"]),
            "status": "worker_complete",
        }
    checkpoint_path = out_dir / "checkpoint.pt"
    pre_calibration_checkpoint_path = out_dir / "checkpoint_pre_calibration.pt"
    calibration_progress_path = out_dir / "calibration_progress.json"

    def write_calibration_progress(stage: str, **extra: Any) -> None:
        write_json(
            calibration_progress_path,
            {
                "schema": "streaming_idm_calibration_progress.v1",
                "status": "running" if stage != "final_checkpoint_saved" else "pass",
                "stage": stage,
                "checkpoint_path": str(checkpoint_path),
                "pre_calibration_checkpoint_path": str(pre_calibration_checkpoint_path),
                "updated_at_epoch": time.time(),
                **extra,
            },
        )

    if bool(config.get("save_pre_calibration_checkpoint", True)):
        write_calibration_progress("saving_pre_calibration_checkpoint")
        torch.save(build_checkpoint_payload("pre_calibration"), pre_calibration_checkpoint_path)
        write_calibration_progress("pre_calibration_checkpoint_saved")
    use_cache_calibration = bool(training_cache_manifests) and bool(config.get("calibration_use_training_cache", True))
    write_calibration_progress(
        "category_calibration_started",
        use_cache_calibration=use_cache_calibration,
        category_calibration_max_examples=config.get("category_calibration_max_examples"),
    )
    if use_cache_calibration:
        category_thresholds, calibration_info = _calibrate_streaming_category_thresholds_from_cache(
            torch,
            model,
            training_cache_manifests=training_cache_manifests,
            config=config,
            device=device,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
    else:
        category_thresholds, calibration_info = _calibrate_streaming_category_thresholds(
            torch,
            model,
            train_records=train_record_paths if len(train_record_paths) > 1 else train_record_paths[0],
            stats=stats,
            config=config,
            device=device,
            category_vocab=category_vocab,
        )
    config["category_thresholds"] = category_thresholds
    write_calibration_progress(
        "category_calibration_finished",
        use_cache_calibration=use_cache_calibration,
        calibration_status=calibration_info.get("status"),
        observed_examples=calibration_info.get("observed_examples"),
    )
    softmax_calibration_info: dict[str, Any] = {"status": "skipped_no_cache"}
    if use_cache_calibration:
        write_calibration_progress(
            "softmax_threshold_calibration_started",
            softmax_calibration_max_examples=config.get("softmax_calibration_max_examples", config.get("category_calibration_max_examples")),
        )
        softmax_calibration_info = _calibrate_streaming_softmax_thresholds_from_cache(
            torch,
            model,
            training_cache_manifests=training_cache_manifests,
            config=config,
            device=device,
            category_vocab=category_vocab,
            keyboard_classes=keyboard_classes,
            button_classes=button_classes,
        )
        write_calibration_progress(
            "softmax_threshold_calibration_finished",
            softmax_calibration_status=softmax_calibration_info.get("status"),
            keyboard_softmax_threshold=config.get("keyboard_softmax_threshold"),
            button_softmax_threshold=config.get("button_softmax_threshold"),
        )
    write_calibration_progress(
        "mouse_gain_calibration_started",
        use_cache_calibration=use_cache_calibration,
        mouse_gain_calibration_max_examples=config.get("mouse_gain_calibration_max_examples"),
    )
    if use_cache_calibration:
        mouse_output_gain, mouse_output_gain_info = _calibrate_streaming_mouse_output_gain_from_cache(
            torch,
            model,
            training_cache_manifests=training_cache_manifests,
            config=config,
            device=device,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
    else:
        mouse_output_gain, mouse_output_gain_info = _calibrate_streaming_mouse_output_gain(
            torch,
            model,
            train_records=train_record_paths if len(train_record_paths) > 1 else train_record_paths[0],
            stats=stats,
            config=config,
            device=device,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
        )
    config["mouse_output_gain"] = mouse_output_gain
    if mouse_output_gain_info.get("gain_x") is not None:
        config["mouse_output_gain_x"] = float(mouse_output_gain_info["gain_x"])
    if mouse_output_gain_info.get("gain_y") is not None:
        config["mouse_output_gain_y"] = float(mouse_output_gain_info["gain_y"])
    write_calibration_progress(
        "mouse_gain_calibration_finished",
        use_cache_calibration=use_cache_calibration,
        mouse_output_gain=mouse_output_gain,
        mouse_output_gain_x=config.get("mouse_output_gain_x"),
        mouse_output_gain_y=config.get("mouse_output_gain_y"),
        mouse_gain_status=mouse_output_gain_info.get("status"),
        value_count=mouse_output_gain_info.get("value_count"),
    )
    checkpoint_payload = build_checkpoint_payload("post_calibration")
    write_calibration_progress("saving_final_checkpoint")
    torch.save(checkpoint_payload, checkpoint_path)
    write_calibration_progress("final_checkpoint_saved")
    prediction_workers = int(config.get("prediction_workers", 1))
    if prediction_workers > 1 and len(target_record_paths) > 1:
        final_action_history_len = int(stats.get("action_history_len", 0) or 0)
        final_residual_mouse = _residual_mouse_from_mode(_mouse_target_mode(config))
        if final_action_history_len > 0 and not _action_history_parallel_by_path(config):
            raise ValueError(
                "parallel prediction with action_history_len>0 requires action_history_parallel_by_path=true "
                "and target record paths that preserve complete recording order"
            )
        if final_residual_mouse and not _action_history_parallel_by_path(config):
            raise ValueError(
                "parallel prediction with residual_last_seen mouse state requires action_history_parallel_by_path=true "
                "and target record paths that preserve complete recording order"
            )
        if final_action_history_len > 0:
            seed_state, seed_source = _action_history_seed_state_for_parallel_prediction(
                config,
                train_paths=train_record_paths,
                history_len=final_action_history_len,
                stats=stats,
            )
            config["_action_history_seed_state"] = seed_state
            config["action_history_seed_state_summary"] = {
                "source": seed_source,
                "recordings": int(seed_state.get("recordings", 0)),
                "history_len": int(seed_state.get("history_len", final_action_history_len)),
            }
        if _residual_mouse_from_mode(_mouse_target_mode(config)):
            mouse_seed_state = _streaming_mouse_seed_state_for_parallel_prediction(
                config,
                train_paths=train_record_paths,
                stats=stats,
            )
            config["_streaming_mouse_seed_state"] = mouse_seed_state
            config["streaming_mouse_seed_state_summary"] = {
                "source": mouse_seed_state.get("source"),
                "recordings": int(mouse_seed_state.get("recordings", 0)),
                "games": int(mouse_seed_state.get("games", 0)),
                "workers": int(mouse_seed_state.get("workers", 1)),
            }
        if str(device).startswith("cuda"):
            model.to("cpu")
            torch.cuda.empty_cache()
        prediction = _predict_streaming_idm_checkpoint_parallel(
            config,
            checkpoint=checkpoint_payload,
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
            target_record_paths=target_record_paths,
            prediction_config_base=config,
        )
    else:
        prediction = _predict_stream(
            torch,
            model,
            target_records=target_record_paths if len(target_record_paths) > 1 else target_record_paths[0],
            stats=stats,
            config=config,
            device=device,
            category_vocab=category_vocab,
            mouse_axis_classes=mouse_axis_classes,
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
        )
    config.pop("_action_history_seed_state", None)
    config.pop("_streaming_mouse_seed_state", None)
    config_fingerprint = stable_hash_json(config)
    resolved_config_path = out_dir / "resolved_config.json"
    write_json(
        resolved_config_path,
        {
            "schema": "streaming_idm_resolved_config.v1",
            "model": str(config.get("model_name", "streaming_compact_idm")),
            "config": config,
            "config_fingerprint": config_fingerprint,
        },
    )
    data_universe_path = config.get("data_universe")
    split_contract_path = config.get("split_contract")
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(config.get("model_name", "streaming_compact_idm")),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "config_fingerprint": config_fingerprint,
        "config_path": str(config.get("config_path", "")),
        "resolved_config_path": str(resolved_config_path),
        "train_records": int(stats["num_examples"]),
        "target_records": int(prediction["target_records"]),
        "train_records_path": str(train_records),
        "target_records_path": str(target_records),
        "data_universe": _file_artifact_metadata(data_universe_path),
        "data_universe_fingerprint": _json_fingerprint(data_universe_path),
        "split_contract": _file_artifact_metadata(split_contract_path),
        "split_contract_fingerprint": _json_fingerprint(split_contract_path),
        "split_id": str(config.get("split_id") or _json_fingerprint(split_contract_path) or "d2e_full_split_contract"),
        "source_namespace": str(config.get("source_namespace", "d2e_full_corpus")),
        "source_ids": list(stats.get("source_ids", [])),
        "resolution_tiers": list(stats.get("resolution_tiers", [])),
        "target_source_ids": list(prediction.get("target_source_ids", [])),
        "target_resolution_tiers": list(prediction.get("target_resolution_tiers", [])),
        "split_names": list(stats.get("split_names", [])),
        "eval_split_tags": list(stats.get("eval_split_tags", [])),
        "target_eval_split_tags": list(prediction.get("target_eval_split_tags", [])),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "filtered_pseudo_label_path": prediction["pseudo_label_path"],
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": prediction["metrics_path"],
        "label_quality_report_path": prediction["label_quality_report_path"],
        "statistical_comparison_path": prediction["statistical_comparison_path"],
        "convergence_report_path": str(out_dir / "convergence_report.json"),
        "convergence_plateau_met": bool(convergence_report.get("plateau_met", False)),
        "calibration": {
            **calibration_info,
            "softmax": softmax_calibration_info,
            "keyboard_softmax_threshold": float(config.get("keyboard_softmax_threshold", 0.5)),
            "button_softmax_threshold": float(config.get("button_softmax_threshold", 0.5)),
            "mouse_output_gain": mouse_output_gain,
            "mouse_output_gain_x": config.get("mouse_output_gain_x"),
            "mouse_output_gain_y": config.get("mouse_output_gain_y"),
            "mouse_output_gain_info": mouse_output_gain_info,
            "last_train_loss": history[-1]["loss"] if history else None,
            "prediction_fingerprint": prediction["prediction_fingerprint"],
        },
        "feature_mode": feature_mode,
        "input_dim": int(stats["input_dim"]),
        "action_history_len": int(stats.get("action_history_len", _action_history_len_from_config(config)) or 0),
        "action_history_dim": int(stats.get("action_history_dim", 0) or 0),
        "action_history_feedback": "autoregressive_predicted" if int(stats.get("action_history_len", 0) or 0) > 0 else "none",
        "categorical_vocab": category_vocab,
        "keyboard_head_mode": keyboard_head_mode,
        "keyboard_classes": [list(tokens) for tokens in keyboard_classes],
        "button_head_mode": button_head_mode,
        "button_classes": [list(tokens) for tokens in button_classes],
        "mouse_head_mode": mouse_head_mode,
        "mouse_target_mode": _mouse_target_mode(config),
        "mouse_emit_mode": str(config.get("mouse_emit_mode", "single")),
        "mouse_max_tokens_per_axis": int(config.get("mouse_max_tokens_per_axis", 8)),
        "mouse_axis_classes": mouse_axis_classes if mouse_head_mode == "axis_softmax" else [],
        "distributed": {
            "enabled": bool(dist["enabled"]),
            "world_size": int(dist["world_size"]),
            "backend": dist["backend"],
            "rank0_device": device,
        },
        "training_cache": {
            "enabled": bool(training_cache_manifests),
            "dir": str(config.get("training_cache_dir", "")),
            "manifest_paths": [str(row.get("manifest_path")) for row in training_cache_manifests],
            "rows": sum(_training_cache_manifest_row_count(row) for row in training_cache_manifests),
            "chunk_size": int(config.get("training_cache_chunk_size", config.get("batch_size", 4096) * 2))
            if config.get("training_cache_dir")
            else None,
            "shard_by_path": bool(config.get("training_cache_shard_by_path", len(train_record_paths) > 1)),
            "progress_interval_batches": int(config.get("training_progress_interval_batches", 0) or 0),
            "shard_assignment": _training_cache_assignment_plan(
                training_cache_manifests,
                world_size=int(dist["world_size"]),
                mode=str(config.get("training_cache_shard_assignment", "greedy_rows")),
            )
            if training_cache_manifests
            else None,
        },
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    summary = {
        "schema": "streaming_idm_train_summary.v1",
        "metadata": metadata,
        "metrics": prediction["metrics"],
        "label_quality_report": prediction["label_quality_report"],
        "statistical_comparison": prediction["statistical_comparison"],
        "convergence_report": convergence_report,
        "history_tail": history[-5:],
        "device": device,
        "stats_path": str(stats_path),
        "predictions_path": prediction["predictions_path"],
        "prediction_resume": prediction["prediction_resume"],
    }
    write_json(config.get("summary_out", out_dir / "summary.json"), summary)
    return summary
