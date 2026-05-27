from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        matches = sorted(glob.glob(text))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(text)
        if path.exists():
            paths.append(path)
    return paths


def _iter_rows(paths: Sequence[str | Path], *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    rows = 0
    for path in _expand_paths(paths):
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and rows >= max_rows:
                    return
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                rows += 1


def _iter_rows_with_next(paths: Sequence[str | Path], *, max_rows: int | None = None) -> Iterable[tuple[dict[str, Any], dict[str, Any] | None]]:
    iterator = iter(_iter_rows(paths, max_rows=max_rows))
    try:
        current = next(iterator)
    except StopIteration:
        return
    for nxt in iterator:
        yield current, nxt
        current = nxt
    yield current, None


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(item) for item in value] if isinstance(value, list) else []


def _previous_tokens(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in row.get("previous_event_tokens") or []]


def _prior_action_tokens(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in row.get("prior_action_tokens") or []]


def previous_motion_tokens(row: dict[str, Any]) -> list[str]:
    return [token for token in _previous_tokens(row) if token.startswith(("MOUSE_DX_", "MOUSE_DY_"))]


def previous_key_button_tokens(row: dict[str, Any]) -> list[str]:
    return [
        token
        for token in _previous_tokens(row)
        if token.startswith(("KEY_", "MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
    ]


def prior_down_as_press_tokens(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for token in _prior_action_tokens(row):
        if token.startswith("KEY_DOWN_"):
            out.append("KEY_PRESS_" + token.removeprefix("KEY_DOWN_"))
        elif token in {"MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN", "MOUSE_MIDDLE_DOWN"}:
            out.append(token)
    return out


def _recording_key(row: dict[str, Any]) -> str | None:
    for key in ("source_recording_key", "recording_id", "source_recording_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    seq = row.get("sequence_id")
    if isinstance(seq, str) and "#" in seq:
        return seq.rsplit("#", 1)[0]
    return str(seq) if isinstance(seq, str) and seq else None


def prior_state_sets(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    buttons: set[str] = set()
    for token in _prior_action_tokens(row):
        if token.startswith("KEY_DOWN_"):
            keys.add(token.removeprefix("KEY_DOWN_"))
        elif token in {"MOUSE_LEFT_DOWN", "MOUSE_RIGHT_DOWN", "MOUSE_MIDDLE_DOWN"}:
            buttons.add(token.rsplit("_DOWN", 1)[0])
    return keys, buttons


def state_delta_tokens(row: dict[str, Any], next_row: dict[str, Any] | None) -> list[str]:
    """Noncausal diagnostic: infer current events from current/next held-state metadata."""

    if next_row is None or _recording_key(row) != _recording_key(next_row):
        return []
    keys0, buttons0 = prior_state_sets(row)
    keys1, buttons1 = prior_state_sets(next_row)
    out: list[str] = []
    out.extend(f"KEY_PRESS_{key}" for key in sorted(keys1 - keys0))
    out.extend(f"KEY_RELEASE_{key}" for key in sorted(keys0 - keys1))
    out.extend(f"{button}_DOWN" for button in sorted(buttons1 - buttons0))
    out.extend(f"{button}_UP" for button in sorted(buttons0 - buttons1))
    return out


def merge_motion_and_categorical(motion: Sequence[str], categorical: Sequence[str]) -> list[str]:
    """Keep repeated mouse-delta tokens but dedupe categorical event tokens."""

    out = [str(token) for token in motion]
    seen: set[str] = set()
    for token in categorical:
        text = str(token)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _metrics_payload(accumulators: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: {"all": accumulator.metrics()} for name, accumulator in sorted(accumulators.items())}


def build_previous_context_heuristics(*, target_paths: Sequence[str | Path], max_rows: int | None = None) -> dict[str, Any]:
    policies: dict[str, Callable[[dict[str, Any]], list[str]]] = {
        "previous_all": _previous_tokens,
        "previous_motion_only": previous_motion_tokens,
        "previous_key_button_only": previous_key_button_tokens,
        "prior_down_as_press": prior_down_as_press_tokens,
        "empty": lambda _row: [],
    }
    accs = {name: _PaperMetricAccumulator(empty_bins_as_correct=False) for name in policies}
    rows = 0
    for row in _iter_rows(target_paths, max_rows=max_rows):
        gt = _tokens(row, "ground_truth_tokens")
        for name, policy in policies.items():
            accs[name].update(policy(row), gt)
        rows += 1
    return {
        "schema": "g005_prefix_heuristic_metrics.v1",
        "rows": rows,
        "target_paths": [str(path) for path in target_paths],
        "policies": _metrics_payload(accs),
        "claim_boundary": "Causal/context-only diagnostic; not G005 completion evidence.",
    }


def build_state_delta_oracle_metrics(*, target_paths: Sequence[str | Path], max_rows: int | None = None) -> dict[str, Any]:
    policies: dict[str, Callable[[dict[str, Any], dict[str, Any] | None], list[str]]] = {
        "next_state_delta_only": state_delta_tokens,
        "next_state_delta_plus_prev_motion": lambda row, nxt: previous_motion_tokens(row) + state_delta_tokens(row, nxt),
        "previous_motion_only": lambda row, _nxt: previous_motion_tokens(row),
    }
    accs = {name: _PaperMetricAccumulator(empty_bins_as_correct=False) for name in policies}
    rows = 0
    for row, next_row in _iter_rows_with_next(target_paths, max_rows=max_rows):
        gt = _tokens(row, "ground_truth_tokens")
        for name, policy in policies.items():
            accs[name].update(policy(row, next_row), gt)
        rows += 1
    return {
        "schema": "g005_state_delta_oracle_metrics.v1",
        "rows": rows,
        "target_paths": [str(path) for path in target_paths],
        "policies": _metrics_payload(accs),
        "claim_boundary": "Noncausal upper-bound diagnostic: next_state_delta policies use future held-state metadata.",
    }


def _repeat_context_global_hold_since(code: str, hold: int, since: int) -> tuple[int, int]:
    del code
    return (min(int(hold), 256), min(int(since), 64))


def _repeat_context_code_hold_since(code: str, hold: int, since: int) -> tuple[str, int, int]:
    return (str(code), min(int(hold), 256), min(int(since), 64))


_REPEAT_CONTEXTS: dict[str, Callable[[str, int, int], tuple[Any, ...]]] = {
    "global_hold_since": _repeat_context_global_hold_since,
    "code_hold_since": _repeat_context_code_hold_since,
}


def _prior_key_holds(row: dict[str, Any]) -> dict[str, int]:
    value = row.get("prior_key_hold_bins")
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _prior_since_key_transition(row: dict[str, Any]) -> int:
    try:
        return int(row.get("prior_since_key_transition_bins") or 0)
    except (TypeError, ValueError):
        return 0


def train_key_repeat_priors(
    *,
    train_paths: Sequence[str | Path],
    max_rows: int | None = None,
    min_support: int = 3,
) -> dict[str, dict[tuple[Any, ...], tuple[float, int, int]]]:
    counts: dict[str, defaultdict[tuple[Any, ...], list[int]]] = {
        name: defaultdict(lambda: [0, 0]) for name in _REPEAT_CONTEXTS
    }
    for row in _iter_rows(train_paths, max_rows=max_rows):
        gt = set(_tokens(row, "ground_truth_tokens"))
        since = _prior_since_key_transition(row)
        for code, hold in _prior_key_holds(row).items():
            positive = int(("KEY_PRESS_" + code) in gt)
            for name, context_fn in _REPEAT_CONTEXTS.items():
                rec = counts[name][context_fn(code, hold, since)]
                rec[0] += 1
                rec[1] += positive
    return {
        name: {key: (pos / n, n, pos) for key, (n, pos) in items.items() if n >= int(min_support)}
        for name, items in counts.items()
    }


def build_key_repeat_prior_metrics(
    *,
    train_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    max_train_rows: int | None = None,
    max_target_rows: int | None = None,
    thresholds: Sequence[float] = (0.1, 0.2, 0.35, 0.5),
    min_support: int = 3,
) -> dict[str, Any]:
    priors = train_key_repeat_priors(train_paths=train_paths, max_rows=max_train_rows, min_support=min_support)
    policy_specs: list[tuple[str, str, Callable[[str, int, int], tuple[Any, ...]], float]] = []
    for name, context_fn in _REPEAT_CONTEXTS.items():
        for threshold in thresholds:
            policy_specs.append((f"{name}_th{threshold:g}", name, context_fn, float(threshold)))
    accs = {label: _PaperMetricAccumulator(empty_bins_as_correct=False) for label, _name, _fn, _th in policy_specs}
    rows = 0
    for row, next_row in _iter_rows_with_next(target_paths, max_rows=max_target_rows):
        gt = _tokens(row, "ground_truth_tokens")
        motion = previous_motion_tokens(row)
        base = state_delta_tokens(row, next_row)
        holds = _prior_key_holds(row)
        since = _prior_since_key_transition(row)
        for label, name, context_fn, threshold in policy_specs:
            categorical = list(base)
            model = priors.get(name, {})
            for code, hold in holds.items():
                rec = model.get(context_fn(code, hold, since))
                if rec and rec[0] >= threshold:
                    categorical.append("KEY_PRESS_" + code)
            accs[label].update(merge_motion_and_categorical(motion, categorical), gt)
        rows += 1
    return {
        "schema": "g005_key_repeat_prior_prefix320k_metrics.v2",
        "rows": rows,
        "train_paths": [str(path) for path in train_paths],
        "target_paths": [str(path) for path in target_paths],
        "motion_source": "previous_event_tokens_with_duplicate_mouse_delta_tokens_preserved",
        "min_support": int(min_support),
        "thresholds": [float(value) for value in thresholds],
        "context_count": {name: len(model) for name, model in priors.items()},
        "policies": _metrics_payload(accs),
        "claim_boundary": "Repeat-prior diagnostic uses train-prefix statistics plus noncausal next-state delta labels; not G005 completion evidence.",
    }


def write_state_transition_diagnostics(
    *,
    train_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    output_dir: str | Path,
    max_train_rows: int | None = None,
    max_target_rows: int | None = None,
    prefix: str = "g005_idm",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous = build_previous_context_heuristics(target_paths=target_paths, max_rows=max_target_rows)
    state_delta = build_state_delta_oracle_metrics(target_paths=target_paths, max_rows=max_target_rows)
    repeat = build_key_repeat_prior_metrics(
        train_paths=train_paths,
        target_paths=target_paths,
        max_train_rows=max_train_rows,
        max_target_rows=max_target_rows,
    )
    paths = {
        "previous_context": output / f"{prefix}_prefix_context_heuristic_matrix.json",
        "state_delta_oracle": output / f"{prefix}_state_delta_oracle_prefix320k_metrics.json",
        "key_repeat_prior": output / f"{prefix}_key_repeat_prior_prefix320k_metrics.json",
    }
    write_json(paths["previous_context"], previous)
    write_json(paths["state_delta_oracle"], state_delta)
    write_json(paths["key_repeat_prior"], repeat)
    return {
        "schema": "g005_state_transition_diagnostics_summary.v1",
        "status": "pass",
        "paths": {key: str(path) for key, path in paths.items()},
        "rows": {
            "previous_context": previous["rows"],
            "state_delta_oracle": state_delta["rows"],
            "key_repeat_prior": repeat["rows"],
        },
        "claim_boundary": "Diagnostic artifact index only; state-delta and repeat-prior outputs include noncausal upper bounds.",
    }
