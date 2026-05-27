from __future__ import annotations

import glob
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import (
    _PaperMetricAccumulator,
    _button_tokens,
    _expand_paths,
    _iter_jsonl,
    _key_tokens,
    _split_tags,
)
from fdm_d2e.io_utils import write_json


_SPLITS = ("temporal", "heldout_recording", "heldout_game")
_COMPONENTS = ("keyboard", "mouse_button", "mouse_move")
_PAPER_TARGET_PATHS: dict[str, tuple[str, ...]] = {
    "pearson_x": ("paper_compatible", "mouse_move", "pearson_x"),
    "pearson_y": ("paper_compatible", "mouse_move", "pearson_y"),
    "keyboard_accuracy": ("paper_compatible", "keyboard", "key_accuracy"),
    "mouse_button_accuracy": ("paper_compatible", "mouse_button", "button_accuracy"),
    "scale_ratio_x": ("paper_compatible", "mouse_move", "scale_ratio_x"),
    "scale_ratio_y": ("paper_compatible", "mouse_move", "scale_ratio_y"),
}


def _get(data: dict[str, Any] | None, path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return ()


def _counter_tuple(counter: Counter[str]) -> tuple[str, ...]:
    return tuple(sorted(counter.elements()))


def _component_tokens(tokens: Sequence[str], component: str) -> tuple[str, ...]:
    if component == "keyboard":
        return _counter_tuple(_key_tokens(tokens))
    if component == "mouse_button":
        return _counter_tuple(_button_tokens(tokens))
    if component == "mouse_move":
        return tuple(sorted(token for token in tokens if token.startswith(("MOUSE_DX_", "MOUSE_DY_"))))
    raise ValueError(f"unknown IDM component: {component}")


def _merge_components(components: dict[str, tuple[str, ...]]) -> list[str]:
    merged: list[str] = []
    for name in _COMPONENTS:
        merged.extend(components.get(name, ()))
    return merged or ["NOOP"]


def _recording_id(row: dict[str, Any]) -> str:
    value = row.get("recording_id")
    if isinstance(value, str) and value:
        return value
    sequence_id = row.get("sequence_id")
    if isinstance(sequence_id, str) and "#" in sequence_id:
        return sequence_id.rsplit("#", 1)[0]
    return str(sequence_id or "__unknown_recording__")


def _game(row: dict[str, Any]) -> str:
    value = row.get("game")
    if isinstance(value, str) and value:
        return value
    recording = _recording_id(row)
    if ":" in recording:
        recording = recording.split(":", 1)[1]
    if "/" in recording:
        return recording.split("/", 1)[0]
    return "__unknown_game__"


def _held_keys(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            token[len("KEY_DOWN_") :]
            for token in _as_tokens(row.get("prior_action_tokens"))
            if token.startswith("KEY_DOWN_")
        )
    )


def _held_buttons(row: dict[str, Any]) -> tuple[str, ...]:
    buttons: list[str] = []
    for token in _as_tokens(row.get("prior_action_tokens")):
        if token.startswith("MOUSE_") and token.endswith("_DOWN") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.append(token[len("MOUSE_") : -len("_DOWN")])
    return tuple(sorted(buttons))


def _bucket(value: Any) -> str:
    if value is None:
        return "na"
    try:
        ivalue = max(0, int(value))
    except Exception:
        return "na"
    for bound in (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987):
        if ivalue <= bound:
            return f"le{bound}"
    return "gt987"


def _duration_features(row: dict[str, Any], field: str, active: Sequence[str]) -> tuple[tuple[str, str], ...]:
    values = row.get(field)
    if not isinstance(values, dict):
        return ()
    return tuple((str(key), _bucket(values.get(str(key)))) for key in active)


def _previous_key_button(row: dict[str, Any]) -> tuple[str, ...]:
    tokens = _as_tokens(row.get("previous_event_tokens"))
    return tuple(
        sorted(
            token
            for token in tokens
            if token.startswith("KEY_")
            or (
                token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
                and not token.startswith(("MOUSE_DX_", "MOUSE_DY_"))
            )
        )
    )


def _previous_motion_signature(row: dict[str, Any]) -> tuple[str, ...]:
    tokens = _as_tokens(row.get("previous_event_tokens"))
    dx = 0
    dy = 0
    saw_x = False
    saw_y = False
    # Import lazily to keep this helper independent of optional metric changes.
    from fdm_d2e.tokenization.actions import token_to_delta_class

    for token in tokens:
        value = token_to_delta_class(token)
        if value is None:
            continue
        if token.startswith("MOUSE_DX_"):
            dx += int(value)
            saw_x = True
        elif token.startswith("MOUSE_DY_"):
            dy += int(value)
            saw_y = True
    def sign(value: int, saw: bool) -> str:
        if not saw or value == 0:
            return "z"
        return "p" if value > 0 else "n"

    return (f"dx:{sign(dx, saw_x)}", f"dy:{sign(dy, saw_y)}")


def _state_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return ("game", _game(row), "keys", _held_keys(row), "buttons", _held_buttons(row))


def _state_duration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    keys = _held_keys(row)
    buttons = _held_buttons(row)
    return (
        *_state_key(row),
        "key_hold",
        _duration_features(row, "prior_key_hold_bins", keys),
        "button_hold",
        _duration_features(row, "prior_button_hold_bins", buttons),
        "key_age",
        _bucket(row.get("prior_since_key_transition_bins")),
        "button_age",
        _bucket(row.get("prior_since_button_transition_bins")),
    )


def _previous_event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return ("game", _game(row), "prev_kb", _previous_key_button(row), "prev_motion", _previous_motion_signature(row))


def _state_previous_duration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*_state_duration_key(row), "prev", _previous_key_button(row), "prev_motion", _previous_motion_signature(row))


def _recording_state_previous_duration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        "recording",
        _recording_id(row),
        "state_prev_duration",
        _state_previous_duration_key(row),
    )


FeatureFn = Callable[[dict[str, Any]], tuple[Any, ...]]


_FEATURE_SPECS: dict[str, FeatureFn] = {
    "global": lambda row: ("global",),
    "game": lambda row: ("game", _game(row)),
    "game_prior_state": _state_key,
    "game_prior_state_duration": _state_duration_key,
    "game_previous_event": _previous_event_key,
    "game_prior_previous_duration": _state_previous_duration_key,
    "recording_prior_previous_duration": _recording_state_previous_duration_key,
}


def _row_fields() -> list[str]:
    return [
        "sequence_id",
        "recording_id",
        "game",
        "eval_split_tags",
        "split_tags",
        "ground_truth_tokens",
        "prior_action_tokens",
        "prior_key_hold_bins",
        "prior_button_hold_bins",
        "prior_since_key_transition_bins",
        "prior_since_button_transition_bins",
        "previous_event_tokens",
    ]


def _iter_limited_jsonl(
    paths: Sequence[Path],
    *,
    max_rows: int | None,
    max_rows_per_path: int | None,
) -> Iterable[dict[str, Any]]:
    total = 0
    fields = _row_fields()
    for path in paths:
        path_rows = 0
        for row in _iter_jsonl([path], fields=fields):
            if max_rows is not None and total >= max_rows:
                return
            if max_rows_per_path is not None and path_rows >= max_rows_per_path:
                break
            total += 1
            path_rows += 1
            yield row


@dataclass(slots=True)
class _ModeTables:
    counters: dict[str, dict[str, dict[tuple[Any, ...], Counter[tuple[str, ...]]]]]
    counts: dict[str, Counter[tuple[Any, ...]]]
    train_rows: int


def _empty_group_accumulators(split_tags: Sequence[str]) -> dict[str, _PaperMetricAccumulator]:
    groups = {"all": _PaperMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        groups[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=False)
    return groups


def _update_groups(
    groups: dict[str, _PaperMetricAccumulator],
    *,
    predicted_tokens: Sequence[str],
    target_row: dict[str, Any],
    split_tags: Sequence[str],
) -> None:
    ground_truth = _as_tokens(target_row.get("ground_truth_tokens"))
    groups["all"].update(predicted_tokens, ground_truth)
    active = set(_split_tags(target_row))
    for tag in split_tags:
        if tag in active:
            groups[f"eval_split:{tag}"].update(predicted_tokens, ground_truth)


def _groups_metrics(groups: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {key: accumulator.metrics() for key, accumulator in sorted(groups.items())}


def _build_tables(
    train_paths: Sequence[Path],
    *,
    max_train_rows: int | None,
    max_train_rows_per_path: int | None,
    progress_output_path: str | Path | None,
    progress_rows: int,
) -> _ModeTables:
    counters: dict[str, dict[str, dict[tuple[Any, ...], Counter[tuple[str, ...]]]]] = {
        spec: {component: defaultdict(Counter) for component in _COMPONENTS} for spec in _FEATURE_SPECS
    }
    counts: dict[str, Counter[tuple[Any, ...]]] = {spec: Counter() for spec in _FEATURE_SPECS}
    rows = 0
    for row in _iter_limited_jsonl(
        train_paths,
        max_rows=max_train_rows,
        max_rows_per_path=max_train_rows_per_path,
    ):
        rows += 1
        tokens = _as_tokens(row.get("ground_truth_tokens"))
        for spec, builder in _FEATURE_SPECS.items():
            key = builder(row)
            counts[spec][key] += 1
            for component in _COMPONENTS:
                counters[spec][component][key][_component_tokens(tokens, component)] += 1
        if progress_output_path and progress_rows > 0 and rows % progress_rows == 0:
            write_json(
                progress_output_path,
                {
                    "schema": "idm_paper_target_feasibility_progress.v1",
                    "status": "building_train_tables",
                    "train_rows": rows,
                    "train_paths": [str(path) for path in train_paths],
                },
            )
    return _ModeTables(counters=counters, counts=counts, train_rows=rows)


def _mode_category(
    tables: _ModeTables,
    *,
    spec: str,
    component: str,
    row: dict[str, Any],
    min_feature_count: int,
) -> tuple[str, ...]:
    for candidate_spec in (spec, "game", "global"):
        key = _FEATURE_SPECS[candidate_spec](row)
        if tables.counts[candidate_spec].get(key, 0) < min_feature_count:
            continue
        counter = tables.counters[candidate_spec][component].get(key)
        if counter:
            return counter.most_common(1)[0][0]
    return ()


def _mode_prediction(tables: _ModeTables, *, spec: str, row: dict[str, Any], min_feature_count: int) -> list[str]:
    return _merge_components(
        {
            component: _mode_category(
                tables,
                spec=spec,
                component=component,
                row=row,
                min_feature_count=min_feature_count,
            )
            for component in _COMPONENTS
        }
    )


def _oracle_seen_prediction(tables: _ModeTables, *, spec: str, row: dict[str, Any], min_feature_count: int) -> list[str]:
    key = _FEATURE_SPECS[spec](row)
    gt_tokens = _as_tokens(row.get("ground_truth_tokens"))
    components: dict[str, tuple[str, ...]] = {}
    for component in _COMPONENTS:
        gt_component = _component_tokens(gt_tokens, component)
        if gt_component in tables.counters[spec][component].get(key, Counter()):
            components[component] = gt_component
        else:
            components[component] = _mode_category(
                tables,
                spec=spec,
                component=component,
                row=row,
                min_feature_count=min_feature_count,
            )
    return _merge_components(components)


def _prior_state_tokens(row: dict[str, Any]) -> list[str]:
    tokens = [token for token in _as_tokens(row.get("prior_action_tokens")) if token != "NOOP"]
    return tokens or ["NOOP"]


def _previous_event_tokens(row: dict[str, Any]) -> list[str]:
    tokens = [token for token in _as_tokens(row.get("previous_event_tokens")) if token != "NOOP"]
    return tokens or ["NOOP"]


def _predictors(tables: _ModeTables, *, min_feature_count: int) -> dict[str, Callable[[dict[str, Any]], list[str]]]:
    predictors: dict[str, Callable[[dict[str, Any]], list[str]]] = {
        "empty_noop": lambda row: ["NOOP"],
        "copy_previous_event": _previous_event_tokens,
        "copy_prior_held_state": _prior_state_tokens,
    }
    for spec in (
        "game",
        "game_prior_state",
        "game_prior_state_duration",
        "game_previous_event",
        "game_prior_previous_duration",
        "recording_prior_previous_duration",
    ):
        predictors[f"lookup_{spec}_mode"] = (
            lambda row, spec=spec: _mode_prediction(
                tables,
                spec=spec,
                row=row,
                min_feature_count=min_feature_count,
            )
        )
        predictors[f"oracle_seen_{spec}"] = (
            lambda row, spec=spec: _oracle_seen_prediction(
                tables,
                spec=spec,
                row=row,
                min_feature_count=min_feature_count,
            )
        )
    return predictors


def _load_paper_targets(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text())
    targets = _get(payload, ["target_sequence", "phase_1", "primary_targets"])
    if not isinstance(targets, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("pearson_x", "pearson_y", "keyboard_accuracy", "mouse_button_accuracy"):
        value = targets.get(key)
        if value is not None:
            result[key] = float(value)
    if targets.get("scale_ratio_x_max") is not None:
        result["scale_ratio_x"] = float(targets["scale_ratio_x_max"])
    if targets.get("scale_ratio_y_max") is not None:
        result["scale_ratio_y"] = float(targets["scale_ratio_y_max"])
    return result


def _load_baseline(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text())
    all_metrics = _get(payload, ["groups", "all"])
    if not isinstance(all_metrics, dict):
        return {}
    values: dict[str, float] = {}
    for name, metric_path in _PAPER_TARGET_PATHS.items():
        value = _get(all_metrics, metric_path)
        if value is not None:
            values[name] = float(value)
    f1 = _get(all_metrics, ["strict_local", "mouse_button", "f1"])
    fpr = _get(all_metrics, ["strict_local", "mouse_button", "no_button_false_positive_rate"])
    if f1 is not None:
        values["mouse_button_f1"] = float(f1)
    if fpr is not None:
        values["no_button_false_positive_rate"] = float(fpr)
    return values


def _summarize_predictor(
    name: str,
    groups: dict[str, Any],
    *,
    paper_targets: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    all_metrics = groups["all"]
    values = {
        metric_name: _get(all_metrics, metric_path)
        for metric_name, metric_path in _PAPER_TARGET_PATHS.items()
    }
    strict_f1 = _get(all_metrics, ["strict_local", "mouse_button", "f1"])
    fpr = _get(all_metrics, ["strict_local", "mouse_button", "no_button_false_positive_rate"])
    if strict_f1 is not None:
        values["mouse_button_f1"] = strict_f1
    if fpr is not None:
        values["no_button_false_positive_rate"] = fpr
    paper_passes: dict[str, bool | None] = {}
    for metric_name, target in paper_targets.items():
        actual = values.get(metric_name)
        if actual is None:
            paper_passes[metric_name] = None
        elif metric_name.startswith("scale_ratio"):
            paper_passes[metric_name] = float(actual) <= target
        else:
            paper_passes[metric_name] = float(actual) >= target
    baseline_delta: dict[str, float | None] = {}
    for metric_name, base_value in baseline.items():
        actual = values.get(metric_name)
        baseline_delta[metric_name] = None if actual is None else float(actual) - float(base_value)
    return {
        "name": name,
        "rows": all_metrics.get("rows"),
        "values": values,
        "paper_passes": paper_passes,
        "paper_pass_count": sum(1 for value in paper_passes.values() if value is True),
        "baseline_delta": baseline_delta,
    }


def _table_cardinality(tables: _ModeTables) -> dict[str, Any]:
    return {
        spec: {
            "feature_keys": len(tables.counts[spec]),
            "rows": sum(tables.counts[spec].values()),
            "top_feature_counts": [
                {"count": count, "key_preview": repr(key)[:240]}
                for key, count in tables.counts[spec].most_common(5)
            ],
        }
        for spec in _FEATURE_SPECS
    }


def build_idm_paper_target_feasibility(
    *,
    train_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    split_tags: Sequence[str] = _SPLITS,
    model_name: str = "idm_paper_target_feasibility",
    max_train_rows: int | None = None,
    max_target_rows: int | None = None,
    max_train_rows_per_path: int | None = None,
    max_target_rows_per_path: int | None = None,
    min_feature_count: int = 3,
    paper_contract_path: str | Path | None = None,
    baseline_metrics_path: str | Path | None = None,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    started = time.time()
    expanded_train = _expand_paths(train_paths)
    expanded_target = _expand_paths(target_paths)
    findings: list[dict[str, Any]] = []
    if not expanded_train:
        findings.append({"severity": "error", "code": "missing_train_paths", "patterns": [str(path) for path in train_paths]})
    if not expanded_target:
        findings.append({"severity": "error", "code": "missing_target_paths", "patterns": [str(path) for path in target_paths]})

    tables = _ModeTables(counters={}, counts={}, train_rows=0)
    predictor_metrics: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    alignment = {"target_rows": 0}
    if expanded_train and expanded_target:
        tables = _build_tables(
            expanded_train,
            max_train_rows=max_train_rows,
            max_train_rows_per_path=max_train_rows_per_path,
            progress_output_path=progress_output_path,
            progress_rows=progress_rows,
        )
        predictors = _predictors(tables, min_feature_count=min_feature_count)
        accumulators = {name: _empty_group_accumulators(split_tags) for name in predictors}
        for row in _iter_limited_jsonl(
            expanded_target,
            max_rows=max_target_rows,
            max_rows_per_path=max_target_rows_per_path,
        ):
            alignment["target_rows"] += 1
            for name, predict in predictors.items():
                _update_groups(
                    accumulators[name],
                    predicted_tokens=predict(row),
                    target_row=row,
                    split_tags=split_tags,
                )
            if progress_output_path and progress_rows > 0 and alignment["target_rows"] % progress_rows == 0:
                write_json(
                    progress_output_path,
                    {
                        "schema": "idm_paper_target_feasibility_progress.v1",
                        "status": "scoring_target_rows",
                        "train_rows": tables.train_rows,
                        "target_rows": alignment["target_rows"],
                        "predictors": len(predictors),
                    },
                )
        predictor_metrics = {name: _groups_metrics(groups) for name, groups in sorted(accumulators.items())}
        paper_targets = _load_paper_targets(paper_contract_path)
        baseline = _load_baseline(baseline_metrics_path)
        summaries = [
            _summarize_predictor(name, metrics, paper_targets=paper_targets, baseline=baseline)
            for name, metrics in predictor_metrics.items()
        ]
        summaries = sorted(
            summaries,
            key=lambda row: (
                int(row.get("paper_pass_count") or 0),
                float((row.get("values") or {}).get("keyboard_accuracy") or -1.0),
                float((row.get("values") or {}).get("mouse_button_accuracy") or -1.0),
                float((row.get("values") or {}).get("pearson_x") or -1.0),
            ),
            reverse=True,
        )

    errors = [item for item in findings if item.get("severity") == "error"]
    best = summaries[0] if summaries else None
    baseline = _load_baseline(baseline_metrics_path)
    context_keyboard = baseline.get("keyboard_accuracy")
    context_button = baseline.get("mouse_button_accuracy")
    best_values = (best or {}).get("values", {}) if isinstance(best, dict) else {}
    beats_context_events = (
        context_keyboard is not None
        and context_button is not None
        and best_values.get("keyboard_accuracy") is not None
        and best_values.get("mouse_button_accuracy") is not None
        and float(best_values["keyboard_accuracy"]) > float(context_keyboard)
        and float(best_values["mouse_button_accuracy"]) > float(context_button)
    )
    recommendation = {
        "status": "diagnostic_only",
        "scale_new_full_gpu_run": bool(beats_context_events),
        "reason": (
            "best prefix rule/oracle diagnostic beats the current event-state-duration-context keyboard and button paper metrics"
            if beats_context_events
            else "no prefix rule/oracle diagnostic beats the current event-state-duration-context keyboard and button paper metrics; avoid another full GPU run until target alignment or feature support is improved"
        ),
        "current_context_baseline": {
            "keyboard_accuracy": context_keyboard,
            "mouse_button_accuracy": context_button,
        },
    }
    return {
        "schema": "idm_paper_target_feasibility.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "model_name": model_name,
        "train_paths": [str(path) for path in expanded_train],
        "target_paths": [str(path) for path in expanded_target],
        "limits": {
            "max_train_rows": max_train_rows,
            "max_target_rows": max_target_rows,
            "max_train_rows_per_path": max_train_rows_per_path,
            "max_target_rows_per_path": max_target_rows_per_path,
            "min_feature_count": min_feature_count,
        },
        "alignment": alignment,
        "paper_targets": _load_paper_targets(paper_contract_path),
        "baseline_metrics": baseline,
        "table_cardinality": _table_cardinality(tables) if tables.counts else {},
        "predictor_summaries": summaries,
        "predictors": predictor_metrics,
        "recommendation": recommendation,
        "findings": findings,
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "CPU/IO-only feasibility diagnostic over train-derived rules and train-support oracles. It is not a trained IDM completion artifact and must not checkpoint G005.",
    }


def write_idm_paper_target_feasibility(
    *,
    train_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    output_path: str | Path,
    split_tags: Sequence[str] = _SPLITS,
    model_name: str = "idm_paper_target_feasibility",
    max_train_rows: int | None = None,
    max_target_rows: int | None = None,
    max_train_rows_per_path: int | None = None,
    max_target_rows_per_path: int | None = None,
    min_feature_count: int = 3,
    paper_contract_path: str | Path | None = None,
    baseline_metrics_path: str | Path | None = None,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    payload = build_idm_paper_target_feasibility(
        train_paths=train_paths,
        target_paths=target_paths,
        split_tags=split_tags,
        model_name=model_name,
        max_train_rows=max_train_rows,
        max_target_rows=max_target_rows,
        max_train_rows_per_path=max_train_rows_per_path,
        max_target_rows_per_path=max_target_rows_per_path,
        min_feature_count=min_feature_count,
        paper_contract_path=paper_contract_path,
        baseline_metrics_path=baseline_metrics_path,
        progress_output_path=progress_output_path,
        progress_rows=progress_rows,
    )
    write_json(output_path, payload)
    return payload

