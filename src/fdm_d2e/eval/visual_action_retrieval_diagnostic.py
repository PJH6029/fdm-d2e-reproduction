from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

Outcome = tuple[str, ...]
ContextKey = tuple[Any, ...]
_DEFAULT_THRESHOLDS = (0.2, 0.5)
_DEFAULT_MIN_SUPPORTS = (1, 3)
_CATEGORICAL_PREFIXES = ("KEY_", "MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")
_MOTION_PREFIXES = ("MOUSE_DX_", "MOUSE_DY_")


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path) -> list[Path]:
    items = [patterns] if isinstance(patterns, (str, Path)) else list(patterns)
    paths: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(item)
        if path.exists():
            paths.append(path)
    return paths


def _iter_rows(patterns: Sequence[str | Path] | str | Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    rows = 0
    for path in _expand_paths(patterns):
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


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(token) for token in value] if isinstance(value, list) else []


def _split_tags(row: dict[str, Any]) -> list[str]:
    for key in ("eval_split_tags", "split_tags"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    return []


def _recording_game(row: dict[str, Any]) -> str:
    game = row.get("game")
    if isinstance(game, str) and game:
        return game
    rec = str(row.get("recording_id") or row.get("source_recording_key") or "")
    if ":" in rec:
        rec = rec.split(":", 1)[1]
    if "/" in rec:
        return rec.split("/", 1)[0]
    return rec or "unknown"


def _sequence_index(row: dict[str, Any]) -> int:
    seq = str(row.get("sequence_id", ""))
    match = re.search(r"#(\d+)$", seq)
    if match:
        return int(match.group(1))
    return 0


def _holds(row: dict[str, Any]) -> dict[str, int]:
    value = row.get("prior_key_hold_bins")
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _since(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("prior_since_key_transition_bins") or 0))
    except (TypeError, ValueError):
        return 0


def _bucket(value: int) -> int:
    value = max(0, int(value))
    if value <= 10:
        return value
    if value <= 20:
        return 10 + ((value - 10) // 2)
    if value <= 80:
        return 15 + ((value - 20) // 5)
    return min(40, 27 + ((value - 80) // 20))


def _hold_signature(row: dict[str, Any], *, with_bucket: bool = True) -> tuple[Any, ...]:
    items: list[Any] = []
    for code, hold in sorted(_holds(row).items())[:16]:
        items.append((code, _bucket(hold), hold % 2, hold % 4, hold % 8) if with_bucket else code)
    return tuple(items)


def _previous_key_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(token for token in _tokens(row, "previous_event_tokens") if token.startswith("KEY_"))[:12])


def _q(values: Sequence[Any], *, scale: float, limit: int | None = None) -> tuple[int, ...]:
    out: list[int] = []
    for raw in list(values)[:limit]:
        try:
            out.append(int(round(float(raw) * scale)))
        except (TypeError, ValueError):
            out.append(0)
    return tuple(out)


def _grid4(values: Sequence[Any]) -> tuple[float, ...]:
    vals: list[float] = []
    try:
        src = [float(v) for v in values]
    except (TypeError, ValueError):
        return ()
    if len(src) < 64:
        return ()
    for by in range(4):
        for bx in range(4):
            block = []
            for y in range(by * 2, by * 2 + 2):
                for x in range(bx * 2, bx * 2 + 2):
                    block.append(src[y * 8 + x])
            vals.append(sum(block) / len(block))
    return tuple(vals)


def _visual_context(row: dict[str, Any], *, style: str) -> tuple[Any, ...]:
    frame = row.get("frame") if isinstance(row.get("frame"), dict) else {}
    cur_features = frame.get("features") if isinstance(frame, dict) else []
    cur_grid = frame.get("grid8") if isinstance(frame, dict) else []
    next_features = row.get("next_frame_features") or []
    delta_features = row.get("frame_delta_features") or []
    next_grid = row.get("next_frame_grid8") or []
    if style == "features":
        return (_q(cur_features, scale=10.0, limit=5), _q(next_features, scale=10.0, limit=5), _q(delta_features, scale=50.0, limit=5))
    if style == "grid4":
        cur4 = _grid4(cur_grid)
        next4 = _grid4(next_grid)
        delta4 = tuple((b - a) for a, b in zip(cur4, next4)) if cur4 and next4 and len(cur4) == len(next4) else ()
        return (_q(cur4, scale=8.0), _q(next4, scale=8.0), _q(delta4, scale=30.0))
    raise KeyError(style)


def _context(row: dict[str, Any], name: str) -> ContextKey:
    game = _recording_game(row)
    held = _hold_signature(row)
    since = _bucket(_since(row))
    phase = (_sequence_index(row) % 2, _sequence_index(row) % 4, _sequence_index(row) % 8)
    if name == "state_only":
        return (game, held, since, _previous_key_signature(row), phase)
    if name == "state_visual_features":
        return (game, held, since, _visual_context(row, style="features"))
    if name == "state_visual_grid4":
        return (game, held, since, _visual_context(row, style="grid4"))
    if name == "visual_grid4":
        return (game, _visual_context(row, style="grid4"))
    if name == "visual_features":
        return (game, _visual_context(row, style="features"))
    raise KeyError(f"unknown visual retrieval context {name}")


_CONTEXT_NAMES = ("state_visual_grid4", "state_visual_features", "visual_grid4", "visual_features", "state_only")


def _outcome(tokens: Sequence[str]) -> Outcome:
    # Official D2E metrics count event multiplicity within each bin, but not order.
    return tuple(sorted(str(token) for token in tokens if str(token) != "NOOP"))


def _motion_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith(_MOTION_PREFIXES)]


def _categorical_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith(_CATEGORICAL_PREFIXES)]


def _merge_max_counts(base_tokens: Sequence[str], retrieved_tokens: Sequence[str]) -> list[str]:
    base_counts = Counter(str(token) for token in base_tokens)
    ret_counts = Counter(str(token) for token in retrieved_tokens)
    ordered = list(dict.fromkeys([str(token) for token in list(base_tokens) + list(retrieved_tokens)]))
    out: list[str] = []
    for token in ordered:
        out.extend([token] * max(base_counts.get(token, 0), ret_counts.get(token, 0)))
    return out


def _summary(group: dict[str, Any]) -> dict[str, Any]:
    pc = group["paper_compatible"]
    strict = group["strict_local"]
    return {
        "keyboard_accuracy": pc["keyboard"].get("key_accuracy"),
        "mouse_button_accuracy": pc["mouse_button"].get("button_accuracy"),
        "pearson_x": pc["mouse_move"].get("pearson_x"),
        "pearson_y": pc["mouse_move"].get("pearson_y"),
        "scale_ratio_x": pc["mouse_move"].get("scale_ratio_x"),
        "scale_ratio_y": pc["mouse_move"].get("scale_ratio_y"),
        "strict_mouse_button_f1": strict["mouse_button"].get("f1"),
        "strict_no_button_fpr": strict["mouse_button"].get("no_button_false_positive_rate"),
    }


@dataclass(frozen=True)
class RetrievalPrediction:
    tokens: tuple[str, ...]
    support: int
    count: int
    confidence: float
    context_name: str


class VisualActionMemory:
    def __init__(self, context_names: Sequence[str] = _CONTEXT_NAMES) -> None:
        self.context_names = tuple(context_names)
        self.tables: dict[str, defaultdict[ContextKey, Counter[Outcome]]] = {name: defaultdict(Counter) for name in self.context_names}
        self.rows = 0
        self.nonempty_rows = 0

    def observe(self, row: dict[str, Any]) -> None:
        outcome = _outcome(_tokens(row, "ground_truth_tokens"))
        self.rows += 1
        self.nonempty_rows += int(bool(outcome))
        for name in self.context_names:
            self.tables[name][_context(row, name)][outcome] += 1

    @staticmethod
    def _top(counter: Counter[Outcome]) -> tuple[Outcome, int, int, float] | None:
        total = sum(counter.values())
        if total <= 0:
            return None
        outcome, count = max(counter.items(), key=lambda item: (item[1], item[0]))
        return outcome, count, total, count / float(total)

    def predict(self, row: dict[str, Any], *, context_name: str, threshold: float, min_support: int) -> RetrievalPrediction | None:
        rec = self._top(self.tables[context_name].get(_context(row, context_name), Counter()))
        if rec is None:
            return None
        outcome, count, total, confidence = rec
        if total < int(min_support) or confidence < float(threshold):
            return None
        return RetrievalPrediction(tuple(outcome), total, count, confidence, context_name)


def train_visual_action_memory(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 100_000,
    context_names: Sequence[str] = _CONTEXT_NAMES,
) -> VisualActionMemory:
    model = VisualActionMemory(context_names=context_names)
    for row in _iter_rows(train_paths, max_rows=max_train_rows):
        model.observe(row)
    return model


def _new_accs(split_tags: Sequence[str]) -> dict[str, _PaperMetricAccumulator]:
    out = {"all": _PaperMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        out[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=False)
    return out


def _metrics(accs: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: acc.metrics() for name, acc in sorted(accs.items())}


def _policy_specs(context_names: Sequence[str], thresholds: Sequence[float], min_supports: Sequence[int]) -> list[tuple[str, str, str, float | None, int | None]]:
    specs: list[tuple[str, str, str, float | None, int | None]] = [("base_all", "base", "", None, None)]
    for context_name in context_names:
        for min_support in min_supports:
            for threshold in thresholds:
                suffix = f"{context_name}_th{float(threshold):g}_s{int(min_support)}"
                specs.extend(
                    [
                        (f"retrieval_replace_all_{suffix}", "replace_all", context_name, float(threshold), int(min_support)),
                        (f"retrieval_categorical_base_motion_{suffix}", "categorical_base_motion", context_name, float(threshold), int(min_support)),
                        (f"retrieval_union_categorical_base_motion_{suffix}", "union_categorical_base_motion", context_name, float(threshold), int(min_support)),
                        (f"retrieval_motion_base_categorical_{suffix}", "motion_base_categorical", context_name, float(threshold), int(min_support)),
                    ]
                )
    return specs


def build_visual_action_retrieval_diagnostic(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    base_prediction_paths: Sequence[str | Path] | str | Path,
    output_prediction_path: str | Path | None = None,
    max_train_rows: int | None = 100_000,
    max_target_rows: int | None = 50_000,
    context_names: Sequence[str] = _CONTEXT_NAMES,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    min_supports: Sequence[int] = _DEFAULT_MIN_SUPPORTS,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    model = train_visual_action_memory(train_paths=train_paths, max_train_rows=max_train_rows, context_names=context_names)
    specs = _policy_specs(context_names, thresholds, min_supports)
    accs = {name: _new_accs(split_tags) for name, *_ in specs}
    usage = {name: {"applied": 0, "empty_predictions": 0} for name, mode, *_ in specs if mode != "base"}
    alignment = {"sequence_id_mismatches": 0, "missing_base_prediction_rows": 0, "examples": []}
    rows = 0
    pred_handle = None
    if output_prediction_path is not None:
        out = Path(output_prediction_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = out.open("w", encoding="utf-8")
    base_iter = iter(_iter_rows(base_prediction_paths, max_rows=max_target_rows))
    try:
        for row in _iter_rows(target_paths, max_rows=max_target_rows):
            try:
                base = next(base_iter)
            except StopIteration:
                base = {}
                alignment["missing_base_prediction_rows"] += 1
            rows += 1
            if base.get("sequence_id") is not None and row.get("sequence_id") is not None and base.get("sequence_id") != row.get("sequence_id"):
                alignment["sequence_id_mismatches"] += 1
                if len(alignment["examples"]) < 20:
                    alignment["examples"].append({"prediction_sequence_id": base.get("sequence_id"), "target_sequence_id": row.get("sequence_id")})
            base_tokens = _tokens(base, "predicted_tokens")
            base_motion = _motion_tokens(base_tokens)
            base_categorical = _categorical_tokens(base_tokens)
            gt = _tokens(row, "ground_truth_tokens")
            tags = set(_split_tags(row))
            export_tokens: list[str] | None = None
            for name, mode, context_name, threshold, min_support in specs:
                if mode == "base":
                    pred_tokens = base_tokens
                else:
                    assert threshold is not None and min_support is not None
                    pred = model.predict(row, context_name=context_name, threshold=threshold, min_support=min_support)
                    if pred is None:
                        pred_tokens = base_tokens
                    else:
                        retrieved = list(pred.tokens)
                        usage[name]["applied"] += 1
                        usage[name]["empty_predictions"] += int(not retrieved)
                        if mode == "replace_all":
                            pred_tokens = retrieved
                        elif mode == "categorical_base_motion":
                            pred_tokens = base_motion + _categorical_tokens(retrieved)
                        elif mode == "union_categorical_base_motion":
                            pred_tokens = base_motion + _merge_max_counts(base_categorical, _categorical_tokens(retrieved))
                        elif mode == "motion_base_categorical":
                            pred_tokens = _motion_tokens(retrieved) + base_categorical
                        else:  # pragma: no cover
                            raise KeyError(mode)
                accs[name]["all"].update(pred_tokens, gt)
                for tag in split_tags:
                    if tag in tags:
                        accs[name][f"eval_split:{tag}"].update(pred_tokens, gt)
                if export_tokens is None and mode != "base":
                    export_tokens = pred_tokens
            if pred_handle is not None:
                pred_handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": export_tokens or base_tokens}, separators=(",", ":")) + "\n")
    finally:
        if pred_handle is not None:
            pred_handle.close()
    policy_payloads: dict[str, Any] = {}
    for name, group_accs in accs.items():
        groups = _metrics(group_accs)
        payload: dict[str, Any] = {"groups": groups, "summary": _summary(groups["all"])}
        if name in usage:
            payload["usage"] = usage[name]
        policy_payloads[name] = payload
    ranked = sorted(
        ({"policy": name, **payload["summary"]} for name, payload in policy_payloads.items()),
        key=lambda item: (
            item.get("keyboard_accuracy") if item.get("keyboard_accuracy") is not None else -1.0,
            item.get("mouse_button_accuracy") if item.get("mouse_button_accuracy") is not None else -1.0,
            item.get("pearson_x") if item.get("pearson_x") is not None else -1.0,
        ),
        reverse=True,
    )
    return {
        "schema": "g005_visual_action_retrieval_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": model.rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "context_names": list(context_names),
        "context_count": {name: len(model.tables[name]) for name in model.context_names},
        "train_nonempty_rows": model.nonempty_rows,
        "thresholds": [float(value) for value in thresholds],
        "min_supports": [int(value) for value in min_supports],
        "policy_count": len(specs),
        "alignment": alignment,
        "policies": policy_payloads,
        "ranked_policies": ranked,
        "output_prediction_path": str(output_prediction_path) if output_prediction_path is not None else None,
        "claim_boundary": "CPU prefix diagnostic only. This approximate visual/state action-memory branch retrieves train-token multisets from causal frame/state contexts and composes with an existing base stream; it is not full-corpus G005 completion evidence.",
    }


def write_visual_action_retrieval_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_visual_action_retrieval_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
