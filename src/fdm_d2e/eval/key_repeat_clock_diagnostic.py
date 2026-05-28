from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_KEY_PREFIXES = ("KEY_PRESS_", "KEY_RELEASE_", "KEY_DOWN_")
_CAT_PREFIXES = ("KEY_", "MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")
_MOTION_PREFIXES = ("MOUSE_DX_", "MOUSE_DY_")
_DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.35, 0.5)
_DEFAULT_MIN_SUPPORTS = (1, 3, 8)


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path) -> list[Path]:
    items = [patterns] if isinstance(patterns, (str, Path)) else list(patterns)
    out: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if matches:
            out.extend(Path(match) for match in matches)
            continue
        path = Path(item)
        if path.exists():
            out.append(path)
    return out


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
                except Exception as exc:  # pragma: no cover - path-specific context
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                rows += 1


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(item) for item in value] if isinstance(value, list) else []


def _split_tags(row: dict[str, Any]) -> list[str]:
    for key in ("eval_split_tags", "split_tags"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    return []


def _recording_key(row: dict[str, Any]) -> str:
    for key in ("source_recording_key", "recording_id", "source_recording_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    seq = str(row.get("sequence_id") or "")
    return seq.rsplit("#", 1)[0] if "#" in seq else seq


def _game(row: dict[str, Any]) -> str:
    value = row.get("game")
    if isinstance(value, str) and value:
        return value
    rec = _recording_key(row)
    if ":" in rec:
        rec = rec.split(":", 1)[1]
    return rec.split("/", 1)[0] if "/" in rec else (rec or "unknown")


def _sequence_index(row: dict[str, Any]) -> int:
    seq = str(row.get("sequence_id") or "")
    match = re.search(r"#(\d+)$", seq)
    if match:
        return int(match.group(1))
    try:
        return int(row.get("timestamp_ns") or 0) // 50_000_000
    except (TypeError, ValueError):
        return 0


def _key_code(token: str) -> tuple[str, str] | None:
    if token.startswith("KEY_PRESS_"):
        return "press", token.removeprefix("KEY_PRESS_")
    if token.startswith("KEY_RELEASE_"):
        return "release", token.removeprefix("KEY_RELEASE_")
    if token.startswith("KEY_DOWN_"):
        return "down", token.removeprefix("KEY_DOWN_")
    return None


def _key_codes_from_tokens(tokens: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        parsed = _key_code(str(token))
        if parsed is not None and parsed[1]:
            out.add(parsed[1])
    return out


def _held_keys(row: dict[str, Any]) -> dict[str, int]:
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


def _since_transition(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("prior_since_key_transition_bins") or 0))
    except (TypeError, ValueError):
        return 0


def _bucket(value: int | None) -> int:
    if value is None:
        return -1
    value = max(0, int(value))
    if value <= 12:
        return value
    if value <= 32:
        return 12 + ((value - 12) // 2)
    if value <= 96:
        return 22 + ((value - 32) // 4)
    return min(64, 38 + ((value - 96) // 16))


def _non_key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if not str(token).startswith("KEY_")]


def _key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith("KEY_")]


def _merge_counts(a: Sequence[str], b: Sequence[str]) -> list[str]:
    ca = Counter(str(t) for t in a)
    cb = Counter(str(t) for t in b)
    ordered = list(dict.fromkeys([*map(str, a), *map(str, b)]))
    out: list[str] = []
    for token in ordered:
        out.extend([token] * max(ca.get(token, 0), cb.get(token, 0)))
    return out


@dataclass
class KeyRepeatClock:
    press_age: dict[str, int] = field(default_factory=dict)
    release_age: dict[str, int] = field(default_factory=dict)

    def observe_codes(self, codes: Iterable[str]) -> None:
        for code in codes:
            self.press_age.setdefault(str(code), 999)
            self.release_age.setdefault(str(code), 999)

    def snapshot(self, code: str) -> tuple[int, int]:
        return (int(self.press_age.get(code, 999)), int(self.release_age.get(code, 999)))

    def advance(self, tokens: Sequence[str], *, candidate_codes: Iterable[str] = ()) -> None:
        codes = set(str(code) for code in candidate_codes)
        codes.update(_key_codes_from_tokens(tokens))
        codes.update(self.press_age)
        codes.update(self.release_age)
        for code in codes:
            self.press_age[code] = min(999, int(self.press_age.get(code, 999)) + 1)
            self.release_age[code] = min(999, int(self.release_age.get(code, 999)) + 1)
        for token in tokens:
            parsed = _key_code(str(token))
            if parsed is None:
                continue
            kind, code = parsed
            if kind == "press":
                self.press_age[code] = 0
            elif kind == "release":
                self.release_age[code] = 0


def _context(row: dict[str, Any], code: str, hold: int, press_age: int, release_age: int, name: str) -> tuple[Any, ...]:
    idx = _sequence_index(row)
    game = _game(row)
    h = _bucket(hold)
    p = _bucket(press_age)
    r = _bucket(release_age)
    s = _bucket(_since_transition(row))
    if name == "code_hold_age":
        return (game, code, h, p, r)
    if name == "code_hold_age_mod":
        return (game, code, h, p, r, hold % 2, hold % 3, hold % 4, hold % 8, press_age % 2, press_age % 4, press_age % 8)
    if name == "code_age_phase":
        return (game, code, p, r, idx % 2, idx % 4, idx % 8, idx % 16)
    if name == "code_hold_since_age":
        return (game, code, h, s, p, r)
    if name == "code_hold_mod_only":
        return (game, code, h, hold % 2, hold % 3, hold % 4, hold % 8, idx % 4)
    raise KeyError(name)


_CONTEXT_NAMES = ("code_hold_age_mod", "code_hold_age", "code_hold_since_age", "code_age_phase", "code_hold_mod_only")


def collect_top_key_codes(paths: Sequence[str | Path] | str | Path, *, max_rows: int | None, limit: int) -> list[str]:
    if int(limit) <= 0:
        return []
    counts: Counter[str] = Counter()
    for row in _iter_rows(paths, max_rows=max_rows):
        counts.update(_key_codes_from_tokens(_tokens(row, "ground_truth_tokens")))
    return [code for code, _count in counts.most_common(int(limit))]


class RepeatClockTables:
    def __init__(self, context_names: Sequence[str] = _CONTEXT_NAMES, top_codes: Sequence[str] = ()) -> None:
        self.context_names = tuple(context_names)
        self.top_codes = tuple(str(code) for code in top_codes)
        self.press: dict[str, defaultdict[tuple[Any, ...], list[int]]] = {name: defaultdict(lambda: [0, 0]) for name in self.context_names}
        self.release: dict[str, defaultdict[tuple[Any, ...], list[int]]] = {name: defaultdict(lambda: [0, 0]) for name in self.context_names}
        self.rows = 0
        self.examples = 0
        self.positive_press = 0
        self.positive_release = 0

    def _candidate_codes(self, row: dict[str, Any], extra_codes: Iterable[str] = ()) -> list[str]:
        codes = set(_held_keys(row))
        codes.update(self.top_codes)
        codes.update(extra_codes)
        return sorted(code for code in codes if code)

    def observe(self, row: dict[str, Any], clock: KeyRepeatClock) -> None:
        gt = _tokens(row, "ground_truth_tokens")
        gt_counts = Counter(gt)
        holds = _held_keys(row)
        codes = self._candidate_codes(row, _key_codes_from_tokens(gt))
        clock.observe_codes(codes)
        for code in codes:
            hold = holds.get(code, 0)
            press_age, release_age = clock.snapshot(code)
            press_positive = int(gt_counts.get("KEY_PRESS_" + code, 0) > 0)
            release_positive = int(gt_counts.get("KEY_RELEASE_" + code, 0) > 0)
            self.positive_press += press_positive
            self.positive_release += release_positive
            self.examples += 1
            for name in self.context_names:
                key = _context(row, code, hold, press_age, release_age, name)
                self.press[name][key][0] += 1
                self.press[name][key][1] += press_positive
                self.release[name][key][0] += 1
                self.release[name][key][1] += release_positive
        clock.advance(gt, candidate_codes=codes)
        self.rows += 1

    @staticmethod
    def _prob(table: defaultdict[tuple[Any, ...], list[int]], key: tuple[Any, ...], min_support: int) -> tuple[float, int] | None:
        total, positive = table.get(key, [0, 0])
        if total < int(min_support) or total <= 0:
            return None
        return positive / float(total), total

    def predict(
        self,
        row: dict[str, Any],
        clock: KeyRepeatClock,
        *,
        context_name: str,
        press_threshold: float,
        release_threshold: float,
        min_support: int,
    ) -> list[str]:
        out: list[str] = []
        holds = _held_keys(row)
        codes = self._candidate_codes(row, [*clock.press_age.keys(), *clock.release_age.keys()])
        clock.observe_codes(codes)
        for code in codes:
            hold = holds.get(code, 0)
            press_age, release_age = clock.snapshot(code)
            key = _context(row, code, hold, press_age, release_age, context_name)
            prec = self._prob(self.press[context_name], key, min_support)
            rrec = self._prob(self.release[context_name], key, min_support)
            if prec is not None and prec[0] >= float(press_threshold):
                out.append("KEY_PRESS_" + code)
            if rrec is not None and rrec[0] >= float(release_threshold):
                out.append("KEY_RELEASE_" + code)
        return out


def train_repeat_clock_tables(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 320_000,
    context_names: Sequence[str] = _CONTEXT_NAMES,
    candidate_key_count: int = 0,
) -> RepeatClockTables:
    top_codes = collect_top_key_codes(train_paths, max_rows=max_train_rows, limit=candidate_key_count)
    model = RepeatClockTables(context_names=context_names, top_codes=top_codes)
    clocks: dict[str, KeyRepeatClock] = defaultdict(KeyRepeatClock)
    for row in _iter_rows(train_paths, max_rows=max_train_rows):
        model.observe(row, clocks[_recording_key(row)])
    return model


def _new_accs(split_tags: Sequence[str]) -> dict[str, _PaperMetricAccumulator]:
    out = {"all": _PaperMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        out[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=False)
    return out


def _metrics(accs: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: acc.metrics() for name, acc in sorted(accs.items())}


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


def _policy_specs(
    context_names: Sequence[str],
    thresholds: Sequence[float],
    min_supports: Sequence[int],
    clock_modes: Sequence[str],
) -> list[tuple[str, str, str, float | None, int | None, str]]:
    specs: list[tuple[str, str, str, float | None, int | None, str]] = [("base_all", "base", "", None, None, "base")]
    for clock_mode in clock_modes:
        for context_name in context_names:
            for min_support in min_supports:
                for threshold in thresholds:
                    suffix = f"{clock_mode}_{context_name}_th{float(threshold):g}_s{int(min_support)}"
                    specs.append((f"clock_replace_keys_{suffix}", "replace", context_name, float(threshold), int(min_support), clock_mode))
                    specs.append((f"clock_union_base_keys_{suffix}", "union", context_name, float(threshold), int(min_support), clock_mode))
                    specs.append((f"clock_press_only_union_{suffix}", "press_union", context_name, float(threshold), int(min_support), clock_mode))
    return specs


def build_key_repeat_clock_diagnostic(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    base_prediction_paths: Sequence[str | Path] | str | Path,
    output_prediction_path: str | Path | None = None,
    max_train_rows: int | None = 320_000,
    max_target_rows: int | None = 50_000,
    context_names: Sequence[str] = _CONTEXT_NAMES,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    min_supports: Sequence[int] = _DEFAULT_MIN_SUPPORTS,
    clock_modes: Sequence[str] = ("predicted", "teacher_forced"),
    candidate_key_count: int = 0,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    model = train_repeat_clock_tables(
        train_paths=train_paths,
        max_train_rows=max_train_rows,
        context_names=context_names,
        candidate_key_count=candidate_key_count,
    )
    specs = _policy_specs(context_names, thresholds, min_supports, clock_modes)
    accs = {name: _new_accs(split_tags) for name, *_ in specs}
    usage = {name: {"predicted_key_events": 0} for name, mode, *_ in specs if mode != "base"}
    target_clocks: dict[str, dict[str, KeyRepeatClock]] = {mode: defaultdict(KeyRepeatClock) for mode in clock_modes}
    rows = 0
    alignment = {"sequence_id_mismatches": 0, "missing_base_prediction_rows": 0, "examples": []}
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
            if base.get("sequence_id") is not None and row.get("sequence_id") is not None and base.get("sequence_id") != row.get("sequence_id"):
                alignment["sequence_id_mismatches"] += 1
                if len(alignment["examples"]) < 20:
                    alignment["examples"].append({"prediction_sequence_id": base.get("sequence_id"), "target_sequence_id": row.get("sequence_id")})
            rows += 1
            gt = _tokens(row, "ground_truth_tokens")
            base_tokens = _tokens(base, "predicted_tokens")
            base_non_key = _non_key_tokens(base_tokens)
            base_keys = _key_tokens(base_tokens)
            tags = set(_split_tags(row))
            rec_key = _recording_key(row)
            export_tokens: list[str] | None = None
            predicted_updates: dict[str, list[str]] = {}
            for name, mode, context_name, threshold, min_support, clock_mode in specs:
                if mode == "base":
                    pred_tokens = base_tokens
                else:
                    assert threshold is not None and min_support is not None
                    clock = target_clocks[clock_mode][rec_key]
                    key_pred = model.predict(
                        row,
                        clock,
                        context_name=context_name,
                        press_threshold=float(threshold),
                        release_threshold=float(threshold),
                        min_support=int(min_support),
                    )
                    if mode == "press_union":
                        key_pred = [token for token in key_pred if token.startswith("KEY_PRESS_")]
                    usage[name]["predicted_key_events"] += len(key_pred)
                    if mode == "replace":
                        pred_tokens = base_non_key + key_pred
                    elif mode in {"union", "press_union"}:
                        pred_tokens = base_non_key + _merge_counts(base_keys, key_pred)
                    else:  # pragma: no cover
                        raise KeyError(mode)
                    predicted_updates[name] = key_pred
                    if export_tokens is None:
                        export_tokens = pred_tokens
                accs[name]["all"].update(pred_tokens, gt)
                for tag in split_tags:
                    if tag in tags:
                        accs[name][f"eval_split:{tag}"].update(pred_tokens, gt)
            # Advance clocks once per row after all policy predictions.
            for mode in clock_modes:
                clock = target_clocks[mode][rec_key]
                if mode == "teacher_forced":
                    clock.advance(gt, candidate_codes=model.top_codes)
                elif mode == "predicted":
                    # Use the most permissive first predicted policy for a stable closed-loop clock.
                    first_pred = next((tokens for _name, tokens in predicted_updates.items() if tokens), [])
                    clock.advance(first_pred, candidate_codes=model.top_codes)
                else:
                    raise KeyError(f"unknown clock_mode={mode}")
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
        "schema": "g005_key_repeat_clock_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": model.rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "candidate_key_count": int(candidate_key_count),
        "candidate_key_codes": list(model.top_codes),
        "context_names": list(context_names),
        "thresholds": [float(value) for value in thresholds],
        "min_supports": [int(value) for value in min_supports],
        "clock_modes": list(clock_modes),
        "model_examples": model.examples,
        "model_positive_press": model.positive_press,
        "model_positive_release": model.positive_release,
        "context_count": {name: len(model.press[name]) for name in model.context_names},
        "alignment": alignment,
        "policies": policy_payloads,
        "ranked_policies": ranked,
        "output_prediction_path": str(output_prediction_path) if output_prediction_path is not None else None,
        "claim_boundary": "CPU prefix diagnostic only. Predicted-clock policies update timing from predicted key events; teacher-forced clock is an upper bound and must not be promoted as non-leaky evidence.",
    }


def write_key_repeat_clock_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_key_repeat_clock_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
