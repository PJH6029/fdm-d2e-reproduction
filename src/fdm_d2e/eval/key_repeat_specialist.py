from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_KEY_PREFIX = "KEY_"
_BUTTON_PREFIXES = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path | None) -> list[Path]:
    if patterns is None:
        return []
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


def _iter_rows(paths: Sequence[str | Path] | str | Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
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


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(token) for token in value] if isinstance(value, list) else []


def _recording_key(row: dict[str, Any]) -> str:
    for key in ("source_recording_key", "recording_id", "source_recording_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    seq = row.get("sequence_id")
    if isinstance(seq, str) and "#" in seq:
        return seq.rsplit("#", 1)[0]
    return str(seq) if seq is not None else ""


def _split_tags(row: dict[str, Any]) -> list[str]:
    for key in ("eval_split_tags", "split_tags"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    for key in ("eval_split", "split", "split_name"):
        value = row.get(key)
        if isinstance(value, str):
            return [value]
    return []


def _prior_key_holds(row: dict[str, Any]) -> dict[str, int]:
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


def _prior_since_key_transition(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("prior_since_key_transition_bins") or 0))
    except (TypeError, ValueError):
        return 0


def _key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith(_KEY_PREFIX)]


def _non_key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if not str(token).startswith(_KEY_PREFIX)]


def _dedupe(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        text = str(token)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _age_bucket(age: int | None, cap: int = 64) -> int:
    if age is None:
        return cap + 1
    return min(max(0, int(age)), cap)


def _hold_bucket(hold: int, cap: int = 256) -> int:
    return min(max(0, int(hold)), cap)


def _context_keys(code: str, hold: int, since: int, last_press_age: int | None) -> list[tuple[str, tuple[Any, ...]]]:
    age = _age_bucket(last_press_age)
    h = _hold_bucket(hold)
    s = min(max(0, int(since)), 64)
    return [
        ("code_age_hold_since", (code, age, h, s)),
        ("code_age_hold_bucket_since", (code, age, min(h // 4, 64), s)),
        ("code_age_hold_mod10_since", (code, age, min(h // 10, 32), h % 10, s)),
        ("code_age_since", (code, age, s)),
        ("code_age", (code, age)),
        ("age_hold_since", (age, h, s)),
        ("age_since", (age, s)),
        ("code_hold_since", (code, h, s)),
        ("code_hold_mod10_since", (code, min(h // 10, 32), h % 10, s)),
    ]


class _KeyStateTracker:
    def __init__(self) -> None:
        self.holds: dict[str, dict[str, int]] = defaultdict(dict)
        self.last_press_age: dict[str, dict[str, int | None]] = defaultdict(dict)
        self.seeded: set[str] = set()

    def seed_from_row(self, row: dict[str, Any]) -> None:
        rec = _recording_key(row)
        if rec in self.seeded:
            return
        holds = _prior_key_holds(row)
        since = _prior_since_key_transition(row)
        self.holds[rec] = dict(holds)
        self.last_press_age[rec] = {code: min(since, hold) for code, hold in holds.items()}
        self.seeded.add(rec)

    def force_prior_from_row(self, row: dict[str, Any]) -> None:
        rec = _recording_key(row)
        current = _prior_key_holds(row)
        ages = self.last_press_age[rec]
        since = _prior_since_key_transition(row)
        for code, hold in current.items():
            ages.setdefault(code, min(since, hold))
        for code in list(ages):
            if code not in current:
                del ages[code]
        self.holds[rec] = dict(current)
        self.seeded.add(rec)

    def held_contexts(self, row: dict[str, Any], *, state_mode: str) -> list[tuple[str, int, int, int | None]]:
        rec = _recording_key(row)
        if state_mode == "prior_state_forced":
            self.force_prior_from_row(row)
        else:
            self.seed_from_row(row)
        since = _prior_since_key_transition(row)
        return [(code, hold, since, self.last_press_age[rec].get(code)) for code, hold in self.holds[rec].items()]

    def observe(self, row: dict[str, Any], key_tokens: Sequence[str]) -> None:
        rec = _recording_key(row)
        self.seed_from_row(row)
        holds = dict(self.holds[rec])
        ages = dict(self.last_press_age[rec])
        for code in list(ages):
            ages[code] = None if ages[code] is None else int(ages[code] or 0) + 1
        for code in list(holds):
            holds[code] = int(holds[code]) + 1
        for token in key_tokens:
            if token.startswith("KEY_PRESS_"):
                code = token.removeprefix("KEY_PRESS_")
                holds.setdefault(code, 1)
                ages[code] = 0
            elif token.startswith("KEY_RELEASE_"):
                code = token.removeprefix("KEY_RELEASE_")
                holds.pop(code, None)
                ages.pop(code, None)
        self.holds[rec] = holds
        self.last_press_age[rec] = ages


class KeyRepeatSpecialist:
    def __init__(self, *, min_support: int = 5) -> None:
        self.min_support = int(min_support)
        self.press_counts: dict[str, defaultdict[tuple[Any, ...], list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.release_counts: dict[str, defaultdict[tuple[Any, ...], list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.train_rows = 0

    def observe_train_row(self, row: dict[str, Any], tracker: _KeyStateTracker) -> None:
        gt = set(_tokens(row, "ground_truth_tokens"))
        for code, hold, since, age in tracker.held_contexts(row, state_mode="prior_state_forced"):
            press = int(f"KEY_PRESS_{code}" in gt)
            release = int(f"KEY_RELEASE_{code}" in gt)
            for name, context in _context_keys(code, hold, since, age):
                p = self.press_counts[name][context]
                p[0] += 1
                p[1] += press
                r = self.release_counts[name][context]
                r[0] += 1
                r[1] += release
        tracker.observe(row, _key_tokens(_tokens(row, "ground_truth_tokens")))
        self.train_rows += 1

    @staticmethod
    def _prob(counts: dict[str, defaultdict[tuple[Any, ...], list[int]]], contexts: list[tuple[str, tuple[Any, ...]]], min_support: int) -> tuple[float, int, str | None]:
        for name, context in contexts:
            n, pos = counts.get(name, {}).get(context, [0, 0])
            if n >= min_support:
                return (float(pos) / float(n), int(n), name)
        return (0.0, 0, None)

    def predict_keys(
        self,
        held_contexts: Sequence[tuple[str, int, int, int | None]],
        *,
        press_threshold: float,
        release_threshold: float,
    ) -> list[str]:
        out: list[str] = []
        for code, hold, since, age in held_contexts:
            contexts = _context_keys(code, hold, since, age)
            press_prob, _press_support, _press_context = self._prob(self.press_counts, contexts, self.min_support)
            release_prob, _release_support, _release_context = self._prob(self.release_counts, contexts, self.min_support)
            if press_prob >= float(press_threshold):
                out.append(f"KEY_PRESS_{code}")
            if release_prob >= float(release_threshold):
                out.append(f"KEY_RELEASE_{code}")
        return _dedupe(out)

    def context_count_summary(self) -> dict[str, Any]:
        return {
            "press": {name: len(table) for name, table in sorted(self.press_counts.items())},
            "release": {name: len(table) for name, table in sorted(self.release_counts.items())},
        }


def train_key_repeat_specialist(*, train_paths: Sequence[str | Path], max_train_rows: int | None = None, min_support: int = 5) -> KeyRepeatSpecialist:
    model = KeyRepeatSpecialist(min_support=min_support)
    tracker = _KeyStateTracker()
    for row in _iter_rows(train_paths, max_rows=max_train_rows):
        model.observe_train_row(row, tracker)
    return model


def _iter_target_base_pairs(
    target_paths: Sequence[str | Path],
    base_prediction_paths: Sequence[str | Path] | None,
    *,
    max_rows: int | None,
) -> Iterable[tuple[dict[str, Any], list[str]]]:
    base_iter = iter(_iter_rows(base_prediction_paths, max_rows=max_rows)) if base_prediction_paths else None
    for row in _iter_rows(target_paths, max_rows=max_rows):
        base_tokens: list[str] = []
        if base_iter is not None:
            try:
                pred = next(base_iter)
            except StopIteration:
                pred = {}
            base_tokens = _tokens(pred, "predicted_tokens")
        yield row, base_tokens


def _new_accs(split_tags: Sequence[str]) -> dict[str, _PaperMetricAccumulator]:
    accs = {"all": _PaperMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        accs[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=False)
    return accs


def _metrics(accs: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: acc.metrics() for name, acc in sorted(accs.items())}


def _score_group(metrics: dict[str, Any]) -> dict[str, Any]:
    pc = metrics["paper_compatible"]
    strict = metrics["strict_local"]
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


def build_key_repeat_specialist_matrix(
    *,
    train_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    base_prediction_paths: Sequence[str | Path] | None = None,
    max_train_rows: int | None = 320_000,
    max_target_rows: int | None = 320_000,
    press_thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.35, 0.5),
    release_thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.35, 0.5),
    min_support: int = 5,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    model = train_key_repeat_specialist(train_paths=train_paths, max_train_rows=max_train_rows, min_support=min_support)
    policy_specs: list[tuple[str, str, float | None, float | None]] = []
    if base_prediction_paths:
        policy_specs.append(("base_all", "base_all", None, None))
    for state_mode in ("prior_state_forced", "closed_loop"):
        for press_th in press_thresholds:
            for release_th in release_thresholds:
                label = f"{state_mode}_press{press_th:g}_release{release_th:g}_replace_base_keys"
                policy_specs.append((label, state_mode, float(press_th), float(release_th)))
                if base_prediction_paths:
                    label = f"{state_mode}_press{press_th:g}_release{release_th:g}_union_base_keys"
                    policy_specs.append((label, state_mode, float(press_th), float(release_th)))
    accs = {label: _new_accs(split_tags) for label, _mode, _p, _r in policy_specs}
    trackers = {label: _KeyStateTracker() for label, _mode, _p, _r in policy_specs if _mode != "base_all"}
    rows = 0
    for row, base_tokens in _iter_target_base_pairs(target_paths, base_prediction_paths, max_rows=max_target_rows):
        rows += 1
        gt = _tokens(row, "ground_truth_tokens")
        tags = set(_split_tags(row))
        for label, state_mode, press_th, release_th in policy_specs:
            if state_mode == "base_all":
                predicted = base_tokens
            else:
                tracker = trackers[label]
                held = tracker.held_contexts(row, state_mode=state_mode)
                specialist_keys = model.predict_keys(held, press_threshold=float(press_th), release_threshold=float(release_th))
                if label.endswith("union_base_keys"):
                    key_tokens = _dedupe(_key_tokens(base_tokens) + specialist_keys)
                else:
                    key_tokens = specialist_keys
                predicted = _non_key_tokens(base_tokens) + key_tokens
                tracker.observe(row, key_tokens)
            accs[label]["all"].update(predicted, gt)
            for tag in split_tags:
                if tag in tags:
                    accs[label][f"eval_split:{tag}"].update(predicted, gt)
    policies: dict[str, Any] = {}
    for label, group_accs in accs.items():
        group_metrics = _metrics(group_accs)
        policies[label] = {
            "groups": group_metrics,
            "summary": _score_group(group_metrics["all"]),
        }
    ranked = sorted(
        (
            {
                "policy": label,
                **payload["summary"],
            }
            for label, payload in policies.items()
        ),
        key=lambda row: (
            row.get("keyboard_accuracy") or -1.0,
            row.get("mouse_button_accuracy") or -1.0,
            row.get("pearson_x") or -1.0,
            row.get("pearson_y") or -1.0,
        ),
        reverse=True,
    )
    return {
        "schema": "g005_key_repeat_specialist_matrix.v1",
        "status": "pass",
        "rows": rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "min_support": int(min_support),
        "press_thresholds": [float(v) for v in press_thresholds],
        "release_thresholds": [float(v) for v in release_thresholds],
        "context_counts": model.context_count_summary(),
        "train_rows": model.train_rows,
        "target_paths": [str(p) for p in target_paths],
        "base_prediction_paths": [str(p) for p in (base_prediction_paths or [])],
        "policies": policies,
        "ranked_policies": ranked,
        "claim_boundary": "Prefix diagnostic. prior_state_forced uses target prior state fields and is not closed-loop evidence; matrix ranking is target-prefix diagnostic, not G005 completion evidence.",
    }


def write_key_repeat_specialist_matrix(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_key_repeat_specialist_matrix(**kwargs)
    write_json(output_path, payload)
    return payload
