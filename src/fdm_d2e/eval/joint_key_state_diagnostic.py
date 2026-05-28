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

try:  # pragma: no cover - exercised only when optional fast parser is present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_DEFAULT_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8)
_DEFAULT_MIN_SUPPORTS = (1, 3, 5)
_PHASE_MODS = (2, 3, 4, 6, 8, 10, 16)
_KEY_PREFIXES = ("KEY_PRESS_", "KEY_RELEASE_", "KEY_DOWN_")

Outcome = tuple[str, ...]
ContextKey = tuple[Any, ...]


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
            return [str(token) for token in value]
        if isinstance(value, str):
            return [value]
    return []


def sequence_bin_index(row: dict[str, Any]) -> int:
    seq = str(row.get("sequence_id", ""))
    match = re.search(r"#(\d+)$", seq)
    if match:
        return int(match.group(1))
    try:
        return int(row.get("timestamp_ns") or 0) // 50_000_000
    except (TypeError, ValueError):
        return 0


def _recording_game(row: dict[str, Any]) -> str:
    rec = str(row.get("recording_id") or row.get("source_recording_key") or "")
    if ":" in rec:
        rec = rec.split(":", 1)[1]
    if "/" in rec:
        return rec.split("/", 1)[0]
    return rec or "unknown"


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


def _count_tuple(tokens: Sequence[str], *, prefixes: tuple[str, ...] = ("KEY_",), limit: int = 16) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(token) for token in tokens if str(token).startswith(prefixes))
    return tuple(sorted(counts.items())[:limit])


def _key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith("KEY_")]


def _non_key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if not str(token).startswith("KEY_")]


def _key_codes_from_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    codes: set[str] = set()
    for token in tokens:
        text = str(token)
        for prefix in _KEY_PREFIXES:
            if text.startswith(prefix):
                code = text[len(prefix) :]
                if code:
                    codes.add(code)
                break
    return tuple(sorted(codes))


def _hold_signature(row: dict[str, Any], *, style: str, limit: int = 16) -> tuple[Any, ...]:
    holds = _holds(row)
    items: list[Any] = []
    for code, hold in sorted(holds.items())[:limit]:
        if style == "codes":
            items.append(code)
        elif style == "bucket":
            items.append((code, _bucket(hold)))
        elif style == "mod":
            items.append((code, _bucket(hold), hold % 2, hold % 3, hold % 4, hold % 6, hold % 8, hold % 10))
        else:
            items.append((code, min(255, hold)))
    return tuple(items)


def _phase_tuple(row: dict[str, Any], mods: Sequence[int] = _PHASE_MODS) -> tuple[int, ...]:
    idx = sequence_bin_index(row)
    return tuple(idx % int(mod) for mod in mods)


def _outcome(tokens: Sequence[str]) -> Outcome:
    # Keep duplicate key events because D2E's public metrics compare token counts.
    return tuple(sorted(_key_tokens(tokens)))


def _tokens_from_outcome(outcome: Outcome) -> list[str]:
    return list(outcome)


def _merge_max_counts(base_tokens: Sequence[str], specialist_tokens: Sequence[str]) -> list[str]:
    base_counts = Counter(str(token) for token in base_tokens)
    specialist_counts = Counter(str(token) for token in specialist_tokens)
    ordered = list(dict.fromkeys([str(token) for token in list(base_tokens) + list(specialist_tokens)]))
    out: list[str] = []
    for token in ordered:
        out.extend([token] * max(base_counts.get(token, 0), specialist_counts.get(token, 0)))
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


def _context(row: dict[str, Any], name: str) -> ContextKey:
    prev_keys = _count_tuple(_tokens(row, "previous_event_tokens"), limit=12)
    prior_keys = _key_codes_from_tokens(_tokens(row, "prior_action_tokens"))[:16]
    since_bucket = _bucket(_since(row))
    held_count = min(len(_holds(row)), 16)
    phases = _phase_tuple(row)
    game = _recording_game(row)
    if name == "held_codes_since_prev_phase":
        return (_hold_signature(row, style="codes"), since_bucket, prev_keys, phases)
    if name == "held_bucket_since_prev":
        return (_hold_signature(row, style="bucket"), since_bucket, prev_keys)
    if name == "held_mod_since_phase":
        return (_hold_signature(row, style="mod"), since_bucket, phases)
    if name == "game_held_mod_since_phase":
        return (game, _hold_signature(row, style="mod"), since_bucket, phases)
    if name == "held_count_prev_phase":
        return (held_count, since_bucket, prev_keys, phases)
    if name == "prior_action_phase":
        return (prior_keys, since_bucket, phases)
    if name == "held_codes_only":
        return (_hold_signature(row, style="codes"),)
    if name == "held_bucket_only":
        return (_hold_signature(row, style="bucket"),)
    raise KeyError(f"unknown joint-key context: {name}")


_CONTEXT_NAMES = (
    "game_held_mod_since_phase",
    "held_mod_since_phase",
    "held_codes_since_prev_phase",
    "held_bucket_since_prev",
    "held_count_prev_phase",
    "prior_action_phase",
    "held_bucket_only",
    "held_codes_only",
)

_CONTEXT_CHAINS: dict[str, tuple[str, ...]] = {
    "specific_to_global": (
        "game_held_mod_since_phase",
        "held_mod_since_phase",
        "held_codes_since_prev_phase",
        "held_bucket_since_prev",
        "held_bucket_only",
        "held_codes_only",
    ),
    "duration_to_codes": ("held_mod_since_phase", "held_bucket_since_prev", "held_bucket_only", "held_codes_only"),
    "prior_to_held": ("prior_action_phase", "held_codes_since_prev_phase", "held_codes_only"),
}


@dataclass(frozen=True)
class JointPrediction:
    tokens: tuple[str, ...]
    support: int
    count: int
    probability: float
    source_context: str
    source: str


@dataclass(frozen=True)
class _OutcomeStats:
    total: int
    top: Outcome
    top_count: int
    top_probability: float
    nonempty: Outcome
    nonempty_count: int
    nonempty_probability: float


class JointKeyStateTable:
    def __init__(self, context_names: Sequence[str] = _CONTEXT_NAMES) -> None:
        self.context_names = tuple(context_names)
        self.tables: dict[str, defaultdict[ContextKey, Counter[Outcome]]] = {
            name: defaultdict(Counter) for name in self.context_names
        }
        self.rows = 0
        self.nonempty_rows = 0
        self.repeated_key_rows = 0
        self.repeated_key_token_cases = 0

    def observe(self, row: dict[str, Any]) -> None:
        outcome = _outcome(_tokens(row, "ground_truth_tokens"))
        counts = Counter(outcome)
        self.rows += 1
        self.nonempty_rows += int(bool(outcome))
        repeated_cases = sum(1 for value in counts.values() if value > 1)
        self.repeated_key_token_cases += repeated_cases
        self.repeated_key_rows += int(repeated_cases > 0)
        for name in self.context_names:
            self.tables[name][_context(row, name)][outcome] += 1

    @staticmethod
    def _stats(counter: Counter[Outcome]) -> _OutcomeStats | None:
        total = sum(counter.values())
        if total <= 0:
            return None
        top, top_count = max(counter.items(), key=lambda item: (item[1], item[0]))
        nonempty_items = [(outcome, count) for outcome, count in counter.items() if outcome]
        if nonempty_items:
            nonempty, nonempty_count = max(nonempty_items, key=lambda item: (item[1], item[0]))
        else:
            nonempty, nonempty_count = (), 0
        return _OutcomeStats(
            total=total,
            top=top,
            top_count=top_count,
            top_probability=top_count / float(total),
            nonempty=nonempty,
            nonempty_count=nonempty_count,
            nonempty_probability=nonempty_count / float(total),
        )

    def predict_context(
        self,
        row: dict[str, Any],
        *,
        context_name: str,
        threshold: float,
        min_support: int,
        source: str,
    ) -> JointPrediction | None:
        stats = self._stats(self.tables[context_name].get(_context(row, context_name), Counter()))
        if stats is None or stats.total < int(min_support):
            return None
        if source == "top":
            if stats.top_probability < float(threshold):
                return None
            return JointPrediction(
                tokens=tuple(stats.top),
                support=stats.total,
                count=stats.top_count,
                probability=stats.top_probability,
                source_context=context_name,
                source=source,
            )
        if source == "nonempty":
            if not stats.nonempty or stats.nonempty_probability < float(threshold):
                return None
            return JointPrediction(
                tokens=tuple(stats.nonempty),
                support=stats.total,
                count=stats.nonempty_count,
                probability=stats.nonempty_probability,
                source_context=context_name,
                source=source,
            )
        raise KeyError(f"unknown joint prediction source: {source}")

    def predict_chain(
        self,
        row: dict[str, Any],
        *,
        chain_name: str,
        threshold: float,
        min_support: int,
        source: str,
    ) -> JointPrediction | None:
        for context_name in _CONTEXT_CHAINS[chain_name]:
            pred = self.predict_context(
                row,
                context_name=context_name,
                threshold=threshold,
                min_support=min_support,
                source=source,
            )
            if pred is not None:
                return pred
        return None


PolicySpec = tuple[str, str, str, str, float | None, int | None]


def train_joint_key_state_table(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 320_000,
    context_names: Sequence[str] = _CONTEXT_NAMES,
) -> JointKeyStateTable:
    model = JointKeyStateTable(context_names=context_names)
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


def _build_policy_specs(
    *,
    thresholds: Sequence[float],
    min_supports: Sequence[int],
    lookup_names: Sequence[str] | None = None,
) -> list[PolicySpec]:
    specs: list[PolicySpec] = [("base_all", "base", "", "", None, None)]
    selected_lookup_names = list(lookup_names) if lookup_names is not None else list(_CONTEXT_NAMES) + [f"chain:{name}" for name in _CONTEXT_CHAINS]
    for lookup_name in selected_lookup_names:
        for source in ("top", "nonempty"):
            for min_support in min_supports:
                for threshold in thresholds:
                    th = float(threshold)
                    sup = int(min_support)
                    source_label = f"{source}_th{th:g}_s{sup}"
                    specs.append((f"joint_union_{lookup_name}_{source_label}", "union", lookup_name, source, th, sup))
                    specs.append((f"joint_replace_{lookup_name}_{source_label}", "replace", lookup_name, source, th, sup))
                    if source == "nonempty":
                        specs.append((f"joint_replace_nonempty_{lookup_name}_{source_label}", "replace_nonempty", lookup_name, source, th, sup))
    # Preserve insertion order while de-duplicating labels.
    seen: set[str] = set()
    unique: list[PolicySpec] = []
    for spec in specs:
        if spec[0] not in seen:
            unique.append(spec)
            seen.add(spec[0])
    return unique


def _predict_for_spec(model: JointKeyStateTable, row: dict[str, Any], spec: PolicySpec) -> JointPrediction | None:
    _label, mode, lookup_name, source, threshold, min_support = spec
    if mode == "base":
        return None
    assert threshold is not None and min_support is not None
    if lookup_name.startswith("chain:"):
        return model.predict_chain(
            row,
            chain_name=lookup_name.split(":", 1)[1],
            threshold=threshold,
            min_support=min_support,
            source=source,
        )
    return model.predict_context(
        row,
        context_name=lookup_name,
        threshold=threshold,
        min_support=min_support,
        source=source,
    )


def build_joint_key_state_diagnostic(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    base_prediction_paths: Sequence[str | Path] | str | Path,
    output_prediction_path: str | Path | None = None,
    max_train_rows: int | None = 320_000,
    max_target_rows: int | None = 50_000,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    min_supports: Sequence[int] = _DEFAULT_MIN_SUPPORTS,
    lookup_names: Sequence[str] | None = None,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    selected_lookup_names = list(lookup_names) if lookup_names is not None else list(_CONTEXT_NAMES) + [f"chain:{name}" for name in _CONTEXT_CHAINS]
    required_contexts: set[str] = set()
    for lookup_name in selected_lookup_names:
        if lookup_name.startswith("chain:"):
            required_contexts.update(_CONTEXT_CHAINS[lookup_name.split(":", 1)[1]])
        else:
            if lookup_name not in _CONTEXT_NAMES:
                raise KeyError(f"unknown joint-key lookup: {lookup_name}")
            required_contexts.add(lookup_name)
    model_contexts = tuple(name for name in _CONTEXT_NAMES if name in required_contexts)
    model = train_joint_key_state_table(train_paths=train_paths, max_train_rows=max_train_rows, context_names=model_contexts)
    policies = _build_policy_specs(thresholds=thresholds, min_supports=min_supports, lookup_names=selected_lookup_names)
    accs = {name: _new_accs(split_tags) for name, _mode, _lookup, _source, _th, _support in policies}
    usage: dict[str, dict[str, int]] = {name: {"applied": 0, "empty_predictions": 0} for name, *_ in policies if name != "base_all"}
    alignment = {"sequence_id_mismatches": 0, "missing_base_prediction_rows": 0, "examples": []}
    rows = 0
    pred_handle = None
    export_policy = next((spec[0] for spec in policies if spec[0] != "base_all"), None)
    if output_prediction_path is not None:
        out_path = Path(output_prediction_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = out_path.open("w", encoding="utf-8")
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
            base_non_key = _non_key_tokens(base_tokens)
            base_key = _key_tokens(base_tokens)
            tags = set(_split_tags(row))
            gt = _tokens(row, "ground_truth_tokens")
            export_tokens: list[str] | None = None
            for spec in policies:
                label, mode, _lookup_name, _source, _threshold, _min_support = spec
                if mode == "base":
                    pred_tokens = base_tokens
                else:
                    specialist = _predict_for_spec(model, row, spec)
                    if specialist is None:
                        pred_tokens = base_tokens
                    else:
                        predicted_key = _tokens_from_outcome(specialist.tokens)
                        usage[label]["applied"] += 1
                        usage[label]["empty_predictions"] += int(not predicted_key)
                        if mode == "union":
                            pred_tokens = base_non_key + _merge_max_counts(base_key, predicted_key)
                        elif mode == "replace":
                            pred_tokens = base_non_key + predicted_key
                        elif mode == "replace_nonempty":
                            pred_tokens = base_non_key + predicted_key if predicted_key else base_tokens
                        else:  # pragma: no cover
                            raise KeyError(mode)
                accs[label]["all"].update(pred_tokens, gt)
                for tag in split_tags:
                    if tag in tags:
                        accs[label][f"eval_split:{tag}"].update(pred_tokens, gt)
                if export_policy == label:
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
        "schema": "g005_joint_key_state_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": model.rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "thresholds": [float(value) for value in thresholds],
        "min_supports": [int(value) for value in min_supports],
        "lookup_names": selected_lookup_names,
        "context_names": list(model.context_names),
        "context_chains": {name: list(chain) for name, chain in _CONTEXT_CHAINS.items()},
        "context_count": {name: len(table) for name, table in model.tables.items()},
        "train_nonempty_key_rows": model.nonempty_rows,
        "train_repeated_key_rows": model.repeated_key_rows,
        "train_repeated_key_token_cases": model.repeated_key_token_cases,
        "policy_count": len(policies),
        "alignment": alignment,
        "policies": policy_payloads,
        "ranked_policies": ranked,
        "output_prediction_path": str(output_prediction_path) if output_prediction_path is not None else None,
        "claim_boundary": "CPU prefix diagnostic only. This joint table predicts full key-event multisets from causal sequence/held-state contexts and composes with an existing base stream; it is not full-corpus G005 completion evidence.",
    }


def write_joint_key_state_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_joint_key_state_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
