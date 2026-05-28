from __future__ import annotations

import glob
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_BUTTON_PREFIXES = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path) -> list[Path]:
    values = [patterns] if isinstance(patterns, (str, Path)) else list(patterns)
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(str(value)))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(value)
        if path.exists():
            paths.append(path)
    return paths


def _iter_rows(patterns: Sequence[str | Path] | str | Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in _expand_paths(patterns):
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and emitted >= max_rows:
                    return
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                emitted += 1


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(token) for token in value] if isinstance(value, list) else []


def _button_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(str(token) for token in tokens if str(token).startswith(_BUTTON_PREFIXES)))


def _row_id(row: dict[str, Any]) -> str | None:
    value = row.get("sequence_id")
    return str(value) if value is not None else None


def _button_metrics(pred_buttons: Sequence[tuple[str, ...]], gt_buttons: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    rows = min(len(pred_buttons), len(gt_buttons))
    predicted = gt_positive = exact_tp = any_overlap = no_button_fp = wrong_positive = 0
    pair_counter: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    for idx in range(rows):
        pred = tuple(pred_buttons[idx])
        gt = tuple(gt_buttons[idx])
        pred_set = set(pred)
        gt_set = set(gt)
        pred_counts.update(pred)
        gt_counts.update(gt)
        if pred:
            predicted += 1
        if gt:
            gt_positive += 1
        if pred and not gt:
            no_button_fp += 1
        if pred and gt:
            pair_counter[f"{'+'.join(pred)} => {'+'.join(gt)}"] += 1
            if pred == gt:
                exact_tp += 1
            elif pred_set & gt_set:
                any_overlap += 1
            else:
                wrong_positive += 1
        elif pred == gt and pred:
            exact_tp += 1
    false_positive = sum(1 for idx in range(rows) if pred_buttons[idx] and pred_buttons[idx] != gt_buttons[idx])
    false_negative = sum(1 for idx in range(rows) if gt_buttons[idx] and pred_buttons[idx] != gt_buttons[idx])
    denom = (2 * exact_tp) + false_positive + false_negative
    return {
        "rows": rows,
        "predicted_examples": predicted,
        "ground_truth_examples": gt_positive,
        "exact_true_positive_examples": exact_tp,
        "any_token_overlap_wrong_examples": any_overlap,
        "both_positive_no_overlap_examples": wrong_positive,
        "false_positive_examples": false_positive,
        "false_negative_examples": false_negative,
        "no_button_false_positive_examples": no_button_fp,
        "no_button_false_positive_rate": no_button_fp / max(1, rows - gt_positive),
        "precision": exact_tp / (exact_tp + false_positive) if (exact_tp + false_positive) else None,
        "recall": exact_tp / (exact_tp + false_negative) if (exact_tp + false_negative) else None,
        "f1": (2 * exact_tp) / denom if denom else 0.0,
        "semantic_any_button_overlap_examples": exact_tp + any_overlap,
        "semantic_any_button_overlap_rate_on_gt": (exact_tp + any_overlap) / gt_positive if gt_positive else None,
        "top_predicted_button_tokens": pred_counts.most_common(20),
        "top_ground_truth_button_tokens": gt_counts.most_common(20),
        "top_predicted_vs_ground_truth_pairs": pair_counter.most_common(50),
    }


def _offset_sweep(pred_buttons: list[tuple[str, ...]], gt_buttons: list[tuple[str, ...]], offsets: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        aligned_pred: list[tuple[str, ...]] = []
        aligned_gt: list[tuple[str, ...]] = []
        for pred_idx, pred in enumerate(pred_buttons):
            target_idx = pred_idx + int(offset)
            if 0 <= target_idx < len(gt_buttons):
                aligned_pred.append(pred)
                aligned_gt.append(gt_buttons[target_idx])
        metrics = _button_metrics(aligned_pred, aligned_gt)
        rows.append({"offset": int(offset), **{k: v for k, v in metrics.items() if not k.startswith("top_")}})
    return rows


def _apply_mapping(buttons: Sequence[tuple[str, ...]], mapping: dict[str, str]) -> list[tuple[str, ...]]:
    return [tuple(sorted(mapping.get(token, token) for token in row)) for row in buttons]


def _mapping_diagnostic(pred_buttons: list[tuple[str, ...]], gt_buttons: list[tuple[str, ...]]) -> dict[str, Any]:
    pred_vocab = sorted({token for row in pred_buttons for token in row})
    gt_vocab = sorted({token for row in gt_buttons for token in row})
    cooccurrence: Counter[str] = Counter()
    for pred, gt in zip(pred_buttons, gt_buttons):
        if not pred or not gt:
            continue
        for ptoken in set(pred):
            for gtoken in set(gt):
                cooccurrence[f"{ptoken}=>{gtoken}"] += 1
    greedy: dict[str, str] = {}
    for ptoken in pred_vocab:
        best = sorted(((count, pair.split("=>", 1)[1]) for pair, count in cooccurrence.items() if pair.startswith(f"{ptoken}=>")), reverse=True)
        if best:
            greedy[ptoken] = best[0][1]
    greedy_metrics = _button_metrics(_apply_mapping(pred_buttons, greedy), gt_buttons) if greedy else None

    exhaustive: dict[str, Any] | None = None
    if 0 < len(pred_vocab) <= 7 and len(pred_vocab) <= len(gt_vocab) <= 7:
        best_payload: dict[str, Any] | None = None
        for perm in itertools.permutations(gt_vocab, len(pred_vocab)):
            mapping = dict(zip(pred_vocab, perm))
            metrics = _button_metrics(_apply_mapping(pred_buttons, mapping), gt_buttons)
            score = (
                int(metrics["exact_true_positive_examples"]),
                int(metrics["semantic_any_button_overlap_examples"]),
                -int(metrics["false_positive_examples"]),
            )
            if best_payload is None or score > best_payload["score_tuple"]:
                best_payload = {"mapping": mapping, "metrics": metrics, "score_tuple": score}
        if best_payload is not None:
            best_payload.pop("score_tuple", None)
            exhaustive = best_payload
    return {
        "predicted_vocab": pred_vocab,
        "ground_truth_vocab": gt_vocab,
        "cooccurrence_top": cooccurrence.most_common(50),
        "greedy_mapping": greedy,
        "greedy_metrics": greedy_metrics,
        "best_one_to_one_mapping": exhaustive,
        "claim_boundary": "Mapping diagnostics use target labels to diagnose semantic/ranking failures only; mappings are not valid calibration/training evidence unless revalidated without target leakage.",
    }


def _examples(pred_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], pred_buttons: list[tuple[str, ...]], gt_buttons: list[tuple[str, ...]], *, max_examples: int) -> dict[str, list[dict[str, Any]]]:
    out = {"wrong_positive": [], "no_button_false_positive": [], "missed_ground_truth": [], "exact_true_positive": []}
    for idx, (pred, gt) in enumerate(zip(pred_buttons, gt_buttons)):
        bucket: str | None = None
        if pred and gt and pred != gt:
            bucket = "wrong_positive"
        elif pred and not gt:
            bucket = "no_button_false_positive"
        elif gt and not pred:
            bucket = "missed_ground_truth"
        elif pred and pred == gt:
            bucket = "exact_true_positive"
        if bucket and len(out[bucket]) < max_examples:
            out[bucket].append(
                {
                    "index": idx,
                    "prediction_sequence_id": _row_id(pred_rows[idx]),
                    "target_sequence_id": _row_id(target_rows[idx]),
                    "predicted_buttons": list(pred),
                    "ground_truth_buttons": list(gt),
                }
            )
    return out


def build_button_semantic_ranking_diagnostic(
    *,
    prediction_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    max_rows: int | None = None,
    offsets: Sequence[int] = tuple(range(-5, 6)),
    max_examples: int = 25,
) -> dict[str, Any]:
    pred_rows = list(_iter_rows(prediction_paths, max_rows=max_rows))
    target_rows = list(_iter_rows(target_paths, max_rows=max_rows))
    rows = min(len(pred_rows), len(target_rows))
    pred_rows = pred_rows[:rows]
    target_rows = target_rows[:rows]
    pred_buttons = [_button_tokens(_tokens(row, "predicted_tokens")) for row in pred_rows]
    gt_buttons = [_button_tokens(_tokens(row, "ground_truth_tokens")) for row in target_rows]
    mismatches = []
    for idx, (pred, target) in enumerate(zip(pred_rows, target_rows)):
        if _row_id(pred) != _row_id(target) and len(mismatches) < max_examples:
            mismatches.append({"index": idx, "prediction_sequence_id": _row_id(pred), "target_sequence_id": _row_id(target)})
    base = _button_metrics(pred_buttons, gt_buttons)
    sweep = _offset_sweep(pred_buttons, gt_buttons, offsets)
    best_exact = max(sweep, key=lambda row: (row["exact_true_positive_examples"], row["semantic_any_button_overlap_examples"], -abs(row["offset"]))) if sweep else None
    best_semantic = max(sweep, key=lambda row: (row["semantic_any_button_overlap_examples"], row["exact_true_positive_examples"], -abs(row["offset"]))) if sweep else None
    return {
        "schema": "g005_button_semantic_ranking_diagnostic.v1",
        "status": "pass",
        "alignment": {
            "rows_seen": rows,
            "prediction_rows": len(pred_rows),
            "target_rows": len(target_rows),
            "sequence_id_mismatches": sum(1 for pred, target in zip(pred_rows, target_rows) if _row_id(pred) != _row_id(target)),
            "examples": mismatches,
        },
        "base": base,
        "offset_sweep": sweep,
        "best_offset_by_exact": best_exact,
        "best_offset_by_semantic_overlap": best_semantic,
        "mapping_diagnostic": _mapping_diagnostic(pred_buttons, gt_buttons),
        "examples": _examples(pred_rows, target_rows, pred_buttons, gt_buttons, max_examples=max_examples),
        "claim_boundary": "Diagnostic may inspect target labels to explain failed prefix probes. It is not training/calibration evidence and cannot by itself satisfy G005.",
    }


def write_button_semantic_ranking_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_button_semantic_ranking_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
