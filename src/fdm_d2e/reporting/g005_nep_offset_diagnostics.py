from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MetricMap = dict[str, float | int | None]


_HIGHER_IS_BETTER = {
    "keyboard_accuracy": True,
    "mouse_button_accuracy": True,
    "pearson_x": True,
    "pearson_y": True,
}
_LOWER_IS_BETTER = {
    "scale_ratio_x": True,
    "scale_ratio_y": True,
}
_TARGET_KEYS = tuple(_HIGHER_IS_BETTER) + tuple(_LOWER_IS_BETTER)


def _get(payload: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_paper_targets(contract_payload: Mapping[str, Any]) -> dict[str, float]:
    primary = _get(contract_payload, ["target_sequence", "phase_1", "primary_targets"], {})
    if not isinstance(primary, Mapping):
        return {}
    mapping = {
        "keyboard_accuracy": "keyboard_accuracy",
        "mouse_button_accuracy": "mouse_button_accuracy",
        "pearson_x": "pearson_x",
        "pearson_y": "pearson_y",
        "scale_ratio_x": "scale_ratio_x_max",
        "scale_ratio_y": "scale_ratio_y_max",
    }
    targets: dict[str, float] = {}
    for metric, source_key in mapping.items():
        value = _as_float(primary.get(source_key))
        if value is not None:
            targets[metric] = value
    return targets


def extract_group_metrics(group_payload: Mapping[str, Any]) -> MetricMap:
    paper = _get(group_payload, ["paper_compatible"], {})
    strict = _get(group_payload, ["strict_local"], {})
    return {
        "rows": _as_float(group_payload.get("rows")),
        "keyboard_accuracy": _as_float(_get(paper, ["keyboard", "key_accuracy"])),
        "mouse_button_accuracy": _as_float(_get(paper, ["mouse_button", "button_accuracy"])),
        "pearson_x": _as_float(_get(paper, ["mouse_move", "pearson_x"])),
        "pearson_y": _as_float(_get(paper, ["mouse_move", "pearson_y"])),
        "scale_ratio_x": _as_float(_get(paper, ["mouse_move", "scale_ratio_x"])),
        "scale_ratio_y": _as_float(_get(paper, ["mouse_move", "scale_ratio_y"])),
        "strict_mouse_button_f1": _as_float(_get(strict, ["mouse_button", "f1"])),
        "strict_no_button_fpr": _as_float(_get(strict, ["mouse_button", "no_button_false_positive_rate"])),
    }


def target_results(metrics: Mapping[str, Any], targets: Mapping[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in _TARGET_KEYS:
        if metric not in targets:
            continue
        actual = _as_float(metrics.get(metric))
        target = float(targets[metric])
        direction = "higher" if metric in _HIGHER_IS_BETTER else "lower"
        if actual is None:
            passed = False
            gap = None
        elif direction == "higher":
            passed = actual >= target
            gap = target - actual
        else:
            passed = actual <= target
            gap = actual - target
        rows.append(
            {
                "metric": metric,
                "actual": actual,
                "target": target,
                "direction": direction,
                "passed": passed,
                "gap_to_target": gap,
            }
        )
    return rows


def _target_passes(results: Sequence[Mapping[str, Any]]) -> bool:
    return bool(results) and all(bool(row.get("passed")) for row in results)


def _baseline_metrics_from_payload(payload: Mapping[str, Any] | None) -> MetricMap | None:
    if not payload:
        return None
    groups = payload.get("groups")
    if isinstance(groups, Mapping) and isinstance(groups.get("all"), Mapping):
        return extract_group_metrics(groups["all"])
    if isinstance(payload.get("paper_compatible"), Mapping):
        return extract_group_metrics(payload)
    return None


def _compare_to_baseline(metrics: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    comparisons: list[dict[str, Any]] = []
    for metric in _TARGET_KEYS:
        actual = _as_float(metrics.get(metric))
        base = _as_float(baseline.get(metric))
        if actual is None or base is None:
            delta = None
            improved = False
        elif metric in _HIGHER_IS_BETTER:
            delta = actual - base
            improved = delta > 0
        else:
            delta = base - actual
            improved = delta > 0
        comparisons.append(
            {
                "metric": metric,
                "target_autocorr_value": actual,
                "baseline_value": base,
                "improvement_direction": "higher" if metric in _HIGHER_IS_BETTER else "lower",
                "delta_in_favor_of_shift": delta,
                "improves_over_baseline": improved,
            }
        )
    return comparisons


def _shift_payload(target_autocorr: Mapping[str, Any], shift: int) -> Mapping[str, Any]:
    value = target_autocorr.get(str(shift), {})
    return value if isinstance(value, Mapping) else {}


def _all_metrics_for_shift(target_autocorr: Mapping[str, Any], shift: int) -> MetricMap:
    payload = _shift_payload(target_autocorr, shift)
    group = payload.get("all", {})
    return extract_group_metrics(group) if isinstance(group, Mapping) else extract_group_metrics({})


def _best_nonzero_shift_by_metric(
    shift_metrics: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in shift_metrics:
        shift = int(row["shift"])
        if shift == 0:
            continue
        value = _as_float(row.get(metric))
        if value is None:
            continue
        if best is None:
            best = {"shift": shift, "value": value}
            continue
        if metric in _HIGHER_IS_BETTER and value > float(best["value"]):
            best = {"shift": shift, "value": value}
        elif metric in _LOWER_IS_BETTER and value < float(best["value"]):
            best = {"shift": shift, "value": value}
    return best


def build_g005_nep_offset_summary(
    *,
    diagnostics_payload: Mapping[str, Any],
    contract_payload: Mapping[str, Any],
    baseline_metrics_payload: Mapping[str, Any] | None = None,
    expected_nep_shift: int = 2,
    source_label: str = "target_autocorr",
) -> dict[str, Any]:
    target_autocorr = _get(diagnostics_payload, ["diagnostics", "target_autocorr"], {})
    if not isinstance(target_autocorr, Mapping):
        target_autocorr = {}
    pair_counts = _get(diagnostics_payload, ["diagnostics", "pair_counts"], {})
    if not isinstance(pair_counts, Mapping):
        pair_counts = {}
    paper_targets = extract_paper_targets(contract_payload)
    baseline = _baseline_metrics_from_payload(baseline_metrics_payload)

    all_shift_metrics: list[dict[str, Any]] = []
    for shift_text in sorted(target_autocorr, key=lambda item: int(item)):
        shift = int(shift_text)
        metrics = _all_metrics_for_shift(target_autocorr, shift)
        results = target_results(metrics, paper_targets)
        all_shift_metrics.append(
            {
                "shift": shift,
                "shift_ms": shift * 50,
                "pair_count": int(pair_counts.get(shift_text, 0) or 0),
                **metrics,
                "paper_target_passes": _target_passes(results),
                "paper_target_results": results,
            }
        )

    expected_metrics = _all_metrics_for_shift(target_autocorr, expected_nep_shift)
    expected_results = target_results(expected_metrics, paper_targets)
    split_metrics: dict[str, Any] = {}
    expected_shift_payload = _shift_payload(target_autocorr, expected_nep_shift)
    for group_name, group_payload in sorted(expected_shift_payload.items()):
        if not isinstance(group_payload, Mapping) or group_name == "all":
            continue
        metrics = extract_group_metrics(group_payload)
        split_metrics[group_name] = {
            **metrics,
            "paper_target_results": target_results(metrics, paper_targets),
        }

    nonzero_passes = [row for row in all_shift_metrics if row["shift"] != 0 and row["paper_target_passes"]]
    best_by_metric = {
        metric: _best_nonzero_shift_by_metric(all_shift_metrics, metric=metric)
        for metric in _TARGET_KEYS
    }
    expected_vs_baseline = _compare_to_baseline(expected_metrics, baseline)
    expected_passes = _target_passes(expected_results)
    diagnostics_status = diagnostics_payload.get("status")

    findings: list[dict[str, Any]] = []
    if diagnostics_status != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "alignment_shift_diagnostics_not_pass",
                "status": diagnostics_status,
                "error_count": diagnostics_payload.get("error_count"),
            }
        )
    if expected_passes:
        findings.append(
            {
                "severity": "info",
                "code": "expected_nep_shift_target_autocorr_meets_paper_targets",
                "shift": expected_nep_shift,
                "shift_ms": expected_nep_shift * 50,
            }
        )
    else:
        blockers = [row for row in expected_results if not row.get("passed")]
        findings.append(
            {
                "severity": "info",
                "code": "expected_nep_shift_target_autocorr_below_paper_targets",
                "shift": expected_nep_shift,
                "shift_ms": expected_nep_shift * 50,
                "blocking_metrics": blockers,
            }
        )
    if nonzero_passes:
        findings.append(
            {
                "severity": "info",
                "code": "nonzero_target_autocorr_shift_meets_all_paper_targets",
                "shifts": [row["shift"] for row in nonzero_passes],
            }
        )
    else:
        findings.append(
            {
                "severity": "info",
                "code": "no_nonzero_target_autocorr_shift_meets_all_paper_targets",
            }
        )

    return {
        "schema": "g005_nep_offset_diagnostics.v1",
        "status": "pass" if diagnostics_status == "pass" else "fail",
        "source_label": source_label,
        "expected_nep_shift": expected_nep_shift,
        "expected_nep_shift_ms": expected_nep_shift * 50,
        "paper_targets": paper_targets,
        "diagnostics_status": diagnostics_status,
        "alignment": diagnostics_payload.get("alignment", {}),
        "block_stats": diagnostics_payload.get("block_stats", {}),
        "claim_boundary": "target_autocorr is a temporal-offset diagnostic/oracle over ground-truth targets, not a trained model result and not valid G005 completion evidence.",
        "identity_shift_note": "shift=0 is an identity sanity check and is excluded from nonzero offset recommendations.",
        "expected_shift": {
            "shift": expected_nep_shift,
            "shift_ms": expected_nep_shift * 50,
            "pair_count": int(pair_counts.get(str(expected_nep_shift), 0) or 0),
            "metrics": expected_metrics,
            "paper_target_results": expected_results,
            "paper_target_passes": expected_passes,
            "comparison_to_baseline": expected_vs_baseline,
            "split_metrics": split_metrics,
        },
        "all_shift_metrics": all_shift_metrics,
        "best_nonzero_shift_by_metric": best_by_metric,
        "nonzero_shifts_meeting_all_paper_targets": nonzero_passes,
        "findings": findings,
        "interpretation": {
            "shift_definition": _get(diagnostics_payload, ["interpretation", "target_autocorr_definition"]),
            "next_action_if_expected_shift_fails": "Do not spend a 4xH200 run on a simple NEP/label-offset hypothesis; change the supervision/modeling problem or use a teacher/latent-state branch with a prefix gate first.",
            "next_action_if_expected_shift_passes": "Validate a shifted-label or NEP-aligned training candidate on prefix rows before any full-corpus 4xH200 promotion.",
        },
    }
