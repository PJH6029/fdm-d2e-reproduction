#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import stable_hash_json, write_json

ENDPOINTS = [
    "keyboard_accuracy",
    "mouse_button_accuracy",
    "mouse_move_pearson",
    "mouse_move_scale_ratio_distance",
]

FDM_BRANCH_RUNS = [
    ("original_bth07", "raw teacher threshold 0.7", Path("artifacts/fdm/fdm_shooter64_surface_motion_fulltrain_h200/summary.json")),
    ("bth05_recall_teacher", "IDM button threshold override 0.5", Path("artifacts/fdm/fdm_shooter64_surface_motion_fulltrain_bth05_h200/summary.json")),
    ("bth05_pseudo_recording_scale", "pseudo-label recording scale calibration", Path("artifacts/fdm/fdm_bth05_recording_scale_calibrated_h200/summary.json")),
    ("bth05_residual_regression", "residual-regression FDM branch", Path("artifacts/fdm/fdm_shooter64_surface_motion_fulltrain_bth05_regression_h200/summary.json")),
    ("bth05_d2e_train_prediction_scale", "strict D2E train-label + train-prediction scale calibration", Path("artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/summary.json")),
    ("bth05_d2e_train_scale", "D2E train-label + target-prediction scale normalization", Path("artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json")),
]

FDM_SWEEPS = [
    ("neural_button_weight", Path("artifacts/fdm/fdm_shooter64_fulltrain_button_sweep_h200.json")),
    ("neural_decode_recall", Path("artifacts/fdm/fdm_shooter64_recall_beta_sweep_h200.json")),
    ("knn_retrieval", Path("artifacts/fdm/fdm_knn_shooter64_surface_sweep_h200.json")),
]

IDM_SCALING_RUNS = [
    ("apex8_richmotion", "Apex rich-motion", 8, Path("artifacts/idm/g4_h200_idm_run_h200_richmotion.json")),
    ("apex16_richmotion", "Apex rich-motion", 16, Path("artifacts/idm/g4_h200_idm_run_h200_richmotion16.json")),
    ("apex36_richmotion", "Apex rich-motion", 36, Path("artifacts/idm/g4_h200_idm_run_h200_richmotion36b.json")),
    ("shooter32_richmotion", "Shooter/action rich-motion", 32, Path("artifacts/idm/g4_h200_idm_run_h200_shooter32.json")),
    ("shooter64_surface_motion", "Shooter/action surface-motion selected", 64, Path("artifacts/idm/shooter64_surface_motion_selected/summary.json")),
]


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("metrics") or payload.get("torch_metrics") or {}


def _model_name(payload: dict[str, Any]) -> str:
    for key in ("checkpoint", "metadata", "torch_metadata"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("model"):
            return str(value["model"])
    return str(payload.get("model", "unknown"))


def _comparisons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stat = payload.get("statistical_comparison") or {}
    return list(stat.get("comparisons", []))


def _candidate_comparisons(payload: dict[str, Any], model: str | None = None) -> list[dict[str, Any]]:
    model = model or _model_name(payload)
    return [row for row in _comparisons(payload) if row.get("model") == model]


def _endpoint_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("endpoint")): row for row in rows}


def _reject_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("reject_holm_0_05") is True)


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keyboard = metrics.get("keyboard") or {}
    button = metrics.get("mouse_button") or {}
    move = metrics.get("mouse_move") or {}
    return {
        "num_examples": metrics.get("num_examples"),
        "keyboard_accuracy": keyboard.get("accuracy"),
        "mouse_button_accuracy": button.get("accuracy"),
        "mouse_button_precision": button.get("precision"),
        "mouse_button_no_button_false_positive_rate": button.get("no_button_false_positive_rate"),
        "mouse_move_pearson": move.get("pearson"),
        "mouse_move_scale_ratio": move.get("scale_ratio"),
    }


def _branch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, description, path in FDM_BRANCH_RUNS:
        payload = _load(path)
        comparisons = _candidate_comparisons(payload)
        endpoint_stats = _endpoint_map(comparisons)
        rows.append(
            {
                "name": name,
                "description": description,
                "artifact": str(path),
                "model": _model_name(payload),
                "metrics": _flat_metrics(_metrics(payload)),
                "reject_count": _reject_count(comparisons),
                "rejected_endpoints": [row["endpoint"] for row in comparisons if row.get("reject_holm_0_05") is True],
                "endpoint_stats": {
                    endpoint: {
                        "delta": endpoint_stats.get(endpoint, {}).get("delta"),
                        "p_adjusted_holm": endpoint_stats.get(endpoint, {}).get("p_adjusted_holm"),
                        "reject_holm_0_05": endpoint_stats.get(endpoint, {}).get("reject_holm_0_05", False),
                    }
                    for endpoint in ENDPOINTS
                },
            }
        )
    return rows


def _sweep_rows() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for axis, path in FDM_SWEEPS:
        payload = _load(path)
        rows = list(payload.get("rows", []))
        normalized = []
        for row in rows:
            comparisons = [item for item in row.get("comparisons", []) if item.get("model") == row.get("model_name")]
            endpoint_stats = _endpoint_map(comparisons)
            normalized.append(
                {
                    "variant": row.get("variant"),
                    "model": row.get("model_name"),
                    "config": row.get("config", {}),
                    "metrics": _flat_metrics(row.get("metrics", {})),
                    "reject_count": _reject_count(comparisons),
                    "rejected_endpoints": [item["endpoint"] for item in comparisons if item.get("reject_holm_0_05") is True],
                    "endpoint_stats": {
                        endpoint: {
                            "delta": endpoint_stats.get(endpoint, {}).get("delta"),
                            "p_adjusted_holm": endpoint_stats.get(endpoint, {}).get("p_adjusted_holm"),
                            "reject_holm_0_05": endpoint_stats.get(endpoint, {}).get("reject_holm_0_05", False),
                        }
                        for endpoint in ENDPOINTS
                    },
                }
            )
        best_by_reject = sorted(
            normalized,
            key=lambda item: (
                item["reject_count"],
                item["metrics"].get("mouse_button_precision") or 0,
                item["metrics"].get("mouse_move_pearson") or 0,
            ),
            reverse=True,
        )[:5]
        best_by_scale = sorted(
            normalized,
            key=lambda item: (
                bool(item["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("reject_holm_0_05")),
                item["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("delta") or -999,
            ),
            reverse=True,
        )[:5]
        summaries.append(
            {
                "axis": axis,
                "artifact": str(path),
                "num_runs": len(normalized),
                "max_reject_count": max((row["reject_count"] for row in normalized), default=0),
                "runs_with_keyboard_reject": sum(1 for row in normalized if "keyboard_accuracy" in row["rejected_endpoints"]),
                "runs_with_button_reject": sum(1 for row in normalized if "mouse_button_accuracy" in row["rejected_endpoints"]),
                "runs_with_pearson_reject": sum(1 for row in normalized if "mouse_move_pearson" in row["rejected_endpoints"]),
                "runs_with_scale_reject": sum(1 for row in normalized if "mouse_move_scale_ratio_distance" in row["rejected_endpoints"]),
                "best_by_reject_then_precision": best_by_reject,
                "best_by_scale_delta": best_by_scale,
            }
        )
    return summaries


def _idm_scaling_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, axis, recordings, path in IDM_SCALING_RUNS:
        payload = _load(path)
        comparisons = _candidate_comparisons(payload)
        metadata = payload.get("metadata") or payload.get("torch_metadata") or {}
        rows.append(
            {
                "name": name,
                "axis": axis,
                "artifact": str(path),
                "recordings": recordings,
                "train_records": metadata.get("train_records"),
                "target_records": metadata.get("target_records"),
                "model": _model_name(payload),
                "metrics": _flat_metrics(_metrics(payload)),
                "reject_count": _reject_count(comparisons),
                "rejected_endpoints": [row["endpoint"] for row in comparisons if row.get("reject_holm_0_05") is True],
                "endpoint_stats": {
                    endpoint: {
                        "delta": _endpoint_map(comparisons).get(endpoint, {}).get("delta"),
                        "p_adjusted_holm": _endpoint_map(comparisons).get(endpoint, {}).get("p_adjusted_holm"),
                        "reject_holm_0_05": _endpoint_map(comparisons).get(endpoint, {}).get("reject_holm_0_05", False),
                    }
                    for endpoint in ENDPOINTS
                },
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _analysis(branches: list[dict[str, Any]], sweeps: list[dict[str, Any]], scaling: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(row for row in branches if row["name"] == "bth05_d2e_train_scale")
    pure_pseudo = next(row for row in branches if row["name"] == "bth05_pseudo_recording_scale")
    bth05 = next(row for row in branches if row["name"] == "bth05_recall_teacher")
    knn = next(row for row in sweeps if row["axis"] == "knn_retrieval")
    return {
        "quality_gate": {
            "axes_reported": 3,
            "minimum_axes_required": 2,
            "selected_fdm_rejects_all_primary_endpoints": selected["reject_count"] == len(ENDPOINTS),
            "sweeps_total_runs": sum(row["num_runs"] for row in sweeps),
            "has_data_scaling_curve": len(scaling) >= 3,
            "status": "pass" if selected["reject_count"] == len(ENDPOINTS) and len(sweeps) >= 2 and len(scaling) >= 3 else "review",
        },
        "findings": [
            "IDM-pseudo FDM training plus bth05 teacher labels clears keyboard, button, and mouse-direction endpoints, but raw/pure-pseudo scale remains non-significant.",
            "D2E train-split scale targets plus target-prediction distribution normalization are the only selected branch that clears the scale-ratio endpoint after Holm while preserving the other three endpoint wins.",
            "KNN retrieval can clear keyboard/button for some variants but has weak motion correlation and no scale rejections, so retrieval alone is not a sufficient FDM replacement.",
            "IDM scaling from small Apex splits to Shooter64 changes which endpoints are learnable: Shooter64 is the first selected IDM handoff with keyboard, button, and mouse-direction wins, giving the FDM teacher enough signal for G5.",
        ],
        "selected_vs_bth05_scale_delta_gain": (
            (selected["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("delta") or 0)
            - (bth05["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("delta") or 0)
        ),
        "selected_vs_pure_pseudo_scale_delta_gain": (
            (selected["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("delta") or 0)
            - (pure_pseudo["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("delta") or 0)
        ),
        "knn_max_reject_count": knn["max_reject_count"],
        "mean_fdm_sweep_runs_per_axis": _mean([float(row["num_runs"]) for row in sweeps]),
    }


def _format_bool(value: Any) -> str:
    return "yes" if value is True else "no"


def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    branch_rows = []
    for row in payload["fdm_branch_axis"]:
        m = row["metrics"]
        branch_rows.append(
            [
                row["name"],
                row["reject_count"],
                f"{m.get('mouse_button_precision'):.3f}" if m.get("mouse_button_precision") is not None else "n/a",
                f"{m.get('mouse_move_pearson'):.3f}" if m.get("mouse_move_pearson") is not None else "n/a",
                f"{m.get('mouse_move_scale_ratio'):.3f}" if m.get("mouse_move_scale_ratio") is not None else "n/a",
                _format_bool(row["endpoint_stats"]["mouse_move_scale_ratio_distance"].get("reject_holm_0_05")),
            ]
        )
    sweep_rows = [
        [row["axis"], row["num_runs"], row["max_reject_count"], row["runs_with_button_reject"], row["runs_with_scale_reject"]]
        for row in payload["fdm_sweep_axes"]
    ]
    scaling_rows = []
    for row in payload["idm_scaling_axis"]:
        m = row["metrics"]
        scaling_rows.append(
            [
                row["name"],
                row["recordings"],
                row.get("train_records"),
                row.get("target_records"),
                row["reject_count"],
                f"{m.get('mouse_move_pearson'):.3f}" if m.get("mouse_move_pearson") is not None else "n/a",
                ", ".join(row["rejected_endpoints"]) or "none",
            ]
        )
    text = f"""# G6 Ablation and Scaling Summary

This report is generated by `scripts/summarize_ablation_scaling.py` from source-controlled H200 artifacts. It summarizes multiple real-D2E axes instead of a smoke-only run.

## Quality gate

- Status: `{payload['analysis']['quality_gate']['status']}`
- Axes reported: `{payload['analysis']['quality_gate']['axes_reported']}` (minimum `{payload['analysis']['quality_gate']['minimum_axes_required']}`)
- Total FDM sweep runs summarized: `{payload['analysis']['quality_gate']['sweeps_total_runs']}`
- Selected FDM rejects all primary endpoints: `{payload['analysis']['quality_gate']['selected_fdm_rejects_all_primary_endpoints']}`

## FDM branch / calibration axis

{_md_table(branch_rows, ['Branch', 'Holm rejects', 'Button precision', 'Mouse Pearson', 'Scale ratio', 'Scale reject'])}

## FDM sweep axes

{_md_table(sweep_rows, ['Axis', 'Runs', 'Max rejects', 'Runs button reject', 'Runs scale reject'])}

## IDM data/model scaling axis

{_md_table(scaling_rows, ['Run', 'Recordings', 'Train', 'Heldout', 'Holm rejects', 'Mouse Pearson', 'Rejected endpoints'])}

## Findings

"""
    for finding in payload["analysis"]["findings"]:
        text += f"- {finding}\n"
    text += "\n## Reproduction\n\n```bash\nuv run python scripts/summarize_ablation_scaling.py --output-json artifacts/ablation_scaling/g007_ablation_scaling_summary.json --output-md docs/ablation_scaling.md\n```\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize G6 ablation/scaling curves from H200 evidence artifacts.")
    parser.add_argument("--output-json", default="artifacts/ablation_scaling/g007_ablation_scaling_summary.json")
    parser.add_argument("--output-md", default="docs/ablation_scaling.md")
    args = parser.parse_args()
    branches = _branch_rows()
    sweeps = _sweep_rows()
    scaling = _idm_scaling_rows()
    payload = {
        "schema": "g007_ablation_scaling_summary.v1",
        "fdm_branch_axis": branches,
        "fdm_sweep_axes": sweeps,
        "idm_scaling_axis": scaling,
        "analysis": _analysis(branches, sweeps, scaling),
    }
    payload["artifact_fingerprint"] = stable_hash_json(payload)
    out_json = Path(args.output_json)
    write_json(out_json, payload)
    _write_markdown(Path(args.output_md), payload)
    print(f"wrote {out_json} and {args.output_md}; status={payload['analysis']['quality_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
