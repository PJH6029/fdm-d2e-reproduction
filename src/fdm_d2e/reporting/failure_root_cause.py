from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _get(data: dict[str, Any] | None, path: str | list[str]) -> Any:
    cur: Any = data
    parts = path if isinstance(path, list) else path.split(".")
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(str(part))
    return cur


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(num: Any, den: Any) -> float | None:
    n = _float(num)
    d = _float(den)
    if n is None or d in (None, 0.0):
        return None
    return n / d


def _metric_snapshot(metrics: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "num_examples": _get(metrics, "num_examples"),
        "keyboard_accuracy": _get(metrics, "keyboard.accuracy"),
        "keyboard_examples": _get(metrics, "keyboard.num_examples"),
        "mouse_button_accuracy": _get(metrics, "mouse_button.accuracy"),
        "mouse_button_f1": _get(metrics, "mouse_button.f1"),
        "mouse_button_precision": _get(metrics, "mouse_button.precision"),
        "mouse_button_recall": _get(metrics, "mouse_button.recall"),
        "mouse_button_examples": _get(metrics, "mouse_button.num_examples"),
        "mouse_button_predicted_examples": _get(metrics, "mouse_button.predicted_examples"),
        "mouse_button_false_positive_examples": _get(metrics, "mouse_button.false_positive_examples"),
        "mouse_button_false_negative_examples": _get(metrics, "mouse_button.false_negative_examples"),
        "no_button_examples": _get(metrics, "mouse_button.no_button_examples"),
        "no_button_false_positive_examples": _get(metrics, "mouse_button.no_button_false_positive_examples"),
        "no_button_false_positive_rate": _get(metrics, "mouse_button.no_button_false_positive_rate"),
        "mouse_move_pearson": _get(metrics, "mouse_move.pearson"),
        "mouse_move_scale_ratio": _get(metrics, "mouse_move.scale_ratio"),
        "mouse_move_num_values": _get(metrics, "mouse_move.num_values"),
        "failure_count": _get(metrics, "failure_count"),
    }


def _button_distribution(metrics: dict[str, Any] | None) -> dict[str, Any]:
    num_examples = _get(metrics, "num_examples")
    true_button = _get(metrics, "mouse_button.num_examples")
    predicted_button = _get(metrics, "mouse_button.predicted_examples")
    no_button = _get(metrics, "mouse_button.no_button_examples")
    return {
        "num_examples": num_examples,
        "true_button_examples": true_button,
        "predicted_button_examples": predicted_button,
        "no_button_examples": no_button,
        "true_button_rate": _ratio(true_button, num_examples),
        "predicted_button_rate": _ratio(predicted_button, num_examples),
        "no_button_rate": _ratio(no_button, num_examples),
        "predicted_to_true_button_rate_ratio": _ratio(_ratio(predicted_button, num_examples), _ratio(true_button, num_examples)),
        "no_button_false_positive_rate": _get(metrics, "mouse_button.no_button_false_positive_rate"),
    }


def _vocab_counts(metadata: dict[str, Any] | None) -> dict[str, Any]:
    vocab = list(_get(metadata, "torch_checkpoint_metadata.categorical_vocab") or _get(metadata, "categorical_vocab") or [])
    prefixes = {
        "key_press": "KEY_PRESS_",
        "key_release": "KEY_RELEASE_",
        "mouse_button": "MOUSE_",
        "mouse_dx": "MOUSE_DX_",
        "mouse_dy": "MOUSE_DY_",
        "noop": "NOOP",
    }
    counts: dict[str, int] = {}
    for name, prefix in prefixes.items():
        if prefix == "NOOP":
            counts[name] = sum(1 for token in vocab if token == "NOOP")
        else:
            counts[name] = sum(1 for token in vocab if str(token).startswith(prefix))
    x_button_tokens = [token for token in vocab if str(token).startswith(("MOUSE_X1_", "MOUSE_X2_"))]
    return {
        "vocab_size": len(vocab),
        "counts": counts,
        "has_noop_token": "NOOP" in vocab,
        "x_button_tokens": sorted(str(token) for token in x_button_tokens),
        "metric_mouse_button_prefixes": ["MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"],
        "metric_compatibility_note": "Current metric helper counts left/right/middle mouse buttons; X1/X2 tokens are present in the model vocabulary and need explicit accounting if D2E labels use them.",
    }


def _split_rows(root: Path, summary_path: str | None, model_name: str) -> list[dict[str, Any]]:
    if not summary_path:
        return []
    summary = _load_json(root / summary_path)
    rows: list[dict[str, Any]] = []
    for output in (summary or {}).get("outputs", []):
        if not isinstance(output, dict):
            continue
        split = str(output.get("split"))
        path = str(output.get("path"))
        payload = _load_json(root / path)
        for row in (payload or {}).get("comparisons", []):
            if not isinstance(row, dict) or str(row.get("model")) != model_name:
                continue
            rows.append(
                {
                    "split": split,
                    "endpoint": row.get("endpoint"),
                    "candidate_value": row.get("candidate_value"),
                    "status": row.get("status"),
                    "reference": row.get("reference"),
                    "p_adjusted_holm": row.get("p_adjusted_holm"),
                    "reject_holm_0_05": row.get("reject_holm_0_05"),
                    "path": path,
                }
            )
    return rows


def _split_pivot(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pivot: dict[str, dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split"))
        endpoint = str(row.get("endpoint"))
        pivot.setdefault(split, {})[endpoint] = row.get("candidate_value")
    return dict(sorted(pivot.items()))


def _worst_split(rows: list[dict[str, Any]], endpoint: str, *, lower_is_better: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("endpoint") == endpoint and _float(row.get("candidate_value")) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: _float(row.get("candidate_value")) or 0.0, reverse=lower_is_better)[0]


def _external_entries(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    entries = (manifest or {}).get("entries", [])
    if isinstance(entries, dict):
        return {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}
    return {str(row.get("path")): dict(row) for row in entries if isinstance(row, dict) and row.get("path")}


def _external_proof(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {"exists": False}
    return {
        "exists": bool(entry.get("exists")),
        "bytes": entry.get("bytes"),
        "sha256": entry.get("sha256"),
        "fingerprint": entry.get("fingerprint"),
        "fingerprint_type": entry.get("fingerprint_type"),
        "storage_uri": entry.get("storage_uri"),
        "proof": entry.get("proof"),
    }


def _axis(status: str, evidence: dict[str, Any], *, summary: str, severity: str = "info") -> dict[str, Any]:
    return {"status": status, "severity": severity, "summary": summary, "evidence": evidence}


def _ranked_causes(
    *,
    idm_metrics: dict[str, Any] | None,
    fdm_metrics: dict[str, Any] | None,
    aux_metrics: dict[str, Any] | None,
    fdm_metadata: dict[str, Any] | None,
    renewed_gate: dict[str, Any] | None,
    raw_status: str,
) -> list[dict[str, Any]]:
    fdm_dist = _button_distribution(fdm_metrics)
    idm_dist = _button_distribution(idm_metrics)
    fdm_fpr = _float(_get(fdm_metrics, "mouse_button.no_button_false_positive_rate")) or 0.0
    idm_fpr = _float(_get(idm_metrics, "mouse_button.no_button_false_positive_rate")) or 0.0
    fdm_pred_rate = _float(fdm_dist.get("predicted_button_rate")) or 0.0
    fdm_true_rate = _float(fdm_dist.get("true_button_rate")) or 0.0
    idm_f1 = _float(_get(idm_metrics, "mouse_button.f1")) or 0.0
    keyboard_gap = _float(_get(fdm_metrics, "keyboard.accuracy")) or 0.0
    aux_delta = None
    if aux_metrics is not None and fdm_metrics is not None:
        aux_delta = (_float(_get(aux_metrics, "mouse_button.f1")) or 0.0) - (_float(_get(fdm_metrics, "mouse_button.f1")) or 0.0)
    causes = [
        {
            "rank": 1,
            "id": "fdm_mouse_button_overfire",
            "severity": "critical",
            "score": round(fdm_fpr * 100.0, 4),
            "evidence": {
                "fdm_predicted_button_rate": fdm_pred_rate,
                "ground_truth_button_rate": fdm_true_rate,
                "predicted_to_true_ratio": _ratio(fdm_pred_rate, fdm_true_rate),
                "fdm_no_button_fpr": fdm_fpr,
                "idm_no_button_fpr": idm_fpr,
                "fdm_to_idm_no_button_fpr_ratio": _ratio(fdm_fpr, idm_fpr),
            },
            "implication": "The current FDM fires mouse buttons on no-button frames far too often; thresholding and objective design must be fixed before larger reruns.",
        },
        {
            "rank": 2,
            "id": "teacher_label_quality_too_low",
            "severity": "critical",
            "score": round((1.0 - idm_f1) * 100.0, 4),
            "evidence": {
                "idm_keyboard_accuracy": _get(idm_metrics, "keyboard.accuracy"),
                "idm_mouse_button_f1": _get(idm_metrics, "mouse_button.f1"),
                "idm_mouse_move_pearson": _get(idm_metrics, "mouse_move.pearson"),
                "idm_predicted_button_rate": idm_dist.get("predicted_button_rate"),
            },
            "implication": "FDM training labels come from an IDM whose action quality is far below the renewed IDM target.",
        },
        {
            "rank": 3,
            "id": "renewed_metric_gap_is_order_of_magnitude",
            "severity": "critical",
            "score": float((renewed_gate or {}).get("gate_error_count") or 0),
            "evidence": {
                "renewed_gate_status": (renewed_gate or {}).get("gate_status"),
                "renewed_gate_error_count": (renewed_gate or {}).get("gate_error_count"),
                "keyboard_accuracy": keyboard_gap,
                "mouse_button_f1": _get(fdm_metrics, "mouse_button.f1"),
                "mouse_move_pearson": _get(fdm_metrics, "mouse_move.pearson"),
            },
            "implication": "The old completed FDM is not a weak pass; it fails every renewed aggregate and split hard gate.",
        },
        {
            "rank": 4,
            "id": "global_threshold_not_action_prior_aware",
            "severity": "high",
            "score": 35.0,
            "evidence": {
                "calibration": _get(fdm_metadata, "torch_checkpoint_metadata.calibration") or _get(fdm_metadata, "calibration"),
                "label_source": _get(fdm_metadata, "label_source"),
            },
            "implication": "A single global category threshold is not controlling the highly imbalanced no-button prior.",
        },
        {
            "rank": 5,
            "id": "raw_per_game_confusion_gap",
            "severity": "high" if raw_status != "computed" else "medium",
            "score": 20.0 if raw_status != "computed" else 5.0,
            "evidence": {"raw_diagnostic_status": raw_status},
            "implication": "Per-game confusion must be computed on PVC rows before interpreting heldout-game improvements.",
        },
        {
            "rank": 6,
            "id": "aux_did_not_move_metrics",
            "severity": "medium",
            "score": 10.0 if aux_delta == 0.0 else 5.0,
            "evidence": {
                "aux_minus_d2e_mouse_button_f1": aux_delta,
                "aux_mouse_button_f1": _get(aux_metrics, "mouse_button.f1") if aux_metrics else None,
                "d2e_only_mouse_button_f1": _get(fdm_metrics, "mouse_button.f1"),
            },
            "implication": "The existing aux candidate should be demoted unless a later renewed ablation proves positive D2E heldout movement.",
        },
    ]
    return causes


def build_failure_root_cause_audit(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    paths = {str(key): str(value) for key, value in dict(config.get("paths", {})).items()}
    required = [str(path) for path in config.get("required_artifacts", [])]
    artifacts = {name: _artifact(root_path / path, path) for name, path in paths.items() if path}
    findings: list[dict[str, Any]] = []
    for rel_path in required:
        if not (root_path / rel_path).is_file():
            findings.append({"severity": "error", "code": "missing_required_artifact", "path": rel_path})

    idm_metrics = _load_json(root_path / paths.get("idm_metrics", ""))
    fdm_metrics = _load_json(root_path / paths.get("fdm_metrics", ""))
    aux_metrics = _load_json(root_path / paths.get("aux_metrics", "")) if paths.get("aux_metrics") else None
    idm_metadata = _load_json(root_path / paths.get("idm_metadata", ""))
    fdm_metadata = _load_json(root_path / paths.get("fdm_metadata", ""))
    split_summary = _load_json(root_path / paths.get("fdm_split_summary", ""))
    renewed_gate = _load_json(root_path / paths.get("renewed_gate_audit", ""))
    external_manifest = _load_json(root_path / paths.get("external_artifact_manifest", ""))
    external_by_path = _external_entries(external_manifest)
    raw_diag_path = paths.get("raw_diagnostics")
    raw_diag = _load_json(root_path / raw_diag_path) if raw_diag_path else None

    fdm_model_name = str(config.get("fdm_model_name", "streaming_fdm_d2e_full_compact"))
    fdm_split_rows = _split_rows(root_path, paths.get("fdm_split_stats_summary"), fdm_model_name)
    fdm_split_pivot = _split_pivot(fdm_split_rows)
    target_game_counts = _get(split_summary, "counts.target_games") or {}
    raw_required_paths = [str(path) for path in config.get("raw_required_paths", [])]
    raw_artifacts = {path: _external_proof(external_by_path.get(path)) for path in raw_required_paths}
    expected_raw_rows = int(config.get("expected_raw_rows") or _get(fdm_metrics, "num_examples") or 0)
    raw_diag_rows = int(_get(raw_diag, "alignment.rows_seen") or 0) if raw_diag else 0
    raw_status = "computed" if raw_diag and raw_diag.get("status") == "pass" and raw_diag_rows >= expected_raw_rows else "pvc_required"
    if raw_status != "computed" and not all(item.get("exists") for item in raw_artifacts.values()):
        findings.append({"severity": "error", "code": "raw_artifact_external_proof_missing", "paths": raw_required_paths})

    axes = {
        "action_distribution": _axis(
            "computed",
            {"fdm": _button_distribution(fdm_metrics), "idm": _button_distribution(idm_metrics)},
            summary="Button/no-button imbalance is computed from full eval metrics.",
        ),
        "token_vocabulary": _axis(
            "computed",
            {"fdm": _vocab_counts(fdm_metadata), "idm": _vocab_counts(idm_metadata)},
            summary="Categorical action vocabulary and metric-token compatibility are computed from checkpoint metadata.",
        ),
        "label_prediction_alignment": _axis(
            "computed",
            {
                "label_source": _get(fdm_metadata, "label_source"),
                "idm_metrics": _metric_snapshot(idm_metrics),
                "fdm_metrics": _metric_snapshot(fdm_metrics),
                "fdm_to_idm_no_button_fpr_ratio": _ratio(
                    _get(fdm_metrics, "mouse_button.no_button_false_positive_rate"),
                    _get(idm_metrics, "mouse_button.no_button_false_positive_rate"),
                ),
            },
            summary="FDM is compared against the IDM teacher used for pseudo-labeling.",
        ),
        "no_op_thresholding": _axis(
            "computed",
            {
                "fdm_calibration": _get(fdm_metadata, "torch_checkpoint_metadata.calibration") or _get(fdm_metadata, "calibration"),
                "fdm_no_button_false_positive_rate": _get(fdm_metrics, "mouse_button.no_button_false_positive_rate"),
                "renewed_target": config.get("no_button_false_positive_rate_target", 0.10),
            },
            summary="No-button false-positive control is diagnosed from calibration metadata and renewed gate results.",
        ),
        "heldout_split_confusion": _axis(
            "computed",
            {
                "pivot": fdm_split_pivot,
                "worst_no_button_fpr_split": _worst_split(fdm_split_rows, "no_button_false_positive_rate", lower_is_better=True),
                "worst_mouse_button_f1_split": _worst_split(fdm_split_rows, "mouse_button_f1"),
                "worst_keyboard_split": _worst_split(fdm_split_rows, "keyboard_accuracy"),
            },
            summary="Split-level confusion endpoints are computed from G004 split statistical comparisons.",
        ),
        "per_game_confusion": _axis(
            raw_status,
            {
                "raw_diagnostic_path": raw_diag_path,
                "raw_diagnostic_summary": raw_diag,
                "raw_diagnostic_rows": raw_diag_rows,
                "expected_raw_rows": expected_raw_rows,
                "target_game_counts": target_game_counts,
                "external_raw_artifacts": raw_artifacts,
            },
            summary=(
                "Per-game confusion requires streaming PVC-resident prediction/target rows; local target game counts "
                "and external artifact proofs are recorded until the raw diagnostic artifact is present."
            ),
            severity="warning" if raw_status != "computed" else "info",
        ),
        "oracle_upper_bound_sanity": _axis(
            "inferred",
            {
                "target_records": _get(fdm_metadata, "target_examples") or _get(fdm_metadata, "torch_checkpoint_metadata.target_records"),
                "oracle_ground_truth_control": _get(fdm_metadata, "oracle_ground_truth_control"),
                "current_failure_count": _get(fdm_metrics, "failure_count"),
                "metric_helper_can_score_ground_truth_tokens": True,
            },
            summary="Ground-truth target rows define a trivial oracle upper bound; the current model misses nearly every exact token set.",
        ),
        "d2e_metric_compatibility": _axis(
            "computed",
            {
                "metrics_schema": _get(fdm_metrics, "schema"),
                "mouse_button_metric_prefixes": ["MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"],
                "fdm_vocab_x_button_tokens": _vocab_counts(fdm_metadata).get("x_button_tokens"),
                "mouse_move_scale_ratio": _get(fdm_metrics, "mouse_move.scale_ratio"),
            },
            summary="Metric schema and known token-prefix handling are explicit before comparing against paper/released baselines.",
        ),
        "feature_sufficiency": _axis(
            "computed",
            {
                "fdm_feature_mode": _get(fdm_metadata, "torch_checkpoint_metadata.feature_mode"),
                "fdm_input_dim": _get(fdm_metadata, "torch_checkpoint_metadata.input_dim"),
                "idm_feature_mode": _get(idm_metadata, "feature_mode"),
                "idm_input_dim": _get(idm_metadata, "input_dim"),
            },
            summary="Existing compact summary-grid features have weak action metrics despite full-corpus training.",
        ),
    }

    accepted_axis_statuses = {str(item) for item in config.get("accepted_axis_statuses", ["computed", "inferred", "pvc_required"])}
    for axis_name in config.get("required_axes", []):
        axis = axes.get(str(axis_name))
        if axis is None:
            findings.append({"severity": "error", "code": "missing_required_axis", "axis": str(axis_name)})
        elif axis.get("status") not in accepted_axis_statuses:
            findings.append({"severity": "error", "code": "required_axis_unaccepted_status", "axis": str(axis_name), "status": axis.get("status")})

    ranked = _ranked_causes(
        idm_metrics=idm_metrics,
        fdm_metrics=fdm_metrics,
        aux_metrics=aux_metrics,
        fdm_metadata=fdm_metadata,
        renewed_gate=renewed_gate,
        raw_status=raw_status,
    )
    next_actions = [
        {
            "id": "run_pvc_per_game_confusion",
            "priority": "high" if raw_status != "computed" else "done",
            "action": "Run the G002 raw diagnostic streamer on PVC prediction/target rows before interpreting per-game heldout failures.",
            "blocks": ["per_game_confusion_claims"],
        },
        {
            "id": "idm_teacher_repair",
            "priority": "critical",
            "action": "Do not scale FDM until IDM keyboard/mouse-button/mouse-move quality beats D2E paper/released G-IDM targets.",
            "blocks": ["G005-idm-full-paper-target", "G007-calibrated-pseudolabels"],
        },
        {
            "id": "fdm_false_positive_control",
            "priority": "critical",
            "action": "Use no-button prior losses, asymmetric thresholds, abstention, and per-head calibration in FDM exploration.",
            "blocks": ["G008-fdm-recipe-exploration", "G009-fdm-full-hard-gates"],
        },
    ]
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g002_failure_root_cause_audit.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "findings": findings,
        "artifacts": artifacts,
        "metrics": {
            "idm": _metric_snapshot(idm_metrics),
            "fdm": _metric_snapshot(fdm_metrics),
            "aux": _metric_snapshot(aux_metrics) if aux_metrics else None,
        },
        "axes": axes,
        "ranked_root_causes": ranked,
        "next_actions": next_actions,
        "claim_boundary": (
            "G002 diagnoses why current artifacts fail renewed gates. A pass does not claim the model is good; "
            "it proves the failure modes and raw-artifact gaps are explicitly ranked before new GPU-heavy training."
        ),
    }


def write_failure_root_cause_audit(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_failure_root_cause_audit(config, root=root)
    output = output_path or config.get("output_path")
    if not output:
        raise ValueError("output_path is required")
    write_json(Path(root) / output, payload)
    return payload
