from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _file_status(root: Path, path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {"path": path_text, "exists": False, "bytes": 0, "sha256": None}
    path = root / path_text
    if not path.exists() or not path.is_file():
        return {"path": path_text, "exists": False, "bytes": 0, "sha256": None}
    return {"path": path_text, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row.get(name)
    return None


def _required_field_findings(row: dict[str, Any], required_fields: list[str], *, index: int) -> list[dict[str, Any]]:
    missing = [field for field in required_fields if row.get(field) is None]
    if not missing:
        return []
    return [{"severity": "error", "code": "comparison_missing_required_fields", "index": index, "missing": missing}]


def _normalise_comparison(row: dict[str, Any], *, source: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    split = row.get("split") or row.get("eval_split") or row.get("split_name") or source.get("split")
    model = row.get("model") or source.get("model") or source.get("model_namespace")
    reference = row.get("reference") or row.get("reference_baseline") or source.get("reference") or source.get("reference_baseline")
    candidate_value = _first_present(row, ["candidate_value", "candidate_mean", "candidate_metric", "model_value"])
    baseline_value = _first_present(row, ["baseline_value", "reference_value", "reference_mean", "baseline_metric"])
    return {
        "split": split,
        "endpoint": row.get("endpoint") or row.get("metric") or row.get("metric_name"),
        "model": model,
        "reference": reference,
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "delta": row.get("delta"),
        "p_value": row.get("p_value"),
        "p_adjusted_holm": row.get("p_adjusted_holm"),
        "reject_holm_0_05": row.get("reject_holm_0_05"),
        "direction": row.get("direction"),
        "num_clusters": row.get("num_clusters"),
        "status": row.get("status"),
        "artifact_path": artifact["path"],
        "artifact_sha256": artifact["sha256"],
        "source_id": source.get("id") or source.get("path"),
    }


def build_final_endpoint_statistics(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    statuses = _goal_statuses(root_path, goals_path)
    findings: list[dict[str, Any]] = []
    for goal_id in config.get("prerequisite_goals", []):
        if statuses.get(goal_id) != "complete":
            findings.append({"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": goal_id, "actual": statuses.get(goal_id, "missing")})

    comparisons: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for source in config.get("comparison_sources", []):
        path_text = str(source.get("path", ""))
        artifact = _file_status(root_path, path_text)
        source_report = {"source": source, "artifact": artifact, "comparisons": 0}
        source_reports.append(source_report)
        if not artifact["exists"]:
            findings.append({"severity": "error", "code": "missing_comparison_source", "path": path_text})
            continue
        payload = _load_json(root_path / path_text) or {}
        rows = payload.get("comparisons") or payload.get("endpoint_tables") or []
        if not isinstance(rows, list) or not rows:
            findings.append({"severity": "error", "code": "comparison_source_has_no_rows", "path": path_text})
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            comparisons.append(_normalise_comparison(row, source=source, artifact=artifact))
        source_report["comparisons"] = len(rows)

    required_splits = set(config.get("required_splits", []))
    required_endpoints = set(config.get("required_endpoints", []))
    seen_splits = {str(row.get("split")) for row in comparisons if row.get("split") is not None}
    seen_endpoints = {str(row.get("endpoint")) for row in comparisons if row.get("endpoint") is not None}
    missing_splits = sorted(required_splits - seen_splits)
    missing_endpoints = sorted(required_endpoints - seen_endpoints)
    if missing_splits:
        findings.append({"severity": "error", "code": "missing_required_splits", "missing": missing_splits})
    if missing_endpoints:
        findings.append({"severity": "error", "code": "missing_required_endpoints", "missing": missing_endpoints})

    required_fields = list(config.get("required_comparison_fields", []))
    for idx, row in enumerate(comparisons):
        findings.extend(_required_field_findings(row, required_fields, index=idx))

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "final_endpoint_statistics.v1",
        "status": "pass" if not errors else "fail",
        "goals_path": goals_path,
        "prerequisite_goal_statuses": {goal_id: statuses.get(goal_id, "missing") for goal_id in config.get("prerequisite_goals", [])},
        "required_splits": sorted(required_splits),
        "required_endpoints": sorted(required_endpoints),
        "comparisons": comparisons,
        "comparison_sources": source_reports,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Endpoint statistics pass only when split-aware G003/G004 comparisons cover all required D2E-only splits/endpoints with artifact hashes.",
    }


def _endpoint_action(endpoint: str | None) -> str:
    if not endpoint:
        return "unknown"
    if endpoint.startswith("keyboard"):
        return "keyboard"
    if endpoint.startswith("mouse_button") or endpoint.startswith("no_button"):
        return "mouse_button"
    if endpoint.startswith("mouse_move"):
        return "mouse_move"
    return str(endpoint).split("_", 1)[0]


def _collect_metadata_axis(metadata: dict[str, Any] | None, key: str) -> list[str]:
    if not metadata:
        return []
    value = metadata.get(key)
    if isinstance(value, list):
        return sorted({str(item) for item in value})
    if isinstance(value, dict):
        return sorted({str(item) for item in value.keys()})
    if value is not None:
        return [str(value)]
    return []


def build_final_failure_analysis(config: dict[str, Any], endpoint_statistics: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    if endpoint_statistics.get("status") != "pass":
        findings.append({"severity": "error", "code": "endpoint_statistics_not_pass", "actual": endpoint_statistics.get("status")})
    comparisons = [row for row in endpoint_statistics.get("comparisons", []) if isinstance(row, dict)]
    non_rejections = [row for row in comparisons if row.get("reject_holm_0_05") is not True]
    negative_examples = sorted(
        non_rejections,
        key=lambda row: (float(row.get("p_adjusted_holm") or 1.0), str(row.get("endpoint"))),
    )[: int(config.get("max_failure_examples", 20))]

    metadata_reports = []
    games: set[str] = set()
    resolutions: set[str] = set()
    sources: set[str] = set()
    calibrations: set[str] = set()
    for path_text in config.get("metadata_sources", []):
        artifact = _file_status(root_path, str(path_text))
        metadata = _load_json(root_path / str(path_text)) if artifact["exists"] else None
        metadata_reports.append({"path": str(path_text), "artifact": artifact})
        games.update(_collect_metadata_axis(metadata, "target_games"))
        resolutions.update(_collect_metadata_axis(metadata, "target_resolution_tiers"))
        sources.update(_collect_metadata_axis(metadata, "target_source_ids"))
        sources.update(_collect_metadata_axis(metadata, "source_ids"))
        if metadata and isinstance(metadata.get("calibration"), dict):
            if metadata["calibration"].get("mode") is not None:
                calibrations.add(str(metadata["calibration"]["mode"]))
        if metadata and metadata.get("label_source") is not None:
            calibrations.add(f"label_source:{metadata['label_source']}")

    action_counts: dict[str, int] = {}
    for row in comparisons:
        action = _endpoint_action(row.get("endpoint"))
        action_counts[action] = action_counts.get(action, 0) + 1
    axes = {
        "action": sorted(action_counts),
        "game": sorted(games) or ["pending_from_full_corpus_metadata"],
        "resolution": sorted(resolutions) or ["pending_from_full_corpus_metadata"],
        "source": sorted(sources) or ["pending_from_full_corpus_metadata"],
        "calibration": sorted(calibrations) or ["pending_from_checkpoint_metadata"],
    }
    required_axes = set(config.get("required_failure_axes", []))
    missing_axes = sorted(axis for axis in required_axes if not axes.get(axis))
    if missing_axes:
        findings.append({"severity": "error", "code": "missing_failure_axes", "missing": missing_axes})
    if config.get("require_non_rejections", True) and not non_rejections:
        findings.append({"severity": "error", "code": "missing_non_rejections"})
    if config.get("require_examples", True) and not negative_examples:
        findings.append({"severity": "error", "code": "missing_failure_examples"})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "final_failure_analysis.v1",
        "status": "pass" if not errors else "fail",
        "axes": axes,
        "axis_counts": {"action": action_counts},
        "non_rejections": non_rejections,
        "examples": negative_examples,
        "metadata_sources": metadata_reports,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Failure analysis must explicitly report non-rejections and axes before final research claims.",
    }


def build_final_claim_taxonomy(config: dict[str, Any], endpoint_statistics: dict[str, Any], failure_analysis: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    statuses = _goal_statuses(root_path, str(config.get("goals_path", ".omx/ultragoal/goals.json")))
    claim_cfg = dict(config.get("claim_taxonomy", {}))
    required = list(config.get("required_claim_taxonomy", []))
    evidence_paths = {str(path) for path in claim_cfg.get("evidence_paths", []) if (root_path / str(path)).exists()}
    findings: list[dict[str, Any]] = []
    claims = []
    for claim_id in required:
        if claim_id == "d2e_only_idm":
            state = "claimable" if statuses.get("G003-d2e-only-idm") == "complete" else "pending_prerequisite"
            evidence = [path for path in evidence_paths if "idm" in path]
        elif claim_id == "d2e_only_fdm":
            state = "claimable" if statuses.get("G004-d2e-only-fdm-4xh200") == "complete" else "pending_prerequisite"
            evidence = [path for path in evidence_paths if "fdm" in path]
        elif claim_id == "d2e_aux_comparison":
            state = "claimable" if statuses.get("G005-aux-data-best-model") == "complete" else "not_claimed_until_g005"
            evidence = [path for path in evidence_paths if "aux" in path]
        elif claim_id == "live_open_game_suite":
            state = "claimable" if statuses.get("G008-live-game-suite") == "complete" else "not_claimed_until_g008"
            evidence = [path for path in evidence_paths if "harness" in path or "live" in path]
        elif claim_id == "negative_results":
            state = "documented" if failure_analysis.get("non_rejections") else "pending_non_rejections"
            evidence = [str(config.get("failure_analysis_path", "artifacts/eval/final_failure_analysis.json"))]
        else:
            state = "tracked"
            evidence = []
        claims.append(
            {
                "id": claim_id,
                "state": state,
                "evidence_paths": evidence,
                "wording_boundary": _claim_boundary_text(claim_id),
            }
        )
    if endpoint_statistics.get("status") != "pass":
        findings.append({"severity": "error", "code": "endpoint_statistics_not_pass", "actual": endpoint_statistics.get("status")})
    if failure_analysis.get("status") != "pass":
        findings.append({"severity": "error", "code": "failure_analysis_not_pass", "actual": failure_analysis.get("status")})
    missing_claims = sorted(set(required) - {claim["id"] for claim in claims})
    if missing_claims:
        findings.append({"severity": "error", "code": "missing_required_claims", "missing": missing_claims})
    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "final_claim_taxonomy.v1",
        "status": "pass" if not errors else "fail",
        "claims": claims,
        "forbidden_claims": ["fdm1_parity", "commercial_game_control_without_live_open_suite", "robotics_transfer", "car_control_transfer"],
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "A pass means claim categories and wording boundaries are explicit; it does not override individual G005/G008 evidence gates.",
    }


def _claim_boundary_text(claim_id: str) -> str:
    boundaries = {
        "d2e_only_idm": "Report only D2E-only IDM metrics/splits supported by G003 artifacts.",
        "d2e_only_fdm": "Report only D2E-only FDM-from-IDM-pseudolabel results supported by G004 artifacts.",
        "d2e_aux_comparison": "D2E+aux may be primary only after D2E-only gates and explicit D2E-only vs D2E+aux ablation.",
        "live_open_game_suite": "Live claims require open-source/offline graphical-game evidence; no commercial-game control claim.",
        "negative_results": "Non-rejections and failure modes must be reported, not hidden.",
    }
    return boundaries.get(claim_id, "Claim must be tied to configured artifact evidence.")


def build_g006_final_artifacts(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    endpoint = build_final_endpoint_statistics(config, root=root_path)
    failure = build_final_failure_analysis(config, endpoint, root=root_path)
    taxonomy = build_final_claim_taxonomy(config, endpoint, failure, root=root_path)
    write_json(root_path / str(config.get("endpoint_statistics_path", "artifacts/eval/final_endpoint_statistics.json")), endpoint)
    write_json(root_path / str(config.get("failure_analysis_path", "artifacts/eval/final_failure_analysis.json")), failure)
    write_json(root_path / str(config.get("claim_taxonomy_path", "artifacts/eval/final_claim_taxonomy.json")), taxonomy)
    statuses = {
        "endpoint_statistics": endpoint["status"],
        "failure_analysis": failure["status"],
        "claim_taxonomy": taxonomy["status"],
    }
    return {
        "schema": "g006_final_artifact_build.v1",
        "status": "pass" if all(value == "pass" for value in statuses.values()) else "fail",
        "statuses": statuses,
        "outputs": {
            "endpoint_statistics": str(config.get("endpoint_statistics_path", "artifacts/eval/final_endpoint_statistics.json")),
            "failure_analysis": str(config.get("failure_analysis_path", "artifacts/eval/final_failure_analysis.json")),
            "claim_taxonomy": str(config.get("claim_taxonomy_path", "artifacts/eval/final_claim_taxonomy.json")),
        },
        "claim_boundary": "Builder output is final G006 evidence only when all statuses pass and prerequisite G003/G004 goals are complete.",
    }
