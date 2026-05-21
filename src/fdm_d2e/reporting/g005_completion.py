from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_status(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _jsonl_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _get(data: dict[str, Any] | None, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _assert_json_expectations(
    payload: dict[str, Any] | None,
    expectations: dict[str, Any],
    *,
    source_name: str,
    findings: list[dict[str, Any]],
) -> None:
    for dotted, expected in expectations.items():
        actual = _get(payload, dotted)
        if actual != expected:
            findings.append(
                {
                    "severity": "error",
                    "code": "json_expectation_mismatch",
                    "source": source_name,
                    "json_path": dotted,
                    "expected": expected,
                    "actual": actual,
                }
            )


def validate_g005_aux_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G005-aux-data-best-model"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    if goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})
    prereq_report = {}
    for prereq in config.get("prerequisite_goals", []):
        actual = statuses.get(str(prereq), "missing")
        prereq_report[str(prereq)] = actual
        if actual != "complete":
            findings.append({"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(prereq), "actual": actual})

    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    aux_candidates = _load_json(root_path / paths.get("aux_candidates", "")) if paths.get("aux_candidates") else None
    ablation = _load_json(root_path / paths.get("ablation_summary", "")) if paths.get("ablation_summary") else None
    metadata = _load_json(root_path / paths.get("checkpoint_metadata", "")) if paths.get("checkpoint_metadata") else None
    run_summary = _load_json(root_path / paths.get("run_summary", "")) if paths.get("run_summary") else None

    _assert_json_expectations(aux_candidates, dict(config.get("aux_candidate_expectations", {})), source_name="aux_candidates", findings=findings)
    _assert_json_expectations(ablation, dict(config.get("ablation_expectations", {})), source_name="ablation_summary", findings=findings)
    _assert_json_expectations(metadata, dict(config.get("metadata_expectations", {})), source_name="checkpoint_metadata", findings=findings)

    if aux_candidates is not None:
        selected = [row for row in aux_candidates.get("candidates", []) if row.get("selection_status") == "selected_candidate"]
        if not selected:
            findings.append({"severity": "error", "code": "no_selected_aux_candidates"})
        selected_total = float(_get(aux_candidates, "storage_policy.selected_plus_d2e_gib") or 0.0)
        cap = float(_get(aux_candidates, "storage_policy.cap_gib") or 0.0)
        if selected_total and cap and selected_total > cap:
            findings.append({"severity": "error", "code": "aux_storage_over_cap", "selected_plus_d2e_gib": selected_total, "cap_gib": cap})

    required_splits = set(config.get("required_splits", []))
    ablation_splits: set[str] = set()
    if ablation is not None:
        for item in ablation.get("split_results", []) or []:
            if isinstance(item, dict) and item.get("split") is not None:
                ablation_splits.add(str(item.get("split")))
        missing = sorted(required_splits - ablation_splits)
        if missing:
            findings.append({"severity": "error", "code": "ablation_missing_required_splits", "missing": missing})
        if bool(config.get("require_d2e_only_baseline", True)) and not bool(ablation.get("d2e_only_baseline_present", False)):
            findings.append({"severity": "error", "code": "ablation_missing_d2e_only_baseline"})
        if bool(config.get("require_d2e_aux_candidate", True)) and not bool(ablation.get("d2e_aux_candidate_present", False)):
            findings.append({"severity": "error", "code": "ablation_missing_d2e_aux_candidate"})

    if metadata is not None:
        aux_sources = metadata.get("aux_sources") or metadata.get("source_aux_datasets") or []
        if not isinstance(aux_sources, list) or not aux_sources:
            findings.append({"severity": "error", "code": "metadata_missing_aux_sources"})
        required_tags = set(config.get("required_target_eval_split_tags", []))
        actual_tags = set(str(tag) for tag in metadata.get("target_eval_split_tags", []) or [])
        missing_tags = sorted(required_tags - actual_tags)
        if missing_tags:
            findings.append({"severity": "error", "code": "metadata_missing_target_eval_split_tags", "missing": missing_tags})

    if run_summary is not None:
        if run_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
        expected_gpus = int(config.get("expected_gpus", 4))
        if int(run_summary.get("expected_gpus", -1)) != expected_gpus:
            findings.append({"severity": "error", "code": "run_summary_expected_gpus_mismatch", "expected": expected_gpus, "actual": run_summary.get("expected_gpus")})

    target_count = _jsonl_count(root_path / paths.get("target_records", "")) if paths.get("target_records") else None
    pred_count = _jsonl_count(root_path / paths.get("predictions", "")) if paths.get("predictions") else None
    count_report = {"target_records": target_count, "predictions": pred_count}
    if target_count is not None and pred_count is not None and pred_count != target_count:
        findings.append({"severity": "error", "code": "predictions_count_mismatch", "expected": target_count, "actual": pred_count})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g005_aux_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "prerequisite_goal_statuses": prereq_report,
        "required_splits": sorted(required_splits),
        "ablation_splits": sorted(ablation_splits),
        "artifacts": artifacts,
        "counts": count_report,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G005 complete; it proves D2E-only prerequisites, aux provenance/storage policy, separated namespaces, D2E-only vs D2E+aux ablations, target split tags, and run evidence.",
    }


def write_g005_aux_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g005_aux_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
