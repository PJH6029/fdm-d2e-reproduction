from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from fdm_d2e.data.full_corpus import included_universe_rows
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


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _expected_count_mismatches(actual: dict[str, int], expected: dict[str, Any], *, code: str) -> list[dict[str, Any]]:
    findings = []
    for key, raw_expected in sorted(expected.items()):
        try:
            expected_count = int(raw_expected)
        except (TypeError, ValueError):
            findings.append({"severity": "error", "code": f"{code}_invalid_expected", "key": key, "expected": raw_expected})
            continue
        actual_count = actual.get(str(key))
        if actual_count != expected_count:
            findings.append({"severity": "error", "code": code, "key": str(key), "expected": expected_count, "actual": actual_count})
    return findings


def _missing_required_values(values: Any, required: set[str]) -> list[str]:
    if not required:
        return []
    if isinstance(values, dict):
        actual = {str(key) for key in values}
    elif isinstance(values, (list, tuple, set)):
        actual = {str(value) for value in values}
    else:
        actual = set()
    return sorted(required - actual)


def _assert_required_values(
    source: dict[str, Any] | None,
    dotted: str,
    required: set[str],
    *,
    code: str,
    findings: list[dict[str, Any]],
) -> None:
    missing = _missing_required_values(_get(source, dotted), required)
    if missing:
        findings.append({"severity": "error", "code": code, "json_path": dotted, "missing": missing})


def _artifact_path_matches(
    metadata: dict[str, Any] | None,
    dotted: str,
    expected_path: str | None,
    *,
    code: str,
    findings: list[dict[str, Any]],
) -> None:
    if metadata is None or not expected_path:
        return
    actual = _get(metadata, dotted)
    if actual != expected_path:
        findings.append({"severity": "error", "code": code, "json_path": dotted, "expected": expected_path, "actual": actual})


def _gpu_monitor_status(path: Path, expected_gpus: int) -> dict[str, Any]:
    status: dict[str, Any] = {
        "rows": 0,
        "unique_gpu_indices": [],
        "expected_gpus": expected_gpus,
        "covers_expected_gpus": False,
    }
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return status
    index_col: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]
            if not row:
                continue
            lowered = [cell.lower() for cell in row]
            if "index" in lowered:
                index_col = lowered.index("index")
                continue
            if index_col is None:
                # nvidia-smi --query-gpu output without a header normally uses timestamp,index,...
                index_col = 1 if len(row) > 1 else 0
            if index_col < len(row):
                status["unique_gpu_indices"].append(row[index_col])
            status["rows"] += 1
    status["unique_gpu_indices"] = sorted(set(status["unique_gpu_indices"]))
    status["covers_expected_gpus"] = len(status["unique_gpu_indices"]) >= expected_gpus
    return status


def validate_g004_full_fdm_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G004-d2e-only-fdm-4xh200"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    require_goal_checkpoint = bool(config.get("require_goal_checkpoint_complete", True))
    if require_goal_checkpoint and goal_status != "complete":
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

    split_summary = _load_json(root_path / paths.get("split_summary", "")) if paths.get("split_summary") else None
    metadata = _load_json(root_path / paths.get("checkpoint_metadata", "")) if paths.get("checkpoint_metadata") else None
    source_idm_metadata = _load_json(root_path / paths.get("source_idm_metadata", "")) if paths.get("source_idm_metadata") else None
    data_universe = _load_json(root_path / paths.get("data_universe", "")) if paths.get("data_universe") else None
    g003_audit = _load_json(root_path / paths.get("g003_completion_audit", "")) if paths.get("g003_completion_audit") else None
    run_summary = _load_json(root_path / paths.get("run_summary", "")) if paths.get("run_summary") else None
    split_stats = _load_json(root_path / paths.get("split_stats_summary", "")) if paths.get("split_stats_summary") else None
    convergence = _load_json(root_path / paths.get("convergence_report", "")) if paths.get("convergence_report") else None
    gpu_monitor_path = root_path / paths.get("gpu_monitor", "")
    expected_gpus = int(config.get("expected_gpus", 4))
    gpu_monitor_status = _gpu_monitor_status(gpu_monitor_path, expected_gpus) if paths.get("gpu_monitor") else {"rows": 0, "unique_gpu_indices": [], "expected_gpus": expected_gpus, "covers_expected_gpus": False}

    train_count = _jsonl_count(root_path / paths.get("fdm_train_records", "")) if paths.get("fdm_train_records") else None
    target_count = _jsonl_count(root_path / paths.get("fdm_target_records", "")) if paths.get("fdm_target_records") else None
    prediction_count = _jsonl_count(root_path / paths.get("predictions", "")) if paths.get("predictions") else None
    split_counts = split_summary.get("counts", {}) if isinstance(split_summary, dict) else {}
    count_report = {
        "fdm_train_records": train_count,
        "fdm_target_records": target_count,
        "predictions": prediction_count,
        "split_train": split_counts.get("train"),
        "split_target": split_counts.get("target"),
        "split_pairs": split_counts.get("pairs"),
    }
    if train_count is not None and split_counts.get("train") is not None and train_count != int(split_counts["train"]):
        findings.append({"severity": "error", "code": "fdm_train_count_mismatch", "expected": split_counts.get("train"), "actual": train_count})
    if target_count is not None and split_counts.get("target") is not None and target_count != int(split_counts["target"]):
        findings.append({"severity": "error", "code": "fdm_target_count_mismatch", "expected": split_counts.get("target"), "actual": target_count})
    if prediction_count is not None and target_count is not None and prediction_count != target_count:
        findings.append({"severity": "error", "code": "predictions_count_mismatch", "expected": target_count, "actual": prediction_count})

    expected_variants = int(config.get("expected_recording_variants", 918))
    required_source_ids = {str(value) for value in config.get("required_source_ids", [])}
    required_tiers = {str(value) for value in config.get("required_resolution_tiers", [])}
    expected_by_source = {str(key): value for key, value in dict(config.get("expected_variants_by_source", {})).items()}
    expected_by_tier = {str(key): value for key, value in dict(config.get("expected_variants_by_resolution_tier", {})).items()}
    universe_rows = included_universe_rows(data_universe) if isinstance(data_universe, dict) else []
    universe_counts = {
        "included_recording_variants": len(universe_rows),
        "source_ids": _counts(universe_rows, "source_id"),
        "resolution_tiers": _counts(universe_rows, "resolution_tier"),
    }
    if data_universe is not None:
        if len(universe_rows) != expected_variants:
            findings.append(
                {
                    "severity": "error",
                    "code": "data_universe_included_variants_mismatch",
                    "expected": expected_variants,
                    "actual": len(universe_rows),
                }
            )
        missing_sources = sorted(required_source_ids - set(universe_counts["source_ids"]))
        if missing_sources:
            findings.append({"severity": "error", "code": "data_universe_missing_required_sources", "missing": missing_sources})
        missing_tiers = sorted(required_tiers - set(universe_counts["resolution_tiers"]))
        if missing_tiers:
            findings.append({"severity": "error", "code": "data_universe_missing_required_resolution_tiers", "missing": missing_tiers})
        findings.extend(_expected_count_mismatches(universe_counts["source_ids"], expected_by_source, code="data_universe_source_count_mismatch"))
        findings.extend(_expected_count_mismatches(universe_counts["resolution_tiers"], expected_by_tier, code="data_universe_resolution_tier_count_mismatch"))
        declared_required_sources = set(str(value) for value in _get(data_universe, "decision_gates.full_success_requires_sources") or [])
        if required_source_ids and declared_required_sources != required_source_ids:
            findings.append(
                {
                    "severity": "error",
                    "code": "data_universe_full_success_sources_mismatch",
                    "expected": sorted(required_source_ids),
                    "actual": sorted(declared_required_sources),
                }
            )

    if bool(config.get("require_g003_completion_audit_pass", True)):
        if g003_audit is None:
            findings.append({"severity": "error", "code": "missing_g003_completion_audit", "path": paths.get("g003_completion_audit")})
        elif g003_audit.get("status") != "pass":
            findings.append(
                {
                    "severity": "error",
                    "code": "g003_completion_audit_not_pass",
                    "status": g003_audit.get("status"),
                    "error_count": g003_audit.get("error_count"),
                }
            )
    if g003_audit is not None:
        if int(g003_audit.get("expected_recording_variants", -1)) != expected_variants:
            findings.append(
                {
                    "severity": "error",
                    "code": "g003_expected_recording_variants_mismatch",
                    "expected": expected_variants,
                    "actual": g003_audit.get("expected_recording_variants"),
                }
            )
        g003_sources = _get(g003_audit, "data_universe_counts.source_ids")
        g003_tiers = _get(g003_audit, "data_universe_counts.resolution_tiers")
        findings.extend(_expected_count_mismatches(g003_sources if isinstance(g003_sources, dict) else {}, expected_by_source, code="g003_data_universe_source_count_mismatch"))
        findings.extend(_expected_count_mismatches(g003_tiers if isinstance(g003_tiers, dict) else {}, expected_by_tier, code="g003_data_universe_resolution_tier_count_mismatch"))

    metadata_expectations = dict(config.get("metadata_expectations", {}))
    for dotted, expected in metadata_expectations.items():
        actual = _get(metadata, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "metadata_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    source_idm_metadata_expectations = dict(config.get("source_idm_metadata_expectations", {}))
    for dotted, expected in source_idm_metadata_expectations.items():
        actual = _get(source_idm_metadata, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "source_idm_metadata_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    split_summary_expectations = dict(config.get("split_summary_expectations", {}))
    for dotted, expected in split_summary_expectations.items():
        actual = _get(split_summary, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "split_summary_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    resolved_config = _load_json(root_path / paths.get("resolved_config", "")) if paths.get("resolved_config") else None
    resolved_config_expectations = dict(config.get("resolved_config_expectations", {}))
    for dotted, expected in resolved_config_expectations.items():
        actual = _get(resolved_config, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "resolved_config_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    if metadata is not None:
        if train_count is not None and metadata.get("num_training_examples") != train_count:
            findings.append({"severity": "error", "code": "metadata_train_examples_mismatch", "expected": train_count, "actual": metadata.get("num_training_examples")})
        if target_count is not None and metadata.get("target_examples") != target_count:
            findings.append({"severity": "error", "code": "metadata_target_examples_mismatch", "expected": target_count, "actual": metadata.get("target_examples")})
        required_tags = set(config.get("required_target_eval_split_tags", []))
        actual_tags = set(str(tag) for tag in metadata.get("target_eval_split_tags", []) or [])
        missing_tags = sorted(required_tags - actual_tags)
        if missing_tags:
            findings.append({"severity": "error", "code": "metadata_missing_target_eval_split_tags", "missing": missing_tags})
        expected_train_path = paths.get("fdm_train_records")
        expected_target_path = paths.get("fdm_target_records")
        if expected_train_path and metadata.get("train_records_path") != expected_train_path:
            findings.append({"severity": "error", "code": "metadata_train_records_path_mismatch", "expected": expected_train_path, "actual": metadata.get("train_records_path")})
        if expected_target_path and metadata.get("target_records_path") != expected_target_path:
            findings.append({"severity": "error", "code": "metadata_target_records_path_mismatch", "expected": expected_target_path, "actual": metadata.get("target_records_path")})
        _artifact_path_matches(metadata, "source_idm_metadata.path", paths.get("source_idm_metadata"), code="metadata_source_idm_metadata_path_mismatch", findings=findings)
        _artifact_path_matches(metadata, "data_universe.path", paths.get("data_universe"), code="metadata_data_universe_path_mismatch", findings=findings)
        _artifact_path_matches(metadata, "split_contract.path", paths.get("split_contract"), code="metadata_split_contract_path_mismatch", findings=findings)
        _assert_required_values(metadata, "source_ids", required_source_ids, code="metadata_missing_required_source_ids", findings=findings)
        _assert_required_values(metadata, "target_source_ids", required_source_ids, code="metadata_missing_required_target_source_ids", findings=findings)
        _assert_required_values(metadata, "resolution_tiers", required_tiers, code="metadata_missing_required_resolution_tiers", findings=findings)
        _assert_required_values(metadata, "target_resolution_tiers", required_tiers, code="metadata_missing_required_target_resolution_tiers", findings=findings)

    if source_idm_metadata is not None:
        _assert_required_values(source_idm_metadata, "source_ids", required_source_ids, code="source_idm_metadata_missing_required_source_ids", findings=findings)
        _assert_required_values(source_idm_metadata, "target_source_ids", required_source_ids, code="source_idm_metadata_missing_required_target_source_ids", findings=findings)
        _assert_required_values(source_idm_metadata, "resolution_tiers", required_tiers, code="source_idm_metadata_missing_required_resolution_tiers", findings=findings)
        _assert_required_values(source_idm_metadata, "target_resolution_tiers", required_tiers, code="source_idm_metadata_missing_required_target_resolution_tiers", findings=findings)

    if split_summary is not None:
        _assert_required_values(split_summary, "counts.source_ids", required_source_ids, code="split_summary_missing_required_source_ids", findings=findings)
        _assert_required_values(split_summary, "counts.target_source_ids", required_source_ids, code="split_summary_missing_required_target_source_ids", findings=findings)
        _assert_required_values(split_summary, "counts.resolution_tiers", required_tiers, code="split_summary_missing_required_resolution_tiers", findings=findings)
        _assert_required_values(split_summary, "counts.target_resolution_tiers", required_tiers, code="split_summary_missing_required_target_resolution_tiers", findings=findings)

    if run_summary is not None:
        if run_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
        if int(run_summary.get("nproc_per_node", -1)) != int(config.get("expected_nproc_per_node", 4)):
            findings.append({"severity": "error", "code": "run_summary_nproc_mismatch", "expected": int(config.get("expected_nproc_per_node", 4)), "actual": run_summary.get("nproc_per_node")})
        if int(run_summary.get("expected_gpus", -1)) != int(config.get("expected_gpus", 4)):
            findings.append({"severity": "error", "code": "run_summary_expected_gpus_mismatch", "expected": int(config.get("expected_gpus", 4)), "actual": run_summary.get("expected_gpus")})
        run_gpu_status = run_summary.get("gpu_monitor_status")
        if isinstance(run_gpu_status, dict) and bool(config.get("require_gpu_monitor_covers_expected_gpus", True)) and run_gpu_status.get("covers_expected_gpus") is not True:
            findings.append({"severity": "error", "code": "run_summary_gpu_monitor_missing_expected_gpus", "actual": run_gpu_status})
    min_gpu_rows = int(config.get("min_gpu_monitor_rows", expected_gpus))
    if gpu_monitor_status["rows"] < min_gpu_rows:
        findings.append({"severity": "error", "code": "gpu_monitor_too_few_rows", "expected_min": min_gpu_rows, "actual": gpu_monitor_status["rows"]})
    if bool(config.get("require_gpu_monitor_covers_expected_gpus", True)) and not gpu_monitor_status["covers_expected_gpus"]:
        findings.append(
            {
                "severity": "error",
                "code": "gpu_monitor_does_not_cover_expected_gpus",
                "expected_gpus": expected_gpus,
                "unique_gpu_indices": gpu_monitor_status["unique_gpu_indices"],
                "rows": gpu_monitor_status["rows"],
            }
        )
    if split_stats is not None and split_stats.get("status") != "pass":
        findings.append({"severity": "error", "code": "split_stats_summary_not_pass", "actual": split_stats.get("status")})
    if convergence is not None:
        min_validation = int(config.get("min_validation_checkpoints", 1))
        actual_validation = int(convergence.get("num_validation_checkpoints", 0))
        if actual_validation < min_validation:
            findings.append({"severity": "error", "code": "convergence_validation_checkpoints_too_low", "expected_min": min_validation, "actual": actual_validation})
        if bool(config.get("require_convergence_plateau", False)) and not bool(convergence.get("plateau_met", False)):
            findings.append({"severity": "error", "code": "convergence_plateau_not_met", "actual": convergence.get("plateau_met")})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g004_full_fdm_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "prerequisite_goal_statuses": prereq_report,
        "expected_nproc_per_node": int(config.get("expected_nproc_per_node", 4)),
        "expected_gpus": int(config.get("expected_gpus", 4)),
        "artifacts": artifacts,
        "counts": count_report,
        "data_universe_counts": universe_counts,
        "gpu_monitor_status": gpu_monitor_status,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G004 complete; it proves D2E-only FDM-from-G003-IDM-pseudolabel provenance, full D2E source/tier coverage, split counts, prediction coverage, split stats, convergence evidence, and 4xH200 run evidence.",
    }


def write_g004_full_fdm_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g004_full_fdm_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
