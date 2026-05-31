from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_status(path: Path, rel_path: str, *, hash_file: bool = True) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    payload = {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": None}
    if hash_file:
        payload["sha256"] = sha256_file(path)
    else:
        payload["hash_omitted_reason"] = "large_artifact_hash_recorded_in_dataset_summary_output_hashes"
    return payload


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _find_goal_status(statuses: dict[str, str], goal_id: str) -> str:
    if goal_id in statuses:
        return statuses[goal_id]
    matches = [status for key, status in statuses.items() if key.startswith(goal_id)]
    return matches[0] if matches else "missing"


def _error(findings: list[dict[str, Any]], code: str, **fields: Any) -> None:
    findings.append({"severity": "error", "code": code, **fields})


def _expect_file(artifacts: dict[str, dict[str, Any]], findings: list[dict[str, Any]], key: str) -> None:
    item = artifacts.get(key, {})
    if not item.get("exists"):
        _error(findings, "missing_required_artifact", artifact_key=key, path=item.get("path"))
    elif int(item.get("bytes", 0)) <= 0:
        _error(findings, "empty_required_artifact", artifact_key=key, path=item.get("path"))


def _strictly_increasing(values: list[Any]) -> bool:
    try:
        nums = [float(v) for v in values]
    except Exception:
        return False
    return len(nums) >= 2 and all(a < b for a, b in zip(nums, nums[1:]))


def validate_fdm1_g003_action_dataset_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    omit_hash_keys = set(map(str, config.get("omit_sha256_artifact_keys", [])))
    artifacts = {key: _file_status(root_path / rel_path, rel_path, hash_file=key not in omit_hash_keys) for key, rel_path in paths.items()}
    for key in config.get("required_artifacts", paths.keys()):
        _expect_file(artifacts, findings, str(key))

    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G003-50ms-action-token-dataset-pipeline"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = _find_goal_status(statuses, goal_id)
    require_goal_checkpoint = bool(config.get("require_goal_checkpoint_complete", False))
    if require_goal_checkpoint and goal_status != "complete":
        _error(findings, "goal_not_checkpointed_complete", goal_id=goal_id, actual=goal_status)

    decode = _load_json(root_path / paths.get("decode_summary", "")) if paths.get("decode_summary") else None
    mouse_bins = _load_json(root_path / paths.get("fitted_mouse_bins", "")) if paths.get("fitted_mouse_bins") else None
    dataset = _load_json(root_path / paths.get("dataset_summary", "")) if paths.get("dataset_summary") else None
    overflow = _load_json(root_path / paths.get("overflow_summary", "")) if paths.get("overflow_summary") else None
    alignment = _load_json(root_path / paths.get("alignment_summary", "")) if paths.get("alignment_summary") else None
    sequence_pack = _load_json(root_path / paths.get("sequence_pack", "")) if paths.get("sequence_pack") else None
    visual = _load_json(root_path / paths.get("visual_alignment_audit", "")) if paths.get("visual_alignment_audit") else None

    expected_variants = int(config.get("expected_recording_variants", 459))
    required_source_ids = set(map(str, config.get("required_source_ids", ["d2e_480p"])))
    required_resolution_tiers = set(map(str, config.get("required_resolution_tiers", ["480p"])))
    decode_counts = (decode or {}).get("counts", {}) if isinstance(decode, dict) else {}
    records_expected = int(decode_counts.get("all", 0) or 0)
    if decode is not None:
        if decode.get("split_mode") != config.get("expected_split_mode", "fdm1-g002"):
            _error(findings, "decode_split_mode_mismatch", expected=config.get("expected_split_mode", "fdm1-g002"), actual=decode.get("split_mode"))
        if int(decode.get("selected_recording_variants", 0) or 0) < expected_variants:
            _error(findings, "decode_too_few_recording_variants", expected_min=expected_variants, actual=decode.get("selected_recording_variants"))
        source_ids = set(map(str, decode.get("source_ids", [])))
        tiers = set(map(str, decode.get("resolution_tiers", [])))
        if source_ids != required_source_ids:
            _error(findings, "decode_source_ids_mismatch", expected=sorted(required_source_ids), actual=sorted(source_ids))
        if tiers != required_resolution_tiers:
            _error(findings, "decode_resolution_tiers_mismatch", expected=sorted(required_resolution_tiers), actual=sorted(tiers))
        if decode.get("failures"):
            _error(findings, "decode_failures_present", failures=decode.get("failures"))
        if records_expected <= 0:
            _error(findings, "decode_zero_records", counts=decode_counts)

    if mouse_bins is not None:
        boundaries = list(mouse_bins.get("positive_boundaries", []))
        if mouse_bins.get("status") != "pass":
            _error(findings, "mouse_bins_status_not_pass", actual=mouse_bins.get("status"))
        if len(boundaries) != 24 or not _strictly_increasing(boundaries):
            _error(findings, "mouse_bins_invalid_boundaries", count=len(boundaries), boundaries=boundaries)
        if int(mouse_bins.get("records_used", 0) or 0) <= 0:
            _error(findings, "mouse_bins_no_training_records", actual=mouse_bins.get("records_used"))
        if int(mouse_bins.get("mouse_events", 0) or 0) <= 0:
            _error(findings, "mouse_bins_no_mouse_events", actual=mouse_bins.get("mouse_events"))

    if dataset is not None:
        if dataset.get("schema") != "fdm1_action_slot_dataset_summary.v1":
            _error(findings, "dataset_summary_schema_mismatch", actual=dataset.get("schema"))
        if dataset.get("streaming") is not True:
            _error(findings, "dataset_not_streaming", actual=dataset.get("streaming"))
        dataset_records = int(dataset.get("records", 0) or 0)
        if records_expected and dataset_records != records_expected:
            _error(findings, "dataset_record_count_mismatch", expected=records_expected, actual=dataset_records)
        split_counts = dataset.get("split_counts", {}) or {}
        if int(split_counts.get("all", 0) or 0) != dataset_records:
            _error(findings, "dataset_all_split_count_mismatch", expected=dataset_records, actual=split_counts.get("all"))
        for split_name in config.get("required_nonzero_splits", ["train_core", "target_all_eval"]):
            if int(split_counts.get(split_name, 0) or 0) <= 0:
                _error(findings, "dataset_required_split_empty", split=split_name, actual=split_counts.get(split_name))
        if int(dataset.get("unique_token_count", 0) or 0) < int(config.get("min_unique_tokens", 16)):
            _error(findings, "dataset_too_few_unique_tokens", expected_min=int(config.get("min_unique_tokens", 16)), actual=dataset.get("unique_token_count"))
        expected_tokenization = config.get("expected_tokenization_config")
        if expected_tokenization and dataset.get("tokenization_config") != expected_tokenization:
            _error(findings, "dataset_tokenization_config_mismatch", expected=expected_tokenization, actual=dataset.get("tokenization_config"))
        output_hashes = dataset.get("output_hashes") or {}
        for role in config.get("required_output_hash_roles", []):
            if not output_hashes.get(str(role)):
                _error(findings, "dataset_missing_output_hash", role=str(role))

    if overflow is not None:
        if records_expected and int(overflow.get("bins", 0) or 0) != records_expected:
            _error(findings, "overflow_bin_count_mismatch", expected=records_expected, actual=overflow.get("bins"))
        if overflow.get("threshold_exceeded") is True:
            _error(findings, "overflow_threshold_exceeded", overflow_rate=overflow.get("overflow_rate"))

    if alignment is not None:
        if alignment.get("status") != "pass":
            _error(findings, "alignment_status_not_pass", actual=alignment.get("status"), errors=alignment.get("errors"))
        if records_expected and int(alignment.get("record_count", 0) or 0) != records_expected:
            _error(findings, "alignment_record_count_mismatch", expected=records_expected, actual=alignment.get("record_count"))

    if sequence_pack is not None and dataset is not None:
        if sequence_pack.get("dataset_fingerprint") != dataset.get("dataset_fingerprint"):
            _error(findings, "sequence_pack_fingerprint_mismatch", expected=dataset.get("dataset_fingerprint"), actual=sequence_pack.get("dataset_fingerprint"))

    if visual is not None:
        if visual.get("status") != "pass":
            _error(findings, "visual_alignment_status_not_pass", actual=visual.get("status"), errors=visual.get("errors"))
        if int(visual.get("row_count", 0) or 0) < int(config.get("min_visual_rows", 8)):
            _error(findings, "visual_alignment_too_few_rows", expected_min=int(config.get("min_visual_rows", 8)), actual=visual.get("row_count"))

    visual_report_path = root_path / paths.get("visual_alignment_report", "") if paths.get("visual_alignment_report") else None
    if visual_report_path and visual_report_path.exists():
        text = visual_report_path.read_text(encoding="utf-8")
        for phrase in config.get("required_visual_report_phrases", ["FDM-1 action-slot alignment visual check", "MOUSE_MOVE_BIN"]):
            if str(phrase) not in text:
                _error(findings, "visual_report_missing_phrase", phrase=str(phrase))

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "fdm1_g003_action_dataset_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "artifacts": artifacts,
        "counts": {"decode_records": records_expected, "dataset_records": (dataset or {}).get("records") if isinstance(dataset, dict) else None},
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit covers G003 action-token dataset pipeline evidence only; it does not prove IDM/FDM training, metric wins, harness control, or FDM-1 parity.",
    }


def write_fdm1_g003_action_dataset_completion_audit(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = validate_fdm1_g003_action_dataset_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload


__all__ = ["validate_fdm1_g003_action_dataset_completion", "write_fdm1_g003_action_dataset_completion_audit"]
