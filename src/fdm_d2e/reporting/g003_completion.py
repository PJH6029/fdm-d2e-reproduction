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


def _goal_status(root: Path, goals_path: str, goal_id: str) -> str:
    payload = _load_json(root / goals_path) or {}
    for goal in payload.get("goals", []):
        if str(goal.get("id")) == goal_id:
            return str(goal.get("status"))
    return "missing"


def validate_g003_full_idm_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G003-d2e-only-idm"))
    goal_status = _goal_status(root_path, goals_path, goal_id)
    if goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})

    expected_variants = int(config.get("expected_recording_variants", 918))
    expected_shards = int(config.get("expected_shards", 16))
    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    decode = _load_json(root_path / paths.get("decode_summary", "")) if paths.get("decode_summary") else None
    metadata = _load_json(root_path / paths.get("checkpoint_metadata", "")) if paths.get("checkpoint_metadata") else None
    run_summary = _load_json(root_path / paths.get("run_summary", "")) if paths.get("run_summary") else None
    split_stats = _load_json(root_path / paths.get("split_stats_summary", "")) if paths.get("split_stats_summary") else None

    if decode is not None:
        selected = int(decode.get("selected_recording_variants", -1))
        if selected != expected_variants:
            findings.append({"severity": "error", "code": "decode_selected_variants_mismatch", "expected": expected_variants, "actual": selected})
        if int(decode.get("num_shards", -1)) != expected_shards:
            findings.append({"severity": "error", "code": "decode_num_shards_mismatch", "expected": expected_shards, "actual": decode.get("num_shards")})
        failures = decode.get("failures") or []
        if failures:
            findings.append({"severity": "error", "code": "decode_failures_present", "count": len(failures)})
    counts = decode.get("counts", {}) if isinstance(decode, dict) else {}
    train_count = _jsonl_count(root_path / paths.get("train_records", "")) if paths.get("train_records") else None
    target_count = _jsonl_count(root_path / paths.get("target_records", "")) if paths.get("target_records") else None
    pseudo_count = _jsonl_count(root_path / paths.get("pseudolabels", "")) if paths.get("pseudolabels") else None
    pred_count = _jsonl_count(root_path / paths.get("predictions", "")) if paths.get("predictions") else None
    count_report = {
        "train_records": train_count,
        "target_records": target_count,
        "pseudolabels": pseudo_count,
        "predictions": pred_count,
        "decode_train_core": counts.get("train_core"),
        "decode_target_all_eval": counts.get("target_all_eval"),
    }
    if train_count is not None and counts.get("train_core") is not None and train_count != int(counts["train_core"]):
        findings.append({"severity": "error", "code": "train_count_mismatch", "expected": counts.get("train_core"), "actual": train_count})
    if target_count is not None and counts.get("target_all_eval") is not None and target_count != int(counts["target_all_eval"]):
        findings.append({"severity": "error", "code": "target_count_mismatch", "expected": counts.get("target_all_eval"), "actual": target_count})
    for name, value in {"pseudolabels": pseudo_count, "predictions": pred_count}.items():
        if value is not None and target_count is not None and value != target_count:
            findings.append({"severity": "error", "code": f"{name}_count_mismatch", "expected": target_count, "actual": value})

    metadata_expectations = dict(config.get("metadata_expectations", {}))
    for dotted, expected in metadata_expectations.items():
        actual = _get(metadata, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "metadata_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    if metadata is not None:
        if train_count is not None and metadata.get("train_records") != train_count:
            findings.append({"severity": "error", "code": "metadata_train_records_mismatch", "expected": train_count, "actual": metadata.get("train_records")})
        if target_count is not None and metadata.get("target_records") != target_count:
            findings.append({"severity": "error", "code": "metadata_target_records_mismatch", "expected": target_count, "actual": metadata.get("target_records")})
        required_tags = set(config.get("required_target_eval_split_tags", []))
        actual_tags = set(str(tag) for tag in metadata.get("target_eval_split_tags", []) or [])
        missing_tags = sorted(required_tags - actual_tags)
        if missing_tags:
            findings.append({"severity": "error", "code": "metadata_missing_target_eval_split_tags", "missing": missing_tags})

    if run_summary is not None:
        if run_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
        if int(run_summary.get("nproc_per_node", -1)) != int(config.get("expected_nproc_per_node", 4)):
            findings.append({"severity": "error", "code": "run_summary_nproc_mismatch", "expected": int(config.get("expected_nproc_per_node", 4)), "actual": run_summary.get("nproc_per_node")})
    if split_stats is not None and split_stats.get("status") != "pass":
        findings.append({"severity": "error", "code": "split_stats_summary_not_pass", "actual": split_stats.get("status")})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g003_full_idm_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "expected_recording_variants": expected_variants,
        "expected_shards": expected_shards,
        "artifacts": artifacts,
        "counts": count_report,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G003 complete; it proves full-corpus D2E-only IDM coverage, counts, provenance, split stats, and 4xH200 run evidence.",
    }


def write_g003_full_idm_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g003_full_idm_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
