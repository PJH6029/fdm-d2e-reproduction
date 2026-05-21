from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.data.full_corpus import included_universe_rows, universe_row_id
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


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _recording_counts(decode: dict[str, Any] | None, key: str) -> dict[str, int]:
    if not isinstance(decode, dict):
        return {}
    recordings = [row for row in decode.get("recordings", []) if isinstance(row, dict)]
    if recordings:
        return _counts(recordings, key)
    values = decode.get(f"{key}s", []) if key in {"source_id", "resolution_tier"} else []
    return {str(value): -1 for value in values}


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


def validate_g003_full_idm_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G003-d2e-only-idm"))
    goal_status = _goal_status(root_path, goals_path, goal_id)
    require_goal_checkpoint = bool(config.get("require_goal_checkpoint_complete", True))
    if require_goal_checkpoint and goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})

    expected_variants = int(config.get("expected_recording_variants", 918))
    expected_shards = int(config.get("expected_shards", 16))
    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    decode = _load_json(root_path / paths.get("decode_summary", "")) if paths.get("decode_summary") else None
    data_universe = _load_json(root_path / paths.get("data_universe", "")) if paths.get("data_universe") else None
    metadata = _load_json(root_path / paths.get("checkpoint_metadata", "")) if paths.get("checkpoint_metadata") else None
    run_summary = _load_json(root_path / paths.get("run_summary", "")) if paths.get("run_summary") else None
    split_stats = _load_json(root_path / paths.get("split_stats_summary", "")) if paths.get("split_stats_summary") else None

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
        required_source_ids = {str(value) for value in config.get("required_source_ids", [])}
        missing_sources = sorted(required_source_ids - set(universe_counts["source_ids"]))
        if missing_sources:
            findings.append({"severity": "error", "code": "data_universe_missing_required_sources", "missing": missing_sources})
        required_tiers = {str(value) for value in config.get("required_resolution_tiers", [])}
        missing_tiers = sorted(required_tiers - set(universe_counts["resolution_tiers"]))
        if missing_tiers:
            findings.append({"severity": "error", "code": "data_universe_missing_required_resolution_tiers", "missing": missing_tiers})
        expected_by_source = {str(key): value for key, value in dict(config.get("expected_variants_by_source", {})).items()}
        findings.extend(_expected_count_mismatches(universe_counts["source_ids"], expected_by_source, code="data_universe_source_count_mismatch"))
        expected_by_tier = {str(key): value for key, value in dict(config.get("expected_variants_by_resolution_tier", {})).items()}
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

    decode_counts_by_source = _recording_counts(decode, "source_id")
    decode_counts_by_tier = _recording_counts(decode, "resolution_tier")
    if decode is not None:
        selected = int(decode.get("selected_recording_variants", -1))
        if selected != expected_variants:
            findings.append({"severity": "error", "code": "decode_selected_variants_mismatch", "expected": expected_variants, "actual": selected})
        if int(decode.get("num_shards", -1)) != expected_shards:
            findings.append({"severity": "error", "code": "decode_num_shards_mismatch", "expected": expected_shards, "actual": decode.get("num_shards")})
        failures = decode.get("failures") or []
        if failures:
            findings.append({"severity": "error", "code": "decode_failures_present", "count": len(failures)})
        required_source_ids = {str(value) for value in config.get("required_source_ids", [])}
        missing_decoded_sources = sorted(required_source_ids - set(decode_counts_by_source))
        if missing_decoded_sources:
            findings.append({"severity": "error", "code": "decode_missing_required_sources", "missing": missing_decoded_sources})
        required_tiers = {str(value) for value in config.get("required_resolution_tiers", [])}
        missing_decoded_tiers = sorted(required_tiers - set(decode_counts_by_tier))
        if missing_decoded_tiers:
            findings.append({"severity": "error", "code": "decode_missing_required_resolution_tiers", "missing": missing_decoded_tiers})
        findings.extend(
            _expected_count_mismatches(
                decode_counts_by_source,
                {str(key): value for key, value in dict(config.get("expected_variants_by_source", {})).items()},
                code="decode_source_count_mismatch",
            )
        )
        findings.extend(
            _expected_count_mismatches(
                decode_counts_by_tier,
                {str(key): value for key, value in dict(config.get("expected_variants_by_resolution_tier", {})).items()},
                code="decode_resolution_tier_count_mismatch",
            )
        )
        if universe_rows and isinstance(decode.get("recordings"), list):
            universe_ids = {universe_row_id(row) for row in universe_rows}
            decoded_ids = {str(row.get("universe_row_id")) for row in decode.get("recordings", []) if isinstance(row, dict) and row.get("universe_row_id")}
            missing_ids = sorted(universe_ids - decoded_ids)
            extra_ids = sorted(decoded_ids - universe_ids)
            if missing_ids or extra_ids:
                findings.append(
                    {
                        "severity": "error",
                        "code": "decode_universe_row_id_mismatch",
                        "missing_count": len(missing_ids),
                        "extra_count": len(extra_ids),
                        "missing_sample": missing_ids[:10],
                        "extra_sample": extra_ids[:10],
                    }
                )
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
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "expected_recording_variants": expected_variants,
        "expected_shards": expected_shards,
        "artifacts": artifacts,
        "counts": count_report,
        "data_universe_counts": universe_counts,
        "decode_counts_by_source": decode_counts_by_source,
        "decode_counts_by_resolution_tier": decode_counts_by_tier,
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
