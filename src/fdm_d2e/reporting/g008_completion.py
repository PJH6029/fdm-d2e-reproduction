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


def _get(data: dict[str, Any] | None, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _expected_count_mismatches(actual: dict[str, Any], expected: dict[str, Any], *, code: str, audit_key: str) -> list[dict[str, Any]]:
    findings = []
    for key, raw_expected in sorted(expected.items()):
        try:
            expected_count = int(raw_expected)
        except (TypeError, ValueError):
            findings.append({"severity": "error", "code": f"{code}_invalid_expected", "audit_key": audit_key, "key": key, "expected": raw_expected})
            continue
        actual_value = actual.get(str(key))
        try:
            actual_count = int(actual_value) if actual_value is not None else None
        except (TypeError, ValueError):
            actual_count = None
        if actual_count != expected_count:
            findings.append(
                {
                    "severity": "error",
                    "code": code,
                    "audit_key": audit_key,
                    "key": str(key),
                    "expected": expected_count,
                    "actual": actual_value,
                }
            )
    return findings


def _validate_d2e_only_audit(
    audit: dict[str, Any] | None,
    *,
    audit_key: str,
    expected_variants: int,
    expected_by_source: dict[str, Any],
    expected_by_tier: dict[str, Any],
    require_pass: bool,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "audit_key": audit_key,
        "status": None if audit is None else audit.get("status"),
        "error_count": None if audit is None else audit.get("error_count"),
        "included_recording_variants": None,
        "source_ids": {},
        "resolution_tiers": {},
        "decode_source_ids": {},
        "decode_resolution_tiers": {},
    }
    if audit is None:
        findings.append({"severity": "error", "code": "missing_d2e_only_completion_audit", "audit_key": audit_key})
        return report
    if require_pass and audit.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "d2e_only_completion_audit_not_pass",
                "audit_key": audit_key,
                "status": audit.get("status"),
                "error_count": audit.get("error_count"),
            }
        )
    universe_counts = audit.get("data_universe_counts") if isinstance(audit.get("data_universe_counts"), dict) else {}
    included = universe_counts.get("included_recording_variants")
    report["included_recording_variants"] = included
    try:
        included_count = int(included)
    except (TypeError, ValueError):
        included_count = None
    if included_count != expected_variants:
        findings.append(
            {
                "severity": "error",
                "code": "d2e_only_audit_included_variants_mismatch",
                "audit_key": audit_key,
                "expected": expected_variants,
                "actual": included,
            }
        )
    source_ids = universe_counts.get("source_ids") if isinstance(universe_counts.get("source_ids"), dict) else {}
    tiers = universe_counts.get("resolution_tiers") if isinstance(universe_counts.get("resolution_tiers"), dict) else {}
    report["source_ids"] = dict(source_ids)
    report["resolution_tiers"] = dict(tiers)
    findings.extend(_expected_count_mismatches(source_ids, expected_by_source, code="d2e_only_audit_source_count_mismatch", audit_key=audit_key))
    findings.extend(_expected_count_mismatches(tiers, expected_by_tier, code="d2e_only_audit_resolution_tier_count_mismatch", audit_key=audit_key))
    decode_sources = audit.get("decode_counts_by_source") if isinstance(audit.get("decode_counts_by_source"), dict) else {}
    decode_tiers = audit.get("decode_counts_by_resolution_tier") if isinstance(audit.get("decode_counts_by_resolution_tier"), dict) else {}
    report["decode_source_ids"] = dict(decode_sources)
    report["decode_resolution_tiers"] = dict(decode_tiers)
    if decode_sources:
        findings.extend(_expected_count_mismatches(decode_sources, expected_by_source, code="d2e_only_audit_decode_source_count_mismatch", audit_key=audit_key))
    if decode_tiers:
        findings.extend(_expected_count_mismatches(decode_tiers, expected_by_tier, code="d2e_only_audit_decode_resolution_tier_count_mismatch", audit_key=audit_key))
    return report


def _episode_artifact_paths(validation: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    if not validation:
        return paths
    for episode in validation.get("episode_results", []) or []:
        artifacts = episode.get("artifacts", {}) if isinstance(episode, dict) else {}
        for evidence in artifacts.values():
            if isinstance(evidence, dict) and evidence.get("path"):
                paths.append(str(evidence["path"]))
        runtime = episode.get("runtime", {}) if isinstance(episode, dict) else {}
        if isinstance(runtime, dict):
            for key in ["checkpoint", "adapter_config"]:
                evidence = runtime.get(key)
                if isinstance(evidence, dict) and evidence.get("path"):
                    paths.append(str(evidence["path"]))
    stats_path = _get(validation, "statistical_comparison_artifact.path")
    if stats_path:
        paths.append(str(stats_path))
    return list(dict.fromkeys(paths))


def validate_g008_live_suite_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G008-live-game-suite"))
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

    validation = _load_json(root_path / paths.get("evidence_validation", "")) if paths.get("evidence_validation") else None
    checkpoint_metadata = _load_json(root_path / paths.get("trained_checkpoint_metadata", "")) if paths.get("trained_checkpoint_metadata") else None
    g003_audit = _load_json(root_path / paths.get("g003_completion_audit", "")) if paths.get("g003_completion_audit") else None
    g004_audit = _load_json(root_path / paths.get("g004_completion_audit", "")) if paths.get("g004_completion_audit") else None
    g005_audit = _load_json(root_path / paths.get("g005_completion_audit", "")) if paths.get("g005_completion_audit") else None

    expected_variants = int(config.get("expected_recording_variants", 918))
    expected_by_source = {str(key): value for key, value in dict(config.get("expected_variants_by_source", {})).items()}
    expected_by_tier = {str(key): value for key, value in dict(config.get("expected_variants_by_resolution_tier", {})).items()}
    require_d2e_only_audits_pass = bool(config.get("require_d2e_only_completion_audits_pass", True))
    d2e_only_audit_report = {
        "g003": _validate_d2e_only_audit(
            g003_audit,
            audit_key="g003",
            expected_variants=expected_variants,
            expected_by_source=expected_by_source,
            expected_by_tier=expected_by_tier,
            require_pass=require_d2e_only_audits_pass,
            findings=findings,
        ),
        "g004": _validate_d2e_only_audit(
            g004_audit,
            audit_key="g004",
            expected_variants=expected_variants,
            expected_by_source=expected_by_source,
            expected_by_tier=expected_by_tier,
            require_pass=require_d2e_only_audits_pass,
            findings=findings,
        ),
    }

    for dotted, expected in dict(config.get("validation_expectations", {})).items():
        actual = _get(validation, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "validation_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})
    for dotted, expected in dict(config.get("checkpoint_expectations", {})).items():
        actual = _get(checkpoint_metadata, dotted)
        if actual != expected:
            findings.append({"severity": "error", "code": "checkpoint_expectation_mismatch", "json_path": dotted, "expected": expected, "actual": actual})

    thresholds = dict(config.get("thresholds", {}))
    if validation is not None:
        if validation.get("schema") != "live_game_suite_evidence_validation.v1":
            findings.append({"severity": "error", "code": "validation_schema_not_live_evidence", "actual": validation.get("schema")})
        gate = validation.get("quality_gate", {})
        if gate.get("status") != "pass":
            findings.append({"severity": "error", "code": "live_suite_quality_gate_not_pass", "actual": gate.get("status")})
        min_games = int(thresholds.get("min_games", config.get("min_games", 3)))
        min_tasks = int(thresholds.get("min_tasks", config.get("min_tasks", 3)))
        min_episodes = int(thresholds.get("min_episodes", config.get("min_episodes", 15)))
        if int(gate.get("games_with_passed_episode", 0)) < min_games:
            findings.append({"severity": "error", "code": "too_few_passed_games", "expected_min": min_games, "actual": gate.get("games_with_passed_episode")})
        if int(gate.get("passed_tasks", 0)) < min_tasks:
            findings.append({"severity": "error", "code": "too_few_passed_tasks", "expected_min": min_tasks, "actual": gate.get("passed_tasks")})
        if int(gate.get("episodes_observed", 0)) < min_episodes:
            findings.append({"severity": "error", "code": "too_few_observed_episodes", "expected_min": min_episodes, "actual": gate.get("episodes_observed")})
        if int(gate.get("findings_count", 0)) != 0:
            findings.append({"severity": "error", "code": "live_suite_validation_findings_present", "actual": gate.get("findings_count")})
        if _get(validation, "statistical_comparison_artifact.exists") is not True:
            findings.append({"severity": "error", "code": "missing_live_suite_statistical_comparison_artifact"})
        stats_summary = validation.get("statistical_comparison_summary") if isinstance(validation.get("statistical_comparison_summary"), dict) else {}
        max_adjusted_p = float(thresholds.get("max_adjusted_p_value", 0.05))
        min_effect_size = float(thresholds.get("min_effect_size", 0.0))
        adjusted_p = _numeric(stats_summary.get("adjusted_p_value"))
        effect_size = _numeric(stats_summary.get("effect_size"))
        episode_count = _numeric(stats_summary.get("episode_count"))
        mean_delta = _numeric(stats_summary.get("mean_score_delta"))
        if stats_summary.get("holm_adjusted_p_lt_0_05") is not True:
            findings.append({"severity": "error", "code": "missing_live_suite_strong_statistical_bar"})
        if adjusted_p is None or adjusted_p > max_adjusted_p:
            findings.append(
                {
                    "severity": "error",
                    "code": "live_suite_adjusted_p_value_not_significant",
                    "max_adjusted_p_value": max_adjusted_p,
                    "actual": adjusted_p,
                }
            )
        if effect_size is None or effect_size <= min_effect_size:
            findings.append(
                {
                    "severity": "error",
                    "code": "live_suite_effect_size_not_positive",
                    "min_effect_size": min_effect_size,
                    "actual": effect_size,
                }
            )
        if mean_delta is None or mean_delta <= 0.0:
            findings.append({"severity": "error", "code": "live_suite_agent_not_above_baseline", "mean_score_delta": mean_delta})
        if episode_count is None or episode_count < min_episodes:
            findings.append(
                {
                    "severity": "error",
                    "code": "live_suite_statistical_episode_count_too_low",
                    "expected_min": min_episodes,
                    "actual": episode_count,
                }
            )
        allowed_modes = {str(item).lower() for item in config.get("allowed_evidence_modes", ["live_desktop_control", "live_graphical_game_control"])}
        evidence_mode = validation.get("evidence_mode")
        if str(evidence_mode).lower() not in allowed_modes:
            findings.append(
                {
                    "severity": "error",
                    "code": "live_suite_evidence_mode_not_allowed",
                    "allowed": sorted(allowed_modes),
                    "actual": evidence_mode,
                }
            )
        if config.get("require_episode_artifact_hashes", True):
            missing_hashes = [path for path in _episode_artifact_paths(validation) if not _file_status(root_path / path, path)["sha256"]]
            if missing_hashes:
                findings.append({"severity": "error", "code": "missing_episode_artifact_hashes", "paths": missing_hashes})
        if config.get("require_runtime_artifact_hashes", True):
            runtime_missing: list[dict[str, Any]] = []
            for idx, episode in enumerate(validation.get("episode_results", []) or []):
                runtime = episode.get("runtime", {}) if isinstance(episode, dict) else {}
                for key in ["checkpoint", "adapter_config"]:
                    evidence = runtime.get(key) if isinstance(runtime, dict) else None
                    path = evidence.get("path") if isinstance(evidence, dict) else None
                    status = _file_status(root_path / str(path), str(path)) if path else {"exists": False, "sha256": None}
                    if not status["exists"] or not status["sha256"]:
                        runtime_missing.append({"episode_index": idx, "artifact": key, "path": path})
            if runtime_missing:
                findings.append({"severity": "error", "code": "missing_runtime_artifact_hashes", "artifacts": runtime_missing})

    if checkpoint_metadata is not None:
        allowed_namespaces = set(str(item) for item in config.get("allowed_checkpoint_namespaces", ["d2e_full_corpus", "d2e_aux"]))
        namespace = str(checkpoint_metadata.get("source_namespace", ""))
        if namespace not in allowed_namespaces:
            findings.append({"severity": "error", "code": "trained_checkpoint_namespace_not_allowed", "allowed": sorted(allowed_namespaces), "actual": namespace})
        if checkpoint_metadata.get("oracle_ground_truth_control") is True:
            findings.append({"severity": "error", "code": "trained_checkpoint_uses_oracle_ground_truth_control"})
        if namespace == "d2e_aux" and bool(config.get("require_g005_for_aux_checkpoint", True)):
            g005_goal_id = str(config.get("g005_goal_id", "G005-aux-data-best-model"))
            g005_status = statuses.get(g005_goal_id, "missing")
            if g005_status != "complete":
                findings.append({"severity": "error", "code": "aux_checkpoint_requires_g005_complete", "goal_id": g005_goal_id, "actual": g005_status})
            if g005_audit is None:
                findings.append({"severity": "error", "code": "missing_g005_completion_audit_for_aux_checkpoint", "path": paths.get("g005_completion_audit")})
            elif g005_audit.get("status") != "pass":
                findings.append(
                    {
                        "severity": "error",
                        "code": "g005_completion_audit_not_pass_for_aux_checkpoint",
                        "status": g005_audit.get("status"),
                        "error_count": g005_audit.get("error_count"),
                    }
                )

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g008_live_suite_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "prerequisite_goal_statuses": prereq_report,
        "d2e_only_audit_report": d2e_only_audit_report,
        "g005_audit_status": None if g005_audit is None else g005_audit.get("status"),
        "artifacts": artifacts,
        "episode_artifact_paths": _episode_artifact_paths(validation),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G008 complete; protocol-only or dry-run evidence cannot pass without passing full-corpus D2E-only G003/G004 audits, trained checkpoint metadata, and live open-source graphical-game validation evidence. D2E+aux checkpoints additionally require passing G005 evidence.",
    }


def write_g008_live_suite_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g008_live_suite_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
