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


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _episode_artifact_paths(validation: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    if not validation:
        return paths
    for episode in validation.get("episode_results", []) or []:
        artifacts = episode.get("artifacts", {}) if isinstance(episode, dict) else {}
        for evidence in artifacts.values():
            if isinstance(evidence, dict) and evidence.get("path"):
                paths.append(str(evidence["path"]))
    stats_path = _get(validation, "statistical_comparison_artifact.path")
    if stats_path:
        paths.append(str(stats_path))
    return paths


def validate_g008_live_suite_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G008-live-game-suite"))
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

    validation = _load_json(root_path / paths.get("evidence_validation", "")) if paths.get("evidence_validation") else None
    checkpoint_metadata = _load_json(root_path / paths.get("trained_checkpoint_metadata", "")) if paths.get("trained_checkpoint_metadata") else None

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
        if config.get("require_episode_artifact_hashes", True):
            missing_hashes = [path for path in _episode_artifact_paths(validation) if not _file_status(root_path / path, path)["sha256"]]
            if missing_hashes:
                findings.append({"severity": "error", "code": "missing_episode_artifact_hashes", "paths": missing_hashes})

    if checkpoint_metadata is not None:
        allowed_namespaces = set(str(item) for item in config.get("allowed_checkpoint_namespaces", ["d2e_full_corpus", "d2e_aux"]))
        namespace = str(checkpoint_metadata.get("source_namespace", ""))
        if namespace not in allowed_namespaces:
            findings.append({"severity": "error", "code": "trained_checkpoint_namespace_not_allowed", "allowed": sorted(allowed_namespaces), "actual": namespace})
        if checkpoint_metadata.get("oracle_ground_truth_control") is True:
            findings.append({"severity": "error", "code": "trained_checkpoint_uses_oracle_ground_truth_control"})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g008_live_suite_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "prerequisite_goal_statuses": prereq_report,
        "artifacts": artifacts,
        "episode_artifact_paths": _episode_artifact_paths(validation),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G008 complete; protocol-only or dry-run evidence cannot pass without trained checkpoint metadata and live open-source graphical-game validation evidence.",
    }


def write_g008_live_suite_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g008_live_suite_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
