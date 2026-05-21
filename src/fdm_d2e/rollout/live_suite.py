from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import write_json

PASS_STATUSES = {"pass", "success", "completed"}
FAIL_STATUSES = {"fail", "failed", "crash", "timeout", "blocked", "error"}
DISALLOWED_CONTROL_BACKENDS = {"dry_run", "deterministic_replay", "replay", "noop", "none", "simulated"}
DEFAULT_ALLOWED_CONTROL_BACKENDS = {
    "evdev_uinput",
    "macos_quartz",
    "native_os_input",
    "os_input",
    "pyautogui_os_input",
    "uinput",
    "win32_sendinput",
    "xdotool",
    "ydotool",
}


class LiveSuiteValidationError(ValueError):
    """Raised when live graphical game-suite evidence is malformed."""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise LiveSuiteValidationError(f"expected list, got {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_status(path_text: str | None, *, root: Path = Path()) -> dict[str, Any]:
    if not path_text:
        return {"path": path_text, "exists": False, "bytes": 0, "sha256": None}
    path = root / path_text
    if not path.exists() or not path.is_file():
        return {"path": path_text, "exists": False, "bytes": 0, "sha256": None}
    return {"path": path_text, "exists": True, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _numeric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def planned_open_source_games(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled open-source graphical/offline games from a suite config."""

    games = []
    for game in _as_list(config.get("games")):
        if not bool(game.get("enabled", True)):
            continue
        if bool(game.get("open_source")) and bool(game.get("graphical")) and bool(game.get("offline_capable", True)):
            games.append(game)
    return games


def suite_protocol_report(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a live-suite plan without requiring evidence files.

    This report is intentionally non-terminal: it can show that a suite is
    correctly specified, but it cannot satisfy G008 until live evidence exists
    and `validate_live_suite_evidence` returns `pass`.
    """

    thresholds = live_suite_thresholds(config)
    games = planned_open_source_games(config)
    task_count = sum(len(_as_list(game.get("tasks"))) for game in games)
    seed_count = sum(len(_as_list(task.get("seeds"))) for game in games for task in _as_list(game.get("tasks")))
    games_ready = [
        {
            "id": game.get("id"),
            "name": game.get("name"),
            "license": game.get("license"),
            "source_url": game.get("source_url"),
            "tasks": [task.get("id") for task in _as_list(game.get("tasks"))],
        }
        for game in games
    ]
    status = "protocol_ready" if len(games) >= thresholds["min_games"] and task_count >= thresholds["min_tasks"] else "incomplete_protocol"
    gates = {
        "status": status,
        "planned_games": len(games),
        "min_games": thresholds["min_games"],
        "planned_tasks": task_count,
        "min_tasks": thresholds["min_tasks"],
        "planned_seeded_episodes": seed_count,
        "min_seeds_per_task": thresholds["min_seeds_per_task"],
        "requires_video": thresholds["require_video"],
        "requires_replay": thresholds["require_replay"],
        "requires_latency_log": thresholds["require_latency_log"],
        "requires_failure_log": thresholds["require_failure_log"],
        "allowed_evidence_modes": sorted(allowed_live_evidence_modes(config)),
    }
    return {
        "schema": "live_game_suite_protocol.v1",
        "suite_id": str(config.get("suite_id", "g008_live_open_game_suite")),
        "status": status,
        "claim_boundary": "Protocol readiness is not live harness success; final G008 requires validated live evidence.",
        "games": games_ready,
        "quality_gate": gates,
    }


def live_suite_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    thresholds = dict(config.get("thresholds", {}))
    return {
        "min_games": int(thresholds.get("min_games", 3)),
        "min_tasks": int(thresholds.get("min_tasks", 3)),
        "min_seeds_per_task": int(thresholds.get("min_seeds_per_task", 5)),
        "min_episode_pass_rate": float(thresholds.get("min_episode_pass_rate", 0.60)),
        "min_task_pass_rate": float(thresholds.get("min_task_pass_rate", 0.67)),
        "max_p95_latency_ms": float(thresholds.get("max_p95_latency_ms", 250.0)),
        "min_baseline_win_rate": float(thresholds.get("min_baseline_win_rate", 0.50)),
        "require_video": bool(thresholds.get("require_video", True)),
        "require_replay": bool(thresholds.get("require_replay", True)),
        "require_latency_log": bool(thresholds.get("require_latency_log", True)),
        "require_failure_log": bool(thresholds.get("require_failure_log", True)),
        "require_statistical_comparison": bool(thresholds.get("require_statistical_comparison", True)),
        "require_runtime_metadata": bool(thresholds.get("require_runtime_metadata", True)),
        "require_window_title_match": bool(thresholds.get("require_window_title_match", True)),
        "min_action_count_per_episode": int(thresholds.get("min_action_count_per_episode", 1)),
    }


def allowed_live_evidence_modes(config: dict[str, Any]) -> set[str]:
    modes = config.get("allowed_evidence_modes") or ["live_desktop_control", "live_graphical_game_control"]
    if not isinstance(modes, list):
        raise LiveSuiteValidationError("allowed_evidence_modes must be a list")
    return {str(mode).lower() for mode in modes}


def _task_lookup(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for game in planned_open_source_games(config):
        game_id = str(game.get("id"))
        for task in _as_list(game.get("tasks")):
            lookup[(game_id, str(task.get("id")))] = {"game": game, "task": task}
    return lookup


def _allowed_control_backends(config: dict[str, Any]) -> set[str]:
    values = config.get("allowed_control_backends") or sorted(DEFAULT_ALLOWED_CONTROL_BACKENDS)
    if not isinstance(values, list):
        raise LiveSuiteValidationError("allowed_control_backends must be a list")
    return {str(value).lower() for value in values}


def _episode_artifact_status(episode: dict[str, Any], *, root: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    video = _file_status(episode.get("video_path"), root=root)
    replay = _file_status(episode.get("replay_path"), root=root)
    latency = _file_status(episode.get("latency_log_path"), root=root)
    failure = _file_status(episode.get("failure_log_path"), root=root)
    missing = []
    if thresholds["require_video"] and not video["exists"]:
        missing.append("video_path")
    if thresholds["require_replay"] and not replay["exists"]:
        missing.append("replay_path")
    if thresholds["require_latency_log"] and not latency["exists"]:
        missing.append("latency_log_path")
    if thresholds["require_failure_log"] and not failure["exists"]:
        missing.append("failure_log_path")
    return {"video": video, "replay": replay, "latency_log": latency, "failure_log": failure, "missing_required_artifacts": missing}


def _window_title_matches(pattern: str | None, title: str) -> bool:
    if not pattern:
        return bool(title.strip())
    try:
        return re.search(pattern, title, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _runtime_metadata_status(
    episode: dict[str, Any],
    *,
    root: Path,
    info: dict[str, Any],
    config: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = episode.get("runtime") if isinstance(episode.get("runtime"), dict) else {}
    findings: list[dict[str, Any]] = []
    allowed_backends = _allowed_control_backends(config)
    control_backend = str(runtime.get("control_backend", "")).lower()
    if thresholds["require_runtime_metadata"] and not runtime:
        findings.append({"severity": "error", "code": "missing_episode_runtime_metadata"})
    if control_backend in DISALLOWED_CONTROL_BACKENDS:
        findings.append({"severity": "error", "code": "non_live_control_backend", "control_backend": runtime.get("control_backend")})
    if control_backend not in allowed_backends:
        findings.append(
            {
                "severity": "error",
                "code": "control_backend_not_allowed",
                "allowed": sorted(allowed_backends),
                "actual": runtime.get("control_backend"),
            }
        )
    agent_mode = str(runtime.get("agent_mode", "")).lower()
    if agent_mode not in {"trained_fdm_policy", "trained_idm_fdm_policy", "fdm_policy"}:
        findings.append({"severity": "error", "code": "agent_mode_not_trained_policy", "actual": runtime.get("agent_mode")})
    checkpoint = _file_status(runtime.get("checkpoint_path"), root=root)
    if not checkpoint["exists"]:
        findings.append({"severity": "error", "code": "missing_runtime_checkpoint_artifact", "path": runtime.get("checkpoint_path")})
    adapter_config = _file_status(runtime.get("adapter_config_path"), root=root)
    if not adapter_config["exists"]:
        findings.append({"severity": "error", "code": "missing_runtime_adapter_config_artifact", "path": runtime.get("adapter_config_path")})
    game = info["game"]
    window_title = str(runtime.get("window_title", ""))
    if thresholds["require_window_title_match"] and not _window_title_matches(game.get("window_title_pattern"), window_title):
        findings.append(
            {
                "severity": "error",
                "code": "window_title_mismatch",
                "expected_pattern": game.get("window_title_pattern"),
                "actual": runtime.get("window_title"),
            }
        )
    process_name = str(runtime.get("process_name", "")).strip()
    if thresholds["require_runtime_metadata"] and not process_name:
        findings.append({"severity": "error", "code": "missing_game_process_name"})
    action_count = int(_numeric(runtime.get("action_count"), default=0.0))
    if action_count < thresholds["min_action_count_per_episode"]:
        findings.append(
            {
                "severity": "error",
                "code": "too_few_live_actions",
                "expected_min": thresholds["min_action_count_per_episode"],
                "actual": action_count,
            }
        )
    start = _numeric(runtime.get("started_at_unix"), default=0.0)
    end = _numeric(runtime.get("ended_at_unix"), default=0.0)
    if thresholds["require_runtime_metadata"] and (start <= 0.0 or end <= start):
        findings.append({"severity": "error", "code": "invalid_episode_runtime_window", "started_at_unix": start, "ended_at_unix": end})
    return (
        {
            "control_backend": runtime.get("control_backend"),
            "agent_mode": runtime.get("agent_mode"),
            "process_name": runtime.get("process_name"),
            "window_title": runtime.get("window_title"),
            "action_count": action_count,
            "started_at_unix": runtime.get("started_at_unix"),
            "ended_at_unix": runtime.get("ended_at_unix"),
            "checkpoint": checkpoint,
            "adapter_config": adapter_config,
        },
        findings,
    )


def validate_live_suite_evidence(config: dict[str, Any], evidence: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    """Validate live open-source graphical game evidence for the G008 gate.

    The validator is deliberately strict: repo-local deterministic harnesses and
    dry-run adapter fixtures cannot pass this gate. Evidence must cover at least
    the configured games/tasks/seeds and include video/replay/latency/failure
    logs plus baseline/statistical comparison when required.
    """

    root_path = Path(root)
    thresholds = live_suite_thresholds(config)
    allowed_modes = allowed_live_evidence_modes(config)
    task_defs = _task_lookup(config)
    episodes = _as_list(evidence.get("episodes"))
    findings: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_game: dict[str, list[dict[str, Any]]] = {}

    evidence_mode = str(evidence.get("evidence_mode", "")).lower()
    if evidence_mode in {"dry_run", "deterministic_replay", "game_adjacent"}:
        findings.append({"severity": "error", "code": "non_live_evidence_mode", "detail": evidence.get("evidence_mode")})
    if evidence_mode not in allowed_modes:
        findings.append(
            {
                "severity": "error",
                "code": "evidence_mode_not_allowed",
                "allowed": sorted(allowed_modes),
                "actual": evidence.get("evidence_mode"),
            }
        )

    for key, info in task_defs.items():
        game_id, task_id = key
        seeds = [int(seed) for seed in _as_list(info["task"].get("seeds"))]
        if len(seeds) < thresholds["min_seeds_per_task"]:
            findings.append({"severity": "error", "code": "task_has_too_few_planned_seeds", "game_id": game_id, "task_id": task_id})

    for idx, episode in enumerate(episodes):
        game_id = str(episode.get("game_id"))
        task_id = str(episode.get("task_id"))
        seed = episode.get("seed")
        key = (game_id, task_id)
        if key not in task_defs:
            findings.append({"severity": "error", "code": "unknown_episode_task", "episode_index": idx, "game_id": game_id, "task_id": task_id})
            continue
        info = task_defs[key]
        planned_seeds = {int(seed_value) for seed_value in _as_list(info["task"].get("seeds"))}
        if seed is None or int(seed) not in planned_seeds:
            findings.append({"severity": "error", "code": "episode_seed_not_planned", "episode_index": idx, "game_id": game_id, "task_id": task_id, "seed": seed})
        artifacts = _episode_artifact_status(episode, root=root_path, thresholds=thresholds)
        for missing in artifacts["missing_required_artifacts"]:
            findings.append({"severity": "error", "code": "missing_episode_artifact", "episode_index": idx, "artifact": missing})
        runtime, runtime_findings = _runtime_metadata_status(episode, root=root_path, info=info, config=config, thresholds=thresholds)
        for finding in runtime_findings:
            findings.append({"episode_index": idx, "game_id": game_id, "task_id": task_id, **finding})
        p95_latency_ms = _numeric(episode.get("latency", {}).get("p95_ms"), default=float("inf"))
        if p95_latency_ms > thresholds["max_p95_latency_ms"]:
            findings.append(
                {
                    "severity": "error",
                    "code": "episode_latency_exceeds_threshold",
                    "episode_index": idx,
                    "p95_ms": p95_latency_ms,
                    "max_p95_ms": thresholds["max_p95_latency_ms"],
                }
            )
        status = str(episode.get("status", "")).lower()
        runtime_passed = not [item for item in runtime_findings if item.get("severity") == "error"]
        passed = status in PASS_STATUSES and not artifacts["missing_required_artifacts"] and runtime_passed and p95_latency_ms <= thresholds["max_p95_latency_ms"]
        result = {
            "episode_index": idx,
            "game_id": game_id,
            "task_id": task_id,
            "seed": seed,
            "status": status,
            "passed": passed,
            "score": _numeric(episode.get("score"), default=0.0),
            "baseline_score": _numeric(episode.get("baseline_score"), default=0.0),
            "p95_latency_ms": p95_latency_ms,
            "artifacts": artifacts,
            "runtime": runtime,
        }
        episode_results.append(result)
        by_task.setdefault(key, []).append(result)
        by_game.setdefault(game_id, []).append(result)

    task_summaries = []
    passed_tasks = []
    for key, info in sorted(task_defs.items()):
        game_id, task_id = key
        task_rows = by_task.get(key, [])
        planned_seeds = {int(seed_value) for seed_value in _as_list(info["task"].get("seeds"))}
        observed_seeds = {int(row["seed"]) for row in task_rows if row.get("seed") is not None}
        missing_seeds = sorted(planned_seeds - observed_seeds)
        if missing_seeds:
            findings.append({"severity": "error", "code": "missing_seed_episodes", "game_id": game_id, "task_id": task_id, "missing_seeds": missing_seeds})
        pass_count = sum(1 for row in task_rows if row["passed"])
        pass_rate = pass_count / max(1, len(planned_seeds))
        baseline_wins = sum(1 for row in task_rows if row["score"] > row["baseline_score"])
        baseline_win_rate = baseline_wins / max(1, len(task_rows))
        task_passed = (
            len(task_rows) >= len(planned_seeds)
            and not missing_seeds
            and pass_rate >= thresholds["min_episode_pass_rate"]
            and baseline_win_rate >= thresholds["min_baseline_win_rate"]
        )
        if task_passed:
            passed_tasks.append(key)
        task_summaries.append(
            {
                "game_id": game_id,
                "task_id": task_id,
                "planned_seeds": sorted(planned_seeds),
                "observed_episodes": len(task_rows),
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "baseline_win_rate": baseline_win_rate,
                "passed": task_passed,
            }
        )

    statistical_comparison = evidence.get("statistical_comparison") or {}
    stats_file = _file_status(statistical_comparison.get("path"), root=root_path)
    if thresholds["require_statistical_comparison"]:
        if not stats_file["exists"]:
            findings.append({"severity": "error", "code": "missing_statistical_comparison_artifact"})
        if statistical_comparison.get("holm_adjusted_p_lt_0_05") is not True:
            findings.append({"severity": "error", "code": "missing_strong_statistical_bar"})

    passed_game_ids = {game_id for game_id, rows in by_game.items() if any(row["passed"] for row in rows)}
    task_pass_rate = len(passed_tasks) / max(1, len(task_defs))
    quality_gate = {
        "status": "pass",
        "planned_games": len(planned_open_source_games(config)),
        "games_with_passed_episode": len(passed_game_ids),
        "min_games": thresholds["min_games"],
        "planned_tasks": len(task_defs),
        "passed_tasks": len(passed_tasks),
        "min_tasks": thresholds["min_tasks"],
        "task_pass_rate": task_pass_rate,
        "min_task_pass_rate": thresholds["min_task_pass_rate"],
        "episodes_observed": len(episode_results),
        "findings_count": len([item for item in findings if item["severity"] == "error"]),
    }
    if quality_gate["planned_games"] < thresholds["min_games"]:
        findings.append({"severity": "error", "code": "too_few_planned_games"})
    if quality_gate["games_with_passed_episode"] < thresholds["min_games"]:
        findings.append({"severity": "error", "code": "too_few_games_with_passed_episode"})
    if quality_gate["planned_tasks"] < thresholds["min_tasks"]:
        findings.append({"severity": "error", "code": "too_few_planned_tasks"})
    if quality_gate["passed_tasks"] < thresholds["min_tasks"] or task_pass_rate < thresholds["min_task_pass_rate"]:
        findings.append({"severity": "error", "code": "too_few_passed_tasks"})

    error_count = len([item for item in findings if item["severity"] == "error"])
    quality_gate["findings_count"] = error_count
    quality_gate["status"] = "pass" if error_count == 0 else "fail"
    return {
        "schema": "live_game_suite_evidence_validation.v1",
        "suite_id": str(config.get("suite_id", "g008_live_open_game_suite")),
        "evidence_mode": evidence.get("evidence_mode"),
        "claim_boundary": "Pass here is required before any G008 live open-source graphical-game claim; deterministic game-adjacent harnesses cannot satisfy this gate.",
        "quality_gate": quality_gate,
        "task_summaries": task_summaries,
        "episode_results": episode_results,
        "statistical_comparison_artifact": stats_file,
        "findings": findings,
    }


def run_live_suite_validation(config: dict[str, Any], evidence: dict[str, Any] | None = None, *, root: str | Path = ".") -> dict[str, Any]:
    if evidence is None:
        return suite_protocol_report(config)
    return validate_live_suite_evidence(config, evidence, root=root)


def write_live_suite_report(config: dict[str, Any], evidence: dict[str, Any] | None = None, *, root: str | Path = ".") -> dict[str, Any]:
    report = run_live_suite_validation(config, evidence, root=root)
    output_path = config.get("output_path")
    if output_path:
        write_json(output_path, report)
    return report


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
