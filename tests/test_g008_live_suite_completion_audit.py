from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g008_completion import validate_g008_live_suite_completion


GAMES = ["supertuxkart", "luanti_minetest", "xonotic"]


def _config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G008",
        "prerequisite_goals": ["G003", "G004", "G007"],
        "thresholds": {"min_games": 3, "min_tasks": 3, "min_episodes": 15},
        "allowed_checkpoint_namespaces": ["d2e_full_corpus", "d2e_aux"],
        "allowed_evidence_modes": ["live_desktop_control", "live_graphical_game_control"],
        "paths": {
            "suite_config": "configs/harness/suite.yaml",
            "evidence_validation": "artifacts/harness/validation.json",
            "trained_checkpoint_metadata": "outputs/fdm/checkpoint_metadata.json",
            "runtime_adapter_contract": "artifacts/runtime/contract.json",
            "live_suite_doc": "docs/live.md",
        },
        "validation_expectations": {"schema": "live_game_suite_evidence_validation.v1", "quality_gate.status": "pass"},
        "checkpoint_expectations": {"oracle_ground_truth_control": False, "data_universe.exists": True, "split_contract.exists": True},
    }


def _write_file(path: Path, text: str = "evidence") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(
        root / cfg["goals_path"],
        {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}, {"id": "G007", "status": "complete"}, {"id": "G008", "status": "complete"}]},
    )
    write_json(root / cfg["paths"]["suite_config"], {"schema": "live_game_suite_config.v1"})
    _write_file(root / cfg["paths"]["live_suite_doc"], "live suite doc")
    write_json(root / cfg["paths"]["runtime_adapter_contract"], {"status": "pass"})
    write_json(
        root / cfg["paths"]["trained_checkpoint_metadata"],
        {
            "source_namespace": "d2e_full_corpus",
            "oracle_ground_truth_control": False,
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
        },
    )
    episode_results = []
    for game in GAMES:
        task_id = f"{game}_task"
        for seed in range(5):
            prefix = root / "artifacts/harness/live" / game / str(seed)
            episode_results.append(
                {
                    "game_id": game,
                    "task_id": task_id,
                    "seed": seed,
                    "passed": True,
                    "artifacts": {
                        "video": {"path": _write_file(prefix / "episode.mp4")},
                        "replay": {"path": _write_file(prefix / "replay.jsonl")},
                        "latency_log": {"path": _write_file(prefix / "latency.jsonl")},
                        "failure_log": {"path": _write_file(prefix / "failures.jsonl", "[]")},
                    },
                }
            )
    stats_path = _write_file(root / "artifacts/harness/live/statistical_comparison.json", "{}")
    write_json(
        root / cfg["paths"]["evidence_validation"],
        {
            "schema": "live_game_suite_evidence_validation.v1",
            "evidence_mode": "live_desktop_control",
            "quality_gate": {
                "status": "pass",
                "games_with_passed_episode": 3,
                "passed_tasks": 3,
                "episodes_observed": 15,
                "findings_count": 0,
            },
            "episode_results": episode_results,
            "statistical_comparison_artifact": {"path": stats_path, "exists": True},
            "findings": [],
        },
    )


def test_g008_completion_audit_passes_on_live_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g008_live_suite_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert len(payload["episode_artifact_paths"]) == 61  # 15 * 4 episode artifacts + statistical comparison


def test_g008_completion_audit_rejects_protocol_only_and_missing_prereq(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    goals_path = tmp_path / cfg["goals_path"]
    write_json(goals_path, {"goals": [{"id": "G003", "status": "in_progress"}, {"id": "G008", "status": "pending"}]})
    write_json(tmp_path / cfg["paths"]["evidence_validation"], {"schema": "live_game_suite_protocol.v1", "quality_gate": {"status": "protocol_ready"}})
    payload = validate_g008_live_suite_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "validation_schema_not_live_evidence" in codes
    assert "live_suite_quality_gate_not_pass" in codes
    assert "live_suite_evidence_mode_not_allowed" in codes
