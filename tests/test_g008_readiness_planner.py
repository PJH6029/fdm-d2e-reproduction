from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from plan_g008_readiness import build_readiness_plan


GAMES = ["supertuxkart", "luanti_minetest", "xonotic"]


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "suite_config": "configs/harness/g008_suite.json",
        "g008_completion_config": "configs/eval/g008_completion.json",
        "allow_precheckpoint": False,
        "skip_system_checks": False,
        "allow_overwrite_evidence": False,
        "output": "artifacts/harness/g008_readiness_plan.json",
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_configs(root: Path, *, launch_command: str = "echo launch-game") -> None:
    games = []
    for idx, game_id in enumerate(GAMES):
        games.append(
            {
                "id": game_id,
                "name": game_id,
                "open_source": True,
                "graphical": True,
                "offline_capable": True,
                "license": "open-source-test-fixture",
                "source_url": f"https://example.test/{game_id}",
                "launch_command_template": launch_command,
                "window_title_pattern": game_id,
                "tasks": [{"id": f"task_{idx}", "deed": "fixture", "success_metric": "fixture", "seeds": [0, 1, 2, 3, 4]}],
            }
        )
    write_json(
        root / "configs/harness/g008_suite.json",
        {
            "schema": "live_game_suite_config.v1",
            "suite_id": "g008_fixture_suite",
            "allowed_evidence_modes": ["live_desktop_control", "live_graphical_game_control"],
            "allowed_control_backends": ["native_os_input", "xdotool"],
            "thresholds": {"min_games": 3, "min_tasks": 3, "min_seeds_per_task": 5},
            "games": games,
        },
    )
    write_json(
        root / "configs/eval/g008_completion.json",
        {
            "schema": "g008_live_suite_completion_config.v1",
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G008",
            "prerequisite_goals": ["G003", "G004", "G007"],
            "paths": {
                "suite_config": "configs/harness/g008_suite.json",
                "evidence_validation": "artifacts/harness/g008_evidence_validation.json",
                "trained_checkpoint_metadata": "outputs/fdm/checkpoint_metadata.json",
                "runtime_adapter_contract": "artifacts/runtime/contract.json",
                "live_suite_doc": "docs/live.md",
            },
            "checkpoint_expectations": {
                "oracle_ground_truth_control": False,
                "data_universe.exists": True,
                "split_contract.exists": True,
            },
        },
    )


def _write_ready_fixture(root: Path, *, launch_command: str = "echo launch-game") -> None:
    _write_configs(root, launch_command=launch_command)
    write_json(
        root / ".omx/ultragoal/goals.json",
        {
            "goals": [
                {"id": "G003", "status": "complete"},
                {"id": "G004", "status": "complete"},
                {"id": "G007", "status": "complete"},
                {"id": "G008", "status": "pending"},
            ]
        },
    )
    write_json(
        root / "outputs/fdm/checkpoint_metadata.json",
        {
            "source_namespace": "d2e_full_corpus",
            "oracle_ground_truth_control": False,
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
        },
    )
    write_json(root / "artifacts/runtime/contract.json", {"status": "pass"})
    doc = root / "docs/live.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("live suite contract", encoding="utf-8")


def test_g008_readiness_plan_ready_with_protocol_prereqs_and_pre_run_artifacts(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = build_readiness_plan(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["protocol_status"] == "protocol_ready"
    assert payload["planned_games"] == 3
    assert payload["planned_seeded_episodes"] == 15
    assert payload["findings"] == []
    assert any(item["code"] == "evidence_validation_not_collected_yet" for item in payload["warnings"])
    assert payload["commands"]["finalize_with_evidence"][:3] == ["uv", "run", "python"]
    assert "does not launch games" in payload["claim_boundary"]


def test_g008_readiness_plan_blocks_missing_prereqs_checkpoint_and_game_binary(tmp_path: Path):
    _write_ready_fixture(tmp_path, launch_command="definitely_missing_g008_game_binary --fixture")
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    (tmp_path / "outputs/fdm/checkpoint_metadata.json").unlink()
    payload = build_readiness_plan(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "prerequisite_goal_not_complete" in codes
    assert "missing_required_pre_run_artifact" in codes
    assert "game_launch_executable_missing" in codes


def test_g008_readiness_plan_allows_precheckpoint_diagnostics_and_skips_system_checks(tmp_path: Path):
    _write_ready_fixture(tmp_path, launch_command="definitely_missing_g008_game_binary --fixture")
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    payload = build_readiness_plan(_args(tmp_path, allow_precheckpoint=True, skip_system_checks=True))
    assert payload["status"] == "ready"
    codes = {item["code"] for item in payload["warnings"]}
    assert "prerequisite_goal_not_complete" in codes
    assert "system_checks_skipped" in codes


def test_g008_readiness_plan_checks_checkpoint_metadata_expectations(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(
        tmp_path / "outputs/fdm/checkpoint_metadata.json",
        {
            "source_namespace": "d2e_full_corpus",
            "oracle_ground_truth_control": True,
            "data_universe": {"exists": True},
            "split_contract": {"exists": False},
        },
    )
    payload = build_readiness_plan(_args(tmp_path))
    mismatches = [item for item in payload["findings"] if item["code"] == "checkpoint_expectation_mismatch"]
    assert payload["status"] == "blocked"
    assert {item["json_path"] for item in mismatches} == {"oracle_ground_truth_control", "split_contract.exists"}


def test_g008_readiness_plan_blocks_existing_evidence_validation_without_overwrite_flag(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    write_json(tmp_path / "artifacts/harness/g008_evidence_validation.json", {"status": "old"})
    payload = build_readiness_plan(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "evidence_validation_already_exists" for item in payload["findings"])
    overwrite_payload = build_readiness_plan(_args(tmp_path, allow_overwrite_evidence=True))
    assert overwrite_payload["status"] == "ready"
