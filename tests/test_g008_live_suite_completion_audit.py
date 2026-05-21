from __future__ import annotations

import json
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
        "expected_recording_variants": 3,
        "require_d2e_only_completion_audits_pass": True,
        "require_g005_for_aux_checkpoint": True,
        "g005_goal_id": "G005",
        "expected_variants_by_source": {"d2e_480p": 2, "d2e_original": 1},
        "expected_variants_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
        "allowed_checkpoint_namespaces": ["d2e_full_corpus", "d2e_aux"],
        "allowed_evidence_modes": ["live_desktop_control", "live_graphical_game_control"],
        "paths": {
            "suite_config": "configs/harness/suite.yaml",
            "evidence_validation": "artifacts/harness/validation.json",
            "trained_checkpoint_metadata": "outputs/fdm/checkpoint_metadata.json",
            "g003_completion_audit": "artifacts/idm/g003_audit.json",
            "g004_completion_audit": "artifacts/fdm/g004_audit.json",
            "g005_completion_audit": "artifacts/aux/g005_audit.json",
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
    d2e_audit_counts = {
        "included_recording_variants": 3,
        "source_ids": {"d2e_480p": 2, "d2e_original": 1},
        "resolution_tiers": {"480p": 2, "original_fhd_qhd": 1},
    }
    write_json(
        root / cfg["paths"]["g003_completion_audit"],
        {
            "schema": "g003_full_idm_completion_audit.v1",
            "status": "pass",
            "error_count": 0,
            "data_universe_counts": d2e_audit_counts,
            "decode_counts_by_source": {"d2e_480p": 2, "d2e_original": 1},
            "decode_counts_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
        },
    )
    write_json(
        root / cfg["paths"]["g004_completion_audit"],
        {
            "schema": "g004_full_fdm_completion_audit.v1",
            "status": "pass",
            "error_count": 0,
            "data_universe_counts": d2e_audit_counts,
        },
    )
    write_json(root / cfg["paths"]["g005_completion_audit"], {"schema": "g005_aux_completion_audit.v1", "status": "pass", "error_count": 0})
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
    checkpoint_path = _write_file(root / "artifacts/harness/live/trained_fdm_checkpoint.pt", "checkpoint")
    adapter_config_path = _write_file(root / "artifacts/harness/live/runtime_adapter.yaml", "adapter")
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
                    "runtime": {
                        "control_backend": "xdotool",
                        "agent_mode": "trained_fdm_policy",
                        "checkpoint": {"path": checkpoint_path, "exists": True},
                        "adapter_config": {"path": adapter_config_path, "exists": True},
                    },
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
            "statistical_comparison_summary": {
                "method": "paired_bootstrap_holm",
                "baseline_name": "random_or_noop_smoke_baseline",
                "adjusted_p_value": 0.01,
                "effect_size": 0.25,
                "agent_mean_score": 10.0,
                "baseline_mean_score": 1.0,
                "mean_score_delta": 9.0,
                "episode_count": 15,
                "holm_adjusted_p_lt_0_05": True,
            },
            "findings": [],
        },
    )


def test_g008_completion_audit_passes_on_live_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g008_live_suite_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert len(payload["episode_artifact_paths"]) == 63  # 15 * 4 episode artifacts + checkpoint + adapter + stats


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


def test_g008_completion_audit_rejects_missing_strong_stats_summary(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    validation_path = tmp_path / cfg["paths"]["evidence_validation"]
    payload = json.loads(validation_path.read_text())
    payload.pop("statistical_comparison_summary")
    write_json(validation_path, payload)
    audit = validate_g008_live_suite_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in audit["findings"]}
    assert audit["status"] == "fail"
    assert "missing_live_suite_strong_statistical_bar" in codes
    assert "live_suite_adjusted_p_value_not_significant" in codes


def test_g008_completion_audit_requires_passing_d2e_only_audits(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    g003_path = tmp_path / cfg["paths"]["g003_completion_audit"]
    g003 = json.loads(g003_path.read_text())
    g003["status"] = "fail"
    g003["error_count"] = 2
    g003["data_universe_counts"]["source_ids"] = {"d2e_480p": 3}
    g003["decode_counts_by_resolution_tier"] = {"480p": 3}
    write_json(g003_path, g003)
    g004_path = tmp_path / cfg["paths"]["g004_completion_audit"]
    g004 = json.loads(g004_path.read_text())
    g004["data_universe_counts"]["included_recording_variants"] = 2
    g004["data_universe_counts"]["resolution_tiers"] = {"480p": 2}
    write_json(g004_path, g004)

    payload = validate_g008_live_suite_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "d2e_only_completion_audit_not_pass" in codes
    assert "d2e_only_audit_included_variants_mismatch" in codes
    assert "d2e_only_audit_source_count_mismatch" in codes
    assert "d2e_only_audit_resolution_tier_count_mismatch" in codes
    assert "d2e_only_audit_decode_resolution_tier_count_mismatch" in codes
    assert payload["d2e_only_audit_report"]["g003"]["status"] == "fail"


def test_g008_completion_audit_requires_g005_for_aux_checkpoint(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    goals = json.loads((tmp_path / cfg["goals_path"]).read_text())
    goals["goals"].append({"id": "G005", "status": "pending"})
    write_json(tmp_path / cfg["goals_path"], goals)
    metadata_path = tmp_path / cfg["paths"]["trained_checkpoint_metadata"]
    metadata = json.loads(metadata_path.read_text())
    metadata["source_namespace"] = "d2e_aux"
    write_json(metadata_path, metadata)
    write_json(tmp_path / cfg["paths"]["g005_completion_audit"], {"schema": "g005_aux_completion_audit.v1", "status": "fail", "error_count": 4})

    payload = validate_g008_live_suite_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "aux_checkpoint_requires_g005_complete" in codes
    assert "g005_completion_audit_not_pass_for_aux_checkpoint" in codes
