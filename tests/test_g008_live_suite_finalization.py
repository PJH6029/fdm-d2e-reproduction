from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from finalize_g008_live_suite import finalize


GAMES = ["supertuxkart", "luanti_minetest", "xonotic"]
SPLITS = {"min_games": 3, "min_tasks": 3, "min_episodes": 15}


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "summary_out": "artifacts/harness/g008_finalize.json",
        "allow_fail": False,
        "suite_config": "configs/harness/g008_suite.json",
        "g008_completion_config": "configs/eval/g008_completion.json",
        "g008_audit_output": "artifacts/harness/g008_completion_audit.json",
        "evidence": None,
        "evidence_validation_output": None,
        "protocol_output": None,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_text(root: Path, rel_path: str, text: str = "evidence") -> str:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return rel_path


def _stats_payload(episode_count: int) -> dict:
    return {
        "schema": "live_suite_statistical_comparison.v1",
        "method": "paired_bootstrap_holm",
        "baseline_name": "random_or_noop_smoke_baseline",
        "adjusted_p_value": 0.01,
        "effect_size": 0.25,
        "agent_mean_score": 10.0,
        "baseline_mean_score": 1.0,
        "episode_count": episode_count,
        "holm_adjusted_p_lt_0_05": True,
    }


def _suite_config() -> dict:
    return {
        "schema": "live_game_suite_config.v1",
        "suite_id": "fixture_suite",
        "output_path": "artifacts/harness/g008_protocol.json",
        "allowed_evidence_modes": ["live_desktop_control", "live_graphical_game_control"],
        "thresholds": {
            "min_games": 3,
            "min_tasks": 3,
            "min_seeds_per_task": 5,
            "min_episode_pass_rate": 0.60,
            "min_task_pass_rate": 0.67,
            "max_p95_latency_ms": 250,
            "min_baseline_win_rate": 0.50,
            "min_action_count_per_episode": 1,
            "require_video": True,
            "require_replay": True,
            "require_latency_log": True,
            "require_failure_log": True,
            "require_statistical_comparison": True,
            "require_runtime_metadata": True,
            "require_window_title_match": True,
        },
        "games": [
            {
                "id": game,
                "name": game,
                "open_source": True,
                "graphical": True,
                "offline_capable": True,
                "license": "open",
                "source_url": f"https://example.invalid/{game}",
                "window_title_pattern": game,
                "tasks": [{"id": f"{game}_task", "seeds": [0, 1, 2, 3, 4]}],
            }
            for game in GAMES
        ],
    }


def _completion_config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G008",
        "prerequisite_goals": ["G003", "G004", "G007"],
        "thresholds": SPLITS,
        "expected_recording_variants": 3,
        "require_d2e_only_completion_audits_pass": True,
        "require_g005_for_aux_checkpoint": True,
        "g005_goal_id": "G005",
        "expected_variants_by_source": {"d2e_480p": 2, "d2e_original": 1},
        "expected_variants_by_resolution_tier": {"480p": 2, "original_fhd_qhd": 1},
        "allowed_checkpoint_namespaces": ["d2e_full_corpus", "d2e_aux"],
        "allowed_evidence_modes": ["live_desktop_control", "live_graphical_game_control"],
        "require_goal_checkpoint_complete": False,
        "paths": {
            "suite_config": "configs/harness/g008_suite.json",
            "evidence_validation": "artifacts/harness/g008_validation.json",
            "trained_checkpoint_metadata": "outputs/fdm/checkpoint_metadata.json",
            "g003_completion_audit": "artifacts/idm/g003_audit.json",
            "g004_completion_audit": "artifacts/fdm/g004_audit.json",
            "g005_completion_audit": "artifacts/aux/g005_audit.json",
            "runtime_adapter_contract": "artifacts/runtime/contract.json",
            "live_suite_doc": "docs/live.md",
        },
        "validation_expectations": {"schema": "live_game_suite_evidence_validation.v1", "quality_gate.status": "pass"},
        "checkpoint_expectations": {
            "oracle_ground_truth_control": False,
            "data_universe.exists": True,
            "split_contract.exists": True,
        },
    }


def _write_base_fixture(root: Path) -> None:
    write_json(root / "configs/harness/g008_suite.json", _suite_config())
    write_json(root / "configs/eval/g008_completion.json", _completion_config())
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
    write_json(root / "artifacts/runtime/contract.json", {"status": "pass"})
    _write_text(root, "docs/live.md", "live suite doc")
    d2e_audit_counts = {
        "included_recording_variants": 3,
        "source_ids": {"d2e_480p": 2, "d2e_original": 1},
        "resolution_tiers": {"480p": 2, "original_fhd_qhd": 1},
    }
    write_json(
        root / "artifacts/idm/g003_audit.json",
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
        root / "artifacts/fdm/g004_audit.json",
        {
            "schema": "g004_full_fdm_completion_audit.v1",
            "status": "pass",
            "error_count": 0,
            "data_universe_counts": d2e_audit_counts,
        },
    )
    write_json(root / "artifacts/aux/g005_audit.json", {"schema": "g005_aux_completion_audit.v1", "status": "pass", "error_count": 0})
    write_json(
        root / "outputs/fdm/checkpoint_metadata.json",
        {
            "source_namespace": "d2e_full_corpus",
            "oracle_ground_truth_control": False,
            "data_universe": {"exists": True},
            "split_contract": {"exists": True},
        },
    )


def _write_passing_evidence(root: Path) -> str:
    episodes = []
    checkpoint = _write_text(root, "artifacts/harness/live/trained_fdm_checkpoint.pt", "checkpoint")
    adapter_config = _write_text(root, "artifacts/harness/live/runtime_adapter.yaml", "adapter")
    for game in GAMES:
        task_id = f"{game}_task"
        for seed in range(5):
            prefix = f"artifacts/harness/live/{game}/{task_id}/seed_{seed}"
            episodes.append(
                {
                    "game_id": game,
                    "task_id": task_id,
                    "seed": seed,
                    "status": "pass",
                    "score": 10.0 + seed,
                    "baseline_score": 1.0,
                    "latency": {"p50_ms": 20.0, "p95_ms": 45.0},
                    "runtime": {
                        "control_backend": "xdotool",
                        "agent_mode": "trained_fdm_policy",
                        "process_name": game,
                        "window_title": game,
                        "checkpoint_path": checkpoint,
                        "adapter_config_path": adapter_config,
                        "action_count": 8,
                        "started_at_unix": 2000.0 + seed,
                        "ended_at_unix": 2010.0 + seed,
                    },
                    "video_path": _write_text(root, f"{prefix}/episode.mp4", "video"),
                    "replay_path": _write_text(root, f"{prefix}/replay.jsonl", "{}\n"),
                    "latency_log_path": _write_text(root, f"{prefix}/latency.jsonl", "{}\n"),
                    "failure_log_path": _write_text(root, f"{prefix}/failures.jsonl", "[]\n"),
                }
            )
    stats_path = _write_text(root, "artifacts/harness/live/statistical_comparison.json", json.dumps(_stats_payload(len(episodes))))
    evidence_path = "artifacts/harness/live/evidence.json"
    write_json(
        root / evidence_path,
        {
            "schema": "live_game_suite_evidence.v1",
            "evidence_mode": "live_desktop_control",
            "episodes": episodes,
            "statistical_comparison": {"path": stats_path, "holm_adjusted_p_lt_0_05": True},
        },
    )
    return evidence_path


def test_finalize_g008_reports_protocol_only_as_non_terminal(tmp_path: Path):
    _write_base_fixture(tmp_path)
    payload = finalize(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert payload["protocol_status"] == "protocol_ready"
    assert "missing_live_evidence" in codes
    assert "g008_completion_audit_not_pass" in codes
    assert (tmp_path / "artifacts/harness/g008_protocol.json").exists()
    assert (tmp_path / "artifacts/harness/g008_completion_audit.json").exists()


def test_finalize_g008_validates_live_evidence_and_completion_audit(tmp_path: Path):
    _write_base_fixture(tmp_path)
    evidence_path = _write_passing_evidence(tmp_path)
    payload = finalize(_args(tmp_path, evidence=evidence_path))
    assert payload["status"] == "pass"
    assert payload["evidence_validation_status"] == "pass"
    assert payload["g008_audit_status"] == "pass"
    validation = json.loads((tmp_path / "artifacts/harness/g008_validation.json").read_text())
    assert validation["quality_gate"]["status"] == "pass"
    audit = json.loads((tmp_path / "artifacts/harness/g008_completion_audit.json").read_text())
    assert audit["status"] == "pass"
