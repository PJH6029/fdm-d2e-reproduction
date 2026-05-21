from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.rollout.live_suite import planned_open_source_games, suite_protocol_report, validate_live_suite_evidence


CONFIG_PATH = Path("configs/harness/g008_live_open_game_suite.yaml")


def test_live_open_game_suite_protocol_requires_three_graphical_games():
    config = load_config(CONFIG_PATH)
    games = planned_open_source_games(config)
    assert [game["id"] for game in games] == ["supertuxkart", "luanti_minetest", "xonotic"]
    report = suite_protocol_report(config)
    assert report["status"] == "protocol_ready"
    assert report["quality_gate"]["status"] == "protocol_ready"
    assert report["quality_gate"]["planned_games"] == 3
    assert report["quality_gate"]["planned_tasks"] == 3
    assert report["quality_gate"]["planned_seeded_episodes"] == 15
    assert "live_desktop_control" in report["quality_gate"]["allowed_evidence_modes"]
    assert "not live harness success" in report["claim_boundary"]


def _write(path: Path, text: str = "evidence") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _stats_payload(episode_count: int = 15) -> dict:
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


def _passing_evidence(tmp_path: Path, config: dict) -> dict:
    episodes = []
    checkpoint = _write(tmp_path / "evidence" / "trained_fdm_checkpoint.pt", "checkpoint")
    adapter_config = _write(tmp_path / "evidence" / "runtime_adapter.yaml", "adapter")
    for game in planned_open_source_games(config):
        for task in game["tasks"]:
            for seed in task["seeds"]:
                prefix = tmp_path / "evidence" / game["id"] / task["id"] / f"seed_{seed}"
                episodes.append(
                    {
                        "game_id": game["id"],
                        "task_id": task["id"],
                        "seed": seed,
                        "status": "pass",
                        "score": 10.0 + seed,
                        "baseline_score": 1.0,
                        "latency": {"p50_ms": 18.0, "p95_ms": 42.0},
                        "runtime": {
                            "control_backend": "xdotool",
                            "agent_mode": "trained_fdm_policy",
                            "process_name": game["id"],
                            "window_title": game["name"],
                            "checkpoint_path": checkpoint,
                            "adapter_config_path": adapter_config,
                            "action_count": 12,
                            "started_at_unix": 1000.0 + seed,
                            "ended_at_unix": 1010.0 + seed,
                        },
                        "video_path": _write(prefix / "episode.mp4", "video"),
                        "replay_path": _write(prefix / "replay.jsonl", "replay"),
                        "latency_log_path": _write(prefix / "latency.jsonl", "latency"),
                        "failure_log_path": _write(prefix / "failures.jsonl", "[]"),
                    }
                )
    stats = _write(tmp_path / "evidence" / "statistical_comparison.json", json.dumps(_stats_payload(len(episodes))))
    return {
        "schema": "live_game_suite_evidence.v1",
        "evidence_mode": "live_desktop_control",
        "episodes": episodes,
        "statistical_comparison": {"path": stats, "holm_adjusted_p_lt_0_05": True},
    }


def test_live_suite_evidence_passes_only_with_videos_replays_latency_failures_and_stats(tmp_path):
    config = load_config(CONFIG_PATH)
    result = validate_live_suite_evidence(config, _passing_evidence(tmp_path, config))
    assert result["quality_gate"]["status"] == "pass"
    assert result["quality_gate"]["games_with_passed_episode"] == 3
    assert result["quality_gate"]["passed_tasks"] == 3
    assert result["quality_gate"]["episodes_observed"] == 15
    assert result["statistical_comparison_artifact"]["exists"] is True
    assert result["statistical_comparison_summary"]["adjusted_p_value"] == 0.01
    assert not result["findings"]


def test_live_suite_rejects_empty_statistical_comparison_artifact(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    Path(evidence["statistical_comparison"]["path"]).write_text("{}")
    result = validate_live_suite_evidence(config, evidence)
    codes = {item["code"] for item in result["findings"]}
    assert result["quality_gate"]["status"] == "fail"
    assert "invalid_or_empty_statistical_comparison_artifact" in codes
    assert "adjusted_p_value_not_significant" in codes


def test_live_suite_rejects_non_object_statistical_metadata(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    evidence["statistical_comparison"] = "not-a-dict"
    result = validate_live_suite_evidence(config, evidence)
    codes = {item["code"] for item in result["findings"]}
    assert result["quality_gate"]["status"] == "fail"
    assert "invalid_statistical_comparison_metadata" in codes


def test_live_suite_rejects_dry_run_or_game_adjacent_evidence(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    evidence["evidence_mode"] = "game_adjacent"
    result = validate_live_suite_evidence(config, evidence)
    assert result["quality_gate"]["status"] == "fail"
    assert any(item["code"] == "non_live_evidence_mode" for item in result["findings"])
    assert any(item["code"] == "evidence_mode_not_allowed" for item in result["findings"])


def test_live_suite_requires_explicit_allowed_live_evidence_mode(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    evidence.pop("evidence_mode")
    result = validate_live_suite_evidence(config, evidence)
    assert result["quality_gate"]["status"] == "fail"
    assert any(item["code"] == "evidence_mode_not_allowed" for item in result["findings"])


def test_live_suite_rejects_missing_required_video(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    first = evidence["episodes"][0]
    Path(first["video_path"]).unlink()
    result = validate_live_suite_evidence(config, evidence)
    assert result["quality_gate"]["status"] == "fail"
    assert any(item["code"] == "missing_episode_artifact" and item["artifact"] == "video_path" for item in result["findings"])


def test_live_suite_requires_live_runtime_metadata(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    first = evidence["episodes"][0]
    first.pop("runtime")
    result = validate_live_suite_evidence(config, evidence)
    codes = {item["code"] for item in result["findings"]}
    assert result["quality_gate"]["status"] == "fail"
    assert "missing_episode_runtime_metadata" in codes
    assert "control_backend_not_allowed" in codes
    assert "missing_runtime_checkpoint_artifact" in codes


def test_live_suite_rejects_replay_control_backend(tmp_path):
    config = load_config(CONFIG_PATH)
    evidence = _passing_evidence(tmp_path, config)
    evidence["episodes"][0]["runtime"]["control_backend"] = "deterministic_replay"
    result = validate_live_suite_evidence(config, evidence)
    codes = {item["code"] for item in result["findings"]}
    assert result["quality_gate"]["status"] == "fail"
    assert "non_live_control_backend" in codes
    assert "control_backend_not_allowed" in codes
