from __future__ import annotations

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


def _passing_evidence(tmp_path: Path, config: dict) -> dict:
    episodes = []
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
                        "video_path": _write(prefix / "episode.mp4", "video"),
                        "replay_path": _write(prefix / "replay.jsonl", "replay"),
                        "latency_log_path": _write(prefix / "latency.jsonl", "latency"),
                        "failure_log_path": _write(prefix / "failures.jsonl", "[]"),
                    }
                )
    stats = _write(tmp_path / "evidence" / "statistical_comparison.json", "{}")
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
    assert not result["findings"]


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
