from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fdm_d2e.io_utils import write_json


SCRIPT = Path("scripts/validate_live_game_suite.py")


def _rel_file(root: Path, rel_path: str, text: str = "evidence") -> str:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel_path


def _stats_payload() -> dict:
    return {
        "schema": "live_suite_statistical_comparison.v1",
        "method": "paired_bootstrap_holm",
        "baseline_name": "random_or_noop_smoke_baseline",
        "adjusted_p_value": 0.01,
        "effect_size": 0.2,
        "agent_mean_score": 2.0,
        "baseline_mean_score": 1.0,
        "episode_count": 1,
        "holm_adjusted_p_lt_0_05": True,
    }


def _write_config(root: Path) -> Path:
    config_path = root / "suite.json"
    write_json(
        config_path,
        {
            "schema": "live_game_suite_config.v1",
            "suite_id": "cli_fixture",
            "output_path": str(root / "default_protocol.json"),
            "allowed_evidence_modes": ["live_desktop_control"],
            "allowed_control_backends": ["xdotool"],
            "thresholds": {
                "min_games": 1,
                "min_tasks": 1,
                "min_seeds_per_task": 1,
                "min_episode_pass_rate": 1.0,
                "min_task_pass_rate": 1.0,
                "max_p95_latency_ms": 100,
                "min_baseline_win_rate": 1.0,
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
                    "id": "fixture_game",
                    "name": "Fixture Game",
                    "open_source": True,
                    "graphical": True,
                    "offline_capable": True,
                    "license": "test-open",
                    "source_url": "https://example.invalid/fixture",
                    "window_title_pattern": "Fixture Game",
                    "tasks": [{"id": "fixture_task", "seeds": [0]}],
                }
            ],
        },
    )
    return config_path


def _write_evidence(root: Path, *, evidence_mode: str = "live_desktop_control") -> Path:
    evidence_path = root / "evidence.json"
    write_json(
        evidence_path,
        {
            "schema": "live_game_suite_evidence.v1",
            "evidence_mode": evidence_mode,
            "episodes": [
                {
                    "game_id": "fixture_game",
                    "task_id": "fixture_task",
                    "seed": 0,
                    "status": "pass",
                    "score": 2.0,
                    "baseline_score": 1.0,
                    "latency": {"p50_ms": 10, "p95_ms": 20},
                    "runtime": {
                        "control_backend": "xdotool",
                        "agent_mode": "trained_fdm_policy",
                        "process_name": "fixture_game",
                        "window_title": "Fixture Game",
                        "checkpoint_path": _rel_file(root, "artifacts/checkpoint.pt", "checkpoint"),
                        "adapter_config_path": _rel_file(root, "artifacts/adapter.yaml", "adapter"),
                        "action_count": 3,
                        "started_at_unix": 1000.0,
                        "ended_at_unix": 1010.0,
                    },
                    "video_path": _rel_file(root, "artifacts/episode.mp4", "video"),
                    "replay_path": _rel_file(root, "artifacts/replay.jsonl", "{}\n"),
                    "latency_log_path": _rel_file(root, "artifacts/latency.jsonl", "{}\n"),
                    "failure_log_path": _rel_file(root, "artifacts/failures.jsonl", "[]\n"),
                }
            ],
            "statistical_comparison": {
                "path": _rel_file(root, "artifacts/statistical_comparison.json", json.dumps(_stats_payload())),
                "holm_adjusted_p_lt_0_05": True,
            },
        },
    )
    return evidence_path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True, check=False)


def test_validate_live_game_suite_cli_writes_protocol_report(tmp_path: Path):
    config = _write_config(tmp_path)
    output = tmp_path / "protocol.json"
    result = _run_cli("--config", str(config), "--output", str(output), "--root", str(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "live_game_suite_protocol.v1"
    assert payload["quality_gate"]["status"] == "protocol_ready"


def test_validate_live_game_suite_cli_writes_passing_evidence_report(tmp_path: Path):
    config = _write_config(tmp_path)
    evidence = _write_evidence(tmp_path)
    output = tmp_path / "validation.json"
    result = _run_cli("--config", str(config), "--evidence", str(evidence), "--output", str(output), "--root", str(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "live_game_suite_evidence_validation.v1"
    assert payload["quality_gate"]["status"] == "pass"
    assert payload["episode_results"][0]["runtime"]["checkpoint"]["exists"] is True


def test_validate_live_game_suite_cli_exits_nonzero_on_invalid_evidence(tmp_path: Path):
    config = _write_config(tmp_path)
    evidence = _write_evidence(tmp_path, evidence_mode="deterministic_replay")
    output = tmp_path / "validation.json"
    result = _run_cli("--config", str(config), "--evidence", str(evidence), "--output", str(output), "--root", str(tmp_path))
    assert result.returncode == 2
    payload = json.loads(output.read_text())
    codes = {item["code"] for item in payload["findings"]}
    assert "non_live_evidence_mode" in codes
    assert payload["quality_gate"]["status"] == "fail"
