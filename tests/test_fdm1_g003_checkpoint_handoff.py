from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_fdm1_g003_checkpoint_handoff import GOAL_ID, build_handoff


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _fixture(root: Path, *, monitor_status: str = "pass", audit_status: str = "pass", bundle_status: str = "pass", goal_status: str = "in_progress") -> dict:
    objective = "Complete the durable ultragoal plan in .omx/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .omx/ultragoal/ledger.jsonl as the audit trail."
    _write_json(root / "artifacts/cluster/fdm1_g003_action_dataset_pod_monitor.json", {"status": monitor_status})
    _write_json(root / "artifacts/sources/fdm1_g003_action_dataset_completion_audit.json", {"status": audit_status})
    _write_json(root / "artifacts/sources/fdm1_g003_evidence_bundle_manifest.json", {"status": bundle_status})
    _write_json(root / ".omx/ultragoal/goals.json", {"codexObjective": objective, "goals": [{"id": GOAL_ID, "status": goal_status}]})
    _write_json(root / "goal.json", {"goal": {"status": "active", "objective": objective}})
    return {"objective": objective, "codex_goal_json": "goal.json"}


def test_handoff_ready_when_all_gates_pass(tmp_path: Path):
    fixture = _fixture(tmp_path)
    payload = build_handoff(root=tmp_path, codex_goal_json=fixture["codex_goal_json"])
    assert payload["status"] == "ready_to_checkpoint"
    assert payload["goal_id"] == GOAL_ID
    assert "omx" == payload["checkpoint_command"][0]
    assert "--codex-goal-json" in payload["checkpoint_command"]
    assert "status=pass" in payload["evidence"]


def test_handoff_blocks_when_monitor_not_pass(tmp_path: Path):
    fixture = _fixture(tmp_path, monitor_status="running")
    payload = build_handoff(root=tmp_path, codex_goal_json=fixture["codex_goal_json"])
    assert payload["status"] == "blocked"
    assert any(f["code"] == "monitor_not_pass" for f in payload["findings"])


def test_handoff_blocks_codex_goal_mismatch(tmp_path: Path):
    _fixture(tmp_path)
    _write_json(tmp_path / "goal.json", {"goal": {"status": "active", "objective": "wrong"}})
    payload = build_handoff(root=tmp_path, codex_goal_json="goal.json")
    assert payload["status"] == "blocked"
    assert any(f["code"] == "codex_goal_objective_mismatch" for f in payload["findings"])


def test_cli_allows_blocked_output(tmp_path: Path):
    _fixture(tmp_path, audit_status="fail")
    output = tmp_path / "handoff.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_fdm1_g003_checkpoint_handoff.py",
            "--root",
            str(tmp_path),
            "--codex-goal-json",
            "goal.json",
            "--output",
            str(output),
            "--allow-blocked",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "status=blocked" in completed.stdout
    assert json.loads(output.read_text())["status"] == "blocked"
