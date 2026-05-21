from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from tests.test_g006_readiness_planner import _write_configs, _write_ready_fixture
from watch_g006_then_finalize import watch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/eval/watcher.json",
        "allow_fail": False,
        "once": True,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "build_config": "configs/eval/g006_build.json",
        "build_summary_out": "artifacts/eval/g006_build_summary.json",
        "readiness_config": "configs/eval/g006_readiness.json",
        "readiness_output": "artifacts/eval/g006_readiness.json",
        "g006_completion_config": "configs/eval/g006_completion.json",
        "g006_audit_output": "artifacts/eval/g006_completion.json",
        "g006_finalization_summary": "artifacts/eval/g006_finalize.json",
        "readiness_plan_output": "artifacts/eval/g006_plan.json",
        "require_existing_final_outputs": False,
        "allow_precheckpoint": False,
        "skip_build": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_g006_watcher_waits_until_inputs_are_ready(tmp_path: Path):
    _write_configs(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "in_progress"}]})
    calls: list[Namespace] = []
    payload = watch(_args(tmp_path), finalize_func=lambda ns: calls.append(ns) or {"status": "pass"})
    assert payload["status"] == "waiting_for_g006_inputs"
    assert payload["readiness_status"] == "blocked"
    assert calls == []
    assert (tmp_path / "artifacts/eval/g006_plan.json").exists()
    assert (tmp_path / "artifacts/eval/watcher.json").exists()


def test_g006_watcher_finalizes_when_inputs_are_ready(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    calls: list[Namespace] = []

    def fake_finalize(ns: Namespace) -> dict:
        calls.append(ns)
        return {"status": "pass", "readiness_status": "pass", "g006_audit_status": "pass", "g006_audit_error_count": 0}

    payload = watch(_args(tmp_path), finalize_func=fake_finalize)
    assert payload["status"] == "finalized_pass"
    assert payload["g006_audit_status"] == "pass"
    assert len(calls) == 1
    assert calls[0].summary_out == "artifacts/eval/g006_finalize.json"
    assert calls[0].build_config == "configs/eval/g006_build.json"


def test_g006_watcher_reports_failed_finalization(tmp_path: Path):
    _write_ready_fixture(tmp_path)
    payload = watch(_args(tmp_path), finalize_func=lambda ns: {"status": "fail", "readiness_status": "fail", "g006_audit_status": "fail", "g006_audit_error_count": 3})
    assert payload["status"] == "finalized_fail"
    assert payload["findings"][0]["code"] == "g006_finalization_not_pass"
