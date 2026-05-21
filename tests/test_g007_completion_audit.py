from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json, write_jsonl
from fdm_d2e.reporting.g007_completion import validate_g007_completion


def _config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G007",
        "allowed_contract_modes": ["deterministic_replay_dry_run"],
        "expected_blocked_actions": 0,
        "min_applied_actions": 2,
        "min_demo_targets": 2,
        "paths": {
            "contract_evidence": "artifacts/runtime/contract.json",
            "fixture_predictions": "artifacts/runtime/predictions.jsonl",
            "contract_config": "configs/runtime/fixture.json",
            "demo_config": "configs/runtime/demo.json",
            "runtime_doc": "docs/runtime.md",
        },
        "contract_expectations": {"schema": "runtime_replay_adapter.v1", "mode": "deterministic_replay_dry_run", "safety.require_focus": True, "latency.schema": "runtime_latency_summary.v1"},
        "fixture_config_expectations": {"schema": "runtime_replay_adapter_config.v1", "safety.require_focus": True},
        "demo_config_expectations": {"schema": "runtime_replay_adapter_config.v1", "safety.require_focus": True},
        "required_contract_note_phrases": ["no OS-level input injection", "commercial-game control"],
        "required_doc_phrases": ["by itself prove live game control", "No G008 live-suite claim"],
    }


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(root / cfg["goals_path"], {"goals": [{"id": "G007", "status": "complete"}]})
    write_jsonl(root / cfg["paths"]["fixture_predictions"], [{"predicted_tokens": ["KEY_PRESS_87"]}, {"predicted_tokens": ["MOUSE_LEFT_DOWN"]}])
    write_json(
        root / cfg["paths"]["contract_evidence"],
        {
            "schema": "runtime_replay_adapter.v1",
            "mode": "deterministic_replay_dry_run",
            "applied_actions": 2,
            "blocked_actions": 0,
            "safety": {"require_focus": True, "kill_switch_path": "outputs/runtime/KILL"},
            "latency": {"schema": "runtime_latency_summary.v1"},
            "adapter_targets": [{"id": "dry", "claim_boundary": "deterministic_sdk_contract_only"}],
            "notes": "Safe replay SDK evidence only; no OS-level input injection or commercial-game control claim.",
        },
    )
    write_json(root / cfg["paths"]["contract_config"], {"schema": "runtime_replay_adapter_config.v1", "safety": {"require_focus": True}})
    write_json(
        root / cfg["paths"]["demo_config"],
        {
            "schema": "runtime_replay_adapter_config.v1",
            "safety": {"require_focus": True},
            "adapter_targets": [
                {"id": "a", "claim_boundary": "open_source_offline_target_candidate", "license_probe_required": True},
                {"id": "b", "claim_boundary": "open_source_offline_target_candidate", "license_probe_required": True},
            ],
        },
    )
    doc = root / cfg["paths"]["runtime_doc"]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("This does not by itself prove live game control. No G008 live-suite claim is made here.")


def test_g007_completion_audit_passes_on_contract_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g007_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0


def test_g007_completion_audit_fails_on_live_claim_boundary_regression(tmp_path: Path):
    _complete_fixture(tmp_path)
    cfg = _config()
    write_json(tmp_path / cfg["goals_path"], {"goals": [{"id": "G007", "status": "pending"}]})
    contract = {
        "schema": "runtime_replay_adapter.v1",
        "mode": "live_desktop_control",
        "applied_actions": 0,
        "blocked_actions": 1,
        "safety": {"require_focus": False},
        "latency": {"schema": "wrong"},
        "adapter_targets": [{"id": "live", "claim_boundary": "live_game_claim"}],
        "notes": "unsafe positive live claim",
    }
    write_json(tmp_path / cfg["paths"]["contract_evidence"], contract)
    payload = validate_g007_completion(cfg, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "unexpected_runtime_contract_mode" in codes
    assert "runtime_contract_boundary_missing" in codes
    assert "runtime_focus_guard_not_enabled" in codes
