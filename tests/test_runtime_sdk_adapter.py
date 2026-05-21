from __future__ import annotations

from pathlib import Path

from fdm_d2e.io_utils import write_jsonl
from fdm_d2e.runtime.sdk import ActionDecoder, RuntimeSafetyConfig, SafeActionAdapter, run_replay_adapter


def test_action_decoder_clamps_mouse_and_rejects_disallowed_keys():
    decoder = ActionDecoder(RuntimeSafetyConfig(max_mouse_delta_per_frame=2.0, allowed_keys={"87"}))
    action = decoder.decode(
        {
            "timestamp_ns": 0,
            "predicted_tokens": ["KEY_PRESS_87", "KEY_PRESS_999", "MOUSE_DX_P5", "MOUSE_DY_N5", "MOUSE_LEFT_DOWN"],
        },
        step=0,
    )

    assert action.key_presses == ["87"]
    assert action.mouse_dx == 2.0
    assert action.mouse_dy == -2.0
    assert action.mouse_buttons == ["MOUSE_LEFT"]
    assert action.valid is False
    assert action.blocked_reason == "unknown_or_disallowed_tokens"


def test_safe_adapter_focus_guard_kill_switch_and_rate_limit(tmp_path: Path):
    safety = RuntimeSafetyConfig(
        allowed_window_title_patterns=["allowed-game"],
        kill_switch_path=str(tmp_path / "KILL"),
        max_actions_per_second=10.0,
    )
    decoder = ActionDecoder(safety)
    adapter = SafeActionAdapter(safety=safety, focus_title_provider=lambda: "wrong-window")

    blocked_focus = adapter.apply(decoder.decode({"timestamp_ns": 0, "predicted_tokens": ["KEY_PRESS_87"]}, step=0))
    assert blocked_focus["reason"] == "focus_guard_failed"

    adapter = SafeActionAdapter(safety=safety, focus_title_provider=lambda: "allowed-game")
    first = adapter.apply(decoder.decode({"timestamp_ns": 0, "predicted_tokens": ["KEY_PRESS_87"]}, step=0))
    assert first["status"] == "applied"
    second = adapter.apply(decoder.decode({"timestamp_ns": 10_000_000, "predicted_tokens": ["KEY_PRESS_87"]}, step=1))
    assert second["reason"] == "rate_limited"

    (tmp_path / "KILL").write_text("stop\n")
    killed = adapter.apply(decoder.decode({"timestamp_ns": 200_000_000, "predicted_tokens": ["KEY_PRESS_87"]}, step=2))
    assert killed["reason"] == "kill_switch_active"


def test_replay_adapter_writes_latency_schema(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "runtime.json"
    write_jsonl(
        predictions,
        [
            {"timestamp_ns": 0, "predicted_tokens": ["KEY_PRESS_87", "MOUSE_DX_P1"]},
            {"timestamp_ns": 20_000_000, "predicted_tokens": ["KEY_RELEASE_87", "MOUSE_LEFT_DOWN"]},
        ],
    )

    result = run_replay_adapter(
        {
            "predictions_path": str(predictions),
            "output_path": str(output),
            "action_limit": 2,
            "safety": {
                "require_focus": True,
                "allowed_window_title_patterns": ["demo-title"],
                "max_actions_per_second": 100.0,
            },
        },
        focus_title_provider=lambda: "demo-title",
    )

    assert result["schema"] == "runtime_replay_adapter.v1"
    assert result["num_actions"] == 2
    assert result["blocked_actions"] == 0
    assert result["latency"]["schema"] == "runtime_latency_summary.v1"
    assert result["latency"]["p50_ms"] is not None
    assert output.exists()
