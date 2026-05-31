from __future__ import annotations

from fdm_d2e.tokenization.fdm1_actions import (
    ActionSlotTokenizer,
    MouseMoveBinner,
    bin_events,
    detokenize_mouse_move,
    fit_signed_exponential_boundaries,
    next_click_position_targets,
    summarize_slot_overflow,
    token_from_discrete_event,
)


def test_fdm1_keyboard_mouse_scroll_event_tokens_use_press_release_semantics():
    assert token_from_discrete_event({"type": "keyboard", "event_type": "press", "vk": 87}) == "KEY_DOWN_87"
    assert token_from_discrete_event({"type": "keyboard", "event_type": "release", "key": "space"}) == "KEY_UP_SPACE"
    assert token_from_discrete_event({"type": "mouse_button", "event_type": "press", "button": "left"}) == "MOUSE_LEFT_DOWN"
    assert token_from_discrete_event({"type": "mouse_button", "event_type": "release", "button": "right"}) == "MOUSE_RIGHT_UP"
    assert token_from_discrete_event({"type": "scroll", "dy": -1}) == "SCROLL_DOWN"
    assert token_from_discrete_event({"type": "scroll", "dx": 2}) == "SCROLL_RIGHT"


def test_mouse_binner_has_49_axis_bins_and_compound_move_token():
    binner = MouseMoveBinner()
    labels = {binner.axis_label(v) for v in range(-5000, 5001)}
    assert "Z00" in labels
    assert "P24" in labels
    assert "N24" in labels
    assert len({"Z00"} | {f"P{i:02d}" for i in range(1, 25)} | {f"N{i:02d}" for i in range(1, 25)}) == 49
    assert binner.tokenize(5, -9) == ["MOUSE_MOVE_BIN_P05_N07"]
    assert detokenize_mouse_move("MOUSE_MOVE_BIN_P05_N07", binner=binner) is not None


def test_bin_events_uses_non_overlapping_50ms_bins():
    events = [
        {"type": "keyboard", "event_type": "press", "vk": 87, "timestamp_ns": 0},
        {"type": "keyboard", "event_type": "release", "vk": 87, "timestamp_ns": 49_999_999},
        {"type": "keyboard", "event_type": "press", "vk": 65, "timestamp_ns": 50_000_000},
    ]
    bins = bin_events(events, start_ns=0, bin_ms=50)
    assert len(bins) == 2
    assert [len(row["events"]) for row in bins] == [2, 1]


def test_action_slot_serialization_orders_motion_then_priority_events_and_no_action_padding():
    events = [
        {"type": "keyboard", "event_type": "release", "vk": 87, "timestamp_ns": 30},
        {"type": "mouse_move", "dx": 3, "dy": -2, "timestamp_ns": 10},
        {"type": "keyboard", "event_type": "press", "vk": 65, "timestamp_ns": 20},
        {"type": "mouse_button", "event_type": "press", "button": "left", "timestamp_ns": 40},
    ]
    row = ActionSlotTokenizer(k_event_slots=4).serialize_bin(events)
    assert row["action_tokens"][0] == "MOUSE_MOVE_BIN_P03_N02"
    assert row["event_slots"] == ["MOUSE_LEFT_DOWN", "KEY_DOWN_65", "KEY_UP_87", "NO_ACTION"]
    assert row["overflow_count"] == 0
    assert ActionSlotTokenizer(k_event_slots=4).mask_for_idm(row) == ["MOUSE_MOVE_BIN_P03_N02", "MASK_ACTION", "MASK_ACTION", "MASK_ACTION", "MASK_ACTION"]


def test_overflow_reserves_final_slot_and_reports_rate():
    events = [
        {"type": "mouse_button", "event_type": "press", "button": "left", "timestamp_ns": 1},
        {"type": "keyboard", "event_type": "press", "vk": 87, "timestamp_ns": 2},
        {"type": "keyboard", "event_type": "release", "vk": 87, "timestamp_ns": 3},
        {"type": "scroll", "dy": 1, "timestamp_ns": 4},
    ]
    row = ActionSlotTokenizer(k_event_slots=3).serialize_bin(events)
    assert row["event_slots"] == ["MOUSE_LEFT_DOWN", "KEY_DOWN_87", "EVENT_OVERFLOW"]
    assert row["overflow_count"] == 1
    summary = summarize_slot_overflow([row], by_game=["Toy"])
    assert summary["overflow_rate"] == 1.0
    assert summary["per_game"]["Toy"]["overflow_events"] == 1


def test_next_click_position_targets_use_future_click_horizon():
    bins = [
        {"events": []},
        {"events": [{"type": "mouse_button", "event_type": "press", "button": "left", "x": 427, "y": 240}]},
        {"events": []},
    ]
    targets = next_click_position_targets(bins, horizon_bins=1, grid_width=32, grid_height=18, screen_width=854, screen_height=480)
    assert targets[0] == "NEXT_CLICK_POSITION_BIN_16_9"
    assert targets[1] == "NEXT_CLICK_POSITION_BIN_16_9"
    assert targets[2] == "NO_CLICK_WITHIN_H"


def test_fit_signed_exponential_boundaries_is_monotonic_and_24_bins():
    boundaries = fit_signed_exponential_boundaries([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    assert len(boundaries) == 24
    assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
