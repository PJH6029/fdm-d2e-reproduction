from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.masked_diffusion_idm import (
    FDM1_ACTION_MASK,
    FDM1_ACTION_NOOP,
    FDM1_ACTION_PAD,
    canonical_action_slot_record,
    canonical_fdm1_action_tokens,
    d2e_metric_tokens_from_fdm1_tokens,
    fdm1_mouse_axis_delta,
    corrupt_action_slots,
    fdm1_mouse_axis_class,
    fdm1_mouse_axis_token,
    iterative_unmask_counts,
    select_topk_masked,
)


def test_fdm1_mouse_axis_bins_are_symmetric_and_exponential():
    zero = fdm1_mouse_axis_class(0, screen_extent=1920)
    small = fdm1_mouse_axis_class(1, screen_extent=1920)
    large = fdm1_mouse_axis_class(960, screen_extent=1920)
    neg_small = fdm1_mouse_axis_class(-1, screen_extent=1920)
    assert zero == 24
    assert 24 < small < large <= 48
    assert neg_small == 24 - (small - 24)
    assert fdm1_mouse_axis_token("x", 1, screen_extent=1920).startswith("FDM1_MOUSE_DX_P")
    assert fdm1_mouse_axis_token("y", -1, screen_extent=1080).startswith("FDM1_MOUSE_DY_N")
    assert fdm1_mouse_axis_delta(small, screen_extent=1920) > 0
    assert fdm1_mouse_axis_delta(neg_small, screen_extent=1920) < 0


def test_canonical_tokens_preserve_event_multiplicity_and_press_release():
    row = {
        "frame": {"width": 1920, "height": 1080},
        "events": [
            {"type": "keyboard", "event_type": "press", "key": "w"},
            {"type": "keyboard", "event_type": "release", "key": "w"},
            {"type": "mouse_move", "dx": 3, "dy": -2},
            {"type": "mouse_button", "event_type": "press", "button": "left"},
            {"type": "mouse_move", "dx": 3, "dy": -2},
        ],
    }
    tokens = canonical_fdm1_action_tokens(row)
    assert tokens[0:2] == ["KEY_PRESS_W", "KEY_RELEASE_W"]
    assert tokens.count("MOUSE_LEFT_DOWN") == 1
    assert sum(token.startswith("FDM1_MOUSE_DX_") for token in tokens) == 2
    assert sum(token.startswith("FDM1_MOUSE_DY_") for token in tokens) == 2


def test_canonical_action_slots_record_overflow_and_padding():
    row = {"ground_truth_tokens": ["KEY_PRESS_A", "KEY_PRESS_B", "KEY_PRESS_C"]}
    record = canonical_action_slot_record(row, max_slots=2)
    assert record.tokens == ("KEY_PRESS_A", "KEY_PRESS_B")
    assert record.overflow_tokens == ("KEY_PRESS_C",)
    assert record.padded_tokens == record.tokens
    noop_record = canonical_action_slot_record({"ground_truth_tokens": []}, max_slots=3)
    assert noop_record.tokens == (FDM1_ACTION_NOOP,)
    assert noop_record.padded_tokens == (FDM1_ACTION_NOOP, FDM1_ACTION_PAD, FDM1_ACTION_PAD)


def test_corrupt_action_slots_masks_non_pad_targets_deterministically():
    slots = ["KEY_PRESS_A", FDM1_ACTION_PAD, "FDM1_MOUSE_DX_P01"]
    corrupted, loss_mask = corrupt_action_slots(
        slots,
        vocab=[FDM1_ACTION_PAD, FDM1_ACTION_MASK, "KEY_PRESS_A", "FDM1_MOUSE_DX_P01"],
        mask_probability=0.0,
        rng=random.Random(3),
    )
    assert any(loss_mask)
    assert loss_mask[1] is False
    assert corrupted[1] == FDM1_ACTION_PAD
    assert FDM1_ACTION_MASK in corrupted


def test_iterative_unmask_schedule_and_topk_selection():
    counts = iterative_unmask_counts(10, steps=16)
    assert sum(counts) == 10
    assert len(counts) == 16
    assert counts[-1] == 0
    assert select_topk_masked([0.2, 0.9, 0.9, 0.1], [True, True, True, False], k=2) == [1, 2]


def test_fdm1_tokens_convert_back_to_d2e_metric_tokens():
    tokens = [
        "KEY_PRESS_W",
        fdm1_mouse_axis_token("x", 8, screen_extent=854),
        fdm1_mouse_axis_token("y", -8, screen_extent=480),
        FDM1_ACTION_NOOP,
        FDM1_ACTION_PAD,
    ]
    converted = d2e_metric_tokens_from_fdm1_tokens(tokens, screen_width=854, screen_height=480)
    assert converted[0] == "KEY_PRESS_W"
    assert any(token.startswith("MOUSE_DX_") for token in converted)
    assert any(token.startswith("MOUSE_DY_") for token in converted)
    assert FDM1_ACTION_NOOP not in converted
    assert FDM1_ACTION_PAD not in converted
