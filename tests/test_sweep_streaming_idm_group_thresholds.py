from __future__ import annotations

import argparse

import pytest

from scripts.sweep_streaming_idm_group_thresholds import _parse_grid, _threshold_slug, _thresholds_for_vocab


def test_thresholds_for_vocab_applies_keyboard_and_button_groups() -> None:
    thresholds = _thresholds_for_vocab(
        ["KEY_W", "MOUSE_LEFT_DOWN", "MOUSE_DX_1", "NOOP"],
        keyboard_threshold=0.05,
        button_threshold=0.2,
        default_threshold=0.35,
    )

    assert thresholds["KEY_W"] == 0.05
    assert thresholds["MOUSE_LEFT_DOWN"] == 0.2
    assert thresholds["MOUSE_DX_1"] == 0.35
    assert thresholds["NOOP"] == 0.35


def test_parse_grid_rejects_empty_or_out_of_range_values() -> None:
    assert _parse_grid("0.05,0.1, 0.35") == [0.05, 0.1, 0.35]

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_grid("")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_grid("0.1,1.5")


def test_threshold_slug_is_filesystem_stable() -> None:
    assert _threshold_slug(0.05) == "0p05"
    assert _threshold_slug(0.1) == "0p1"
    assert _threshold_slug(1.0) == "1"
