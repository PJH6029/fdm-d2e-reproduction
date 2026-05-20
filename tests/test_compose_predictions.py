from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.compose_predictions import compose_prediction_tokens, _tokens_for_group


def test_tokens_for_group_filters_only_requested_control_surface():
    tokens = [
        "KEY_PRESS_87",
        "MOUSE_DX_P1",
        "MOUSE_DY_N1",
        "MOUSE_LEFT_DOWN",
        "MOUSE_RIGHT_UP",
    ]

    assert _tokens_for_group(tokens, "keyboard") == ["KEY_PRESS_87"]
    assert _tokens_for_group(tokens, "mouse_move") == ["MOUSE_DX_P1", "MOUSE_DY_N1"]
    assert _tokens_for_group(tokens, "mouse_button") == ["MOUSE_LEFT_DOWN", "MOUSE_RIGHT_UP"]


def test_compose_prediction_tokens_uses_predeclared_group_sources():
    sources_by_group = {
        "mouse_move": {
            "s0": {"predicted_tokens": ["MOUSE_DX_P2", "MOUSE_DY_Z0", "KEY_PRESS_65"]}
        },
        "keyboard": {
            "s0": {"predicted_tokens": ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"]}
        },
        "mouse_button": {
            "s0": {"predicted_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_N1"]}
        },
    }

    assert compose_prediction_tokens("s0", sources_by_group=sources_by_group) == [
        "MOUSE_DX_P2",
        "MOUSE_DY_Z0",
        "KEY_PRESS_87",
        "MOUSE_LEFT_DOWN",
    ]
