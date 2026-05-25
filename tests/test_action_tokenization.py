import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.tokenization.actions import bin_delta, state_tokens_from_event_tokens, tokenize_event, tokenize_record, tokens_from_delta


class TokenizationTests(unittest.TestCase):
    def test_keyboard_mouse_scroll_tokens(self):
        self.assertEqual(tokenize_event({'type': 'keyboard', 'event_type': 'press', 'key': 'w'}), ['KEY_PRESS_W'])
        self.assertEqual(tokenize_event({'type': 'mouse_button', 'event_type': 'release', 'button': 'left'}), ['MOUSE_LEFT_UP'])
        self.assertEqual(tokenize_event({'type': 'scroll', 'dy': -1}), ['SCROLL_DOWN'])
        self.assertEqual(bin_delta(0), 'Z0')
        self.assertEqual(bin_delta(-9), 'N4')

    def test_record_tokenization_includes_mouse_move_bins(self):
        tokens = tokenize_record({'events': [{'type': 'mouse_move', 'dx': 6, 'dy': -2}]})
        self.assertIn('MOUSE_DX_P3', tokens)
        self.assertIn('MOUSE_DY_N2', tokens)

    def test_state_tokens_track_held_controls_and_sum_mouse(self):
        first, keys, buttons = state_tokens_from_event_tokens(
            ['KEY_PRESS_w', 'MOUSE_LEFT_DOWN', 'MOUSE_DX_P3', 'MOUSE_DX_P2', 'MOUSE_DY_N1'],
            mouse_emit_mode='single',
        )
        self.assertIn('KEY_DOWN_W', first)
        self.assertIn('MOUSE_LEFT_DOWN', first)
        self.assertIn('MOUSE_DX_P3', first)
        self.assertIn('MOUSE_DY_N1', first)
        second, _keys, _buttons = state_tokens_from_event_tokens(
            ['KEY_RELEASE_w', 'MOUSE_LEFT_UP', 'MOUSE_DX_Z0', 'MOUSE_DY_Z0'],
            pressed_keys=keys,
            pressed_buttons=buttons,
            mouse_emit_mode='single',
        )
        self.assertEqual(second, ['MOUSE_DX_Z0', 'MOUSE_DY_Z0'])

    def test_tokens_from_delta_decomposes_large_motion(self):
        tokens = tokens_from_delta(31, -13, emit_mode='decompose', max_tokens_per_axis=4)
        self.assertEqual(tokens[:2], ['MOUSE_DX_P5', 'MOUSE_DX_P3'])
        self.assertEqual(tokens[-2:], ['MOUSE_DY_N4', 'MOUSE_DY_N1'])


if __name__ == '__main__':
    unittest.main()
