import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.tokenization.actions import bin_delta, tokenize_event, tokenize_record


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


if __name__ == '__main__':
    unittest.main()
