import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.data.d2e_reader import prepare_smoke_dataset


class D2EReaderTests(unittest.TestCase):
    def test_prepare_smoke_dataset_orders_timestamps_and_notes_categories(self):
        with tempfile.TemporaryDirectory() as td:
            out = prepare_smoke_dataset({'output_dir': td, 'recording_id': 'r', 'num_records': 8, 'train_count': 5})
            timestamps = [r['timestamp_ns'] for r in out['records']]
            self.assertEqual(timestamps, sorted(timestamps))
            manifest = out['manifest']
            self.assertIn('keyboard', manifest['event_categories'])
            self.assertTrue(manifest['source_contract']['paired_video_mcap'])
            self.assertEqual(manifest['source_contract']['timestamp_unit'], 'nanoseconds')


if __name__ == '__main__':
    unittest.main()
