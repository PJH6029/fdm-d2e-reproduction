import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.data.d2e_reader import prepare_smoke_dataset
from fdm_d2e.tokenization.actions import add_tokens
from fdm_d2e.training.pseudolabel import generate_pseudolabels


class IDMSmokeTests(unittest.TestCase):
    def test_generates_schema_valid_aligned_pseudolabels(self):
        with tempfile.TemporaryDirectory() as td:
            out = prepare_smoke_dataset({'output_dir': td, 'recording_id': 'r', 'num_records': 8, 'train_count': 4})
            records = add_tokens(out['records'])
            train = [r for r in records if r['split'] == 'train']
            labels = generate_pseudolabels(train, records)
            self.assertEqual(len(labels), len(records))
            self.assertTrue(all(l['label_source'] == 'idm_generated' for l in labels))
            self.assertEqual([l['timestamp_ns'] for l in labels], [r['timestamp_ns'] for r in records])


if __name__ == '__main__':
    unittest.main()
