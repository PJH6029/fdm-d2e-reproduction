import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.io_utils import read_json, write_jsonl
from fdm_d2e.training.train_fdm import run_fdm_smoke


class FDMSmokeTests(unittest.TestCase):
    def test_fdm_requires_idm_generated_labels(self):
        with tempfile.TemporaryDirectory() as td:
            labels = Path(td) / 'labels.jsonl'
            write_jsonl(labels, [{'schema': 'idm_pseudolabel.v1', 'sequence_id': 's', 'timestamp_ns': 0, 'predicted_tokens': ['KEY_PRESS_W'], 'label_source': 'ground_truth', 'confidence': 1.0, 'model': 'bad', 'training_split_hash': 'x'}])
            with self.assertRaises(ValueError):
                run_fdm_smoke({'predictions_path': f'{td}/pred.jsonl', 'checkpoint_metadata_path': f'{td}/ckpt.json', 'train_log_path': f'{td}/log.json'}, labels)

    def test_fdm_records_pseudolabel_hash(self):
        with tempfile.TemporaryDirectory() as td:
            labels = Path(td) / 'labels.jsonl'
            write_jsonl(labels, [{'schema': 'idm_pseudolabel.v1', 'sequence_id': 's', 'timestamp_ns': 0, 'predicted_tokens': ['KEY_PRESS_W'], 'label_source': 'idm_generated', 'confidence': 1.0, 'model': 'idm', 'training_split_hash': 'x'}])
            ckpt = run_fdm_smoke({'predictions_path': f'{td}/pred.jsonl', 'checkpoint_metadata_path': f'{td}/ckpt.json', 'train_log_path': f'{td}/log.json'}, labels)
            self.assertEqual(ckpt['label_source'], 'idm_pseudolabel')
            self.assertTrue(read_json(f'{td}/log.json')['consumed_idm_pseudolabels'])


if __name__ == '__main__':
    unittest.main()
