import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.io_utils import read_json, write_jsonl
from fdm_d2e.training.train_fdm import _records_with_pseudolabel_tokens, _split_labels_by_recording_tail, run_fdm_smoke


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

    def test_fdm_real_split_uses_tail_targets_per_recording(self):
        labels = [
            {'sequence_id': f'r#{idx}', 'timestamp_ns': idx, 'predicted_tokens': [f'KEY_PRESS_{idx}'], 'label_source': 'idm_generated'}
            for idx in range(4)
        ] + [
            {'sequence_id': f's#{idx}', 'timestamp_ns': idx, 'predicted_tokens': [f'KEY_PRESS_{idx}'], 'label_source': 'idm_generated'}
            for idx in range(4)
        ]

        train, target = _split_labels_by_recording_tail(labels, train_fraction=0.75)

        self.assertEqual([row['sequence_id'] for row in train], ['r#0', 'r#1', 'r#2', 's#0', 's#1', 's#2'])
        self.assertEqual([row['sequence_id'] for row in target], ['r#3', 's#3'])

    def test_fdm_real_records_replace_training_tokens_with_pseudolabels(self):
        records = {
            'r#0': {'sequence_id': 'r#0', 'timestamp_ns': 0, 'ground_truth_tokens': ['KEY_PRESS_1']},
        }
        labels = [{'sequence_id': 'r#0', 'timestamp_ns': 0, 'predicted_tokens': ['MOUSE_DX_P1'], 'label_source': 'idm_generated'}]

        rows = _records_with_pseudolabel_tokens(records, labels)

        self.assertEqual(rows[0]['ground_truth_tokens'], ['MOUSE_DX_P1'])
        self.assertEqual(rows[0]['label_source'], 'idm_pseudolabel_for_fdm')


if __name__ == '__main__':
    unittest.main()
