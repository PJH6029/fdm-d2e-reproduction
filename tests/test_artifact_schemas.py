import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.schema import SchemaError, validate_named


class SchemaTests(unittest.TestCase):
    def test_required_schema_files_exist(self):
        for name in [
            'data_manifest.schema.json', 'action_vocab.schema.json', 'sequence_pack.schema.json',
            'idm_pseudolabel.schema.json', 'fdm_checkpoint_metadata.schema.json', 'metrics.schema.json',
            'rollout_action.schema.json'
        ]:
            self.assertTrue((Path('schemas') / name).exists(), name)

    def test_idm_pseudolabel_schema_rejects_ground_truth_label_source(self):
        row = {
            'schema': 'idm_pseudolabel.v1', 'sequence_id': 's', 'timestamp_ns': 0,
            'predicted_tokens': ['KEY_PRESS_W'], 'label_source': 'ground_truth',
            'confidence': 1.0, 'model': 'bad', 'training_split_hash': 'x'
        }
        with self.assertRaises(SchemaError):
            validate_named(row, 'idm_pseudolabel.schema.json')

    def test_checkpoint_requires_idm_pseudolabel_source(self):
        row = {
            'schema': 'fdm_checkpoint_metadata.v1', 'model': 'm', 'label_source': 'ground_truth',
            'source_label_artifact': 'x', 'source_label_sha256': 'abc', 'predictions_path': 'p',
            'num_training_examples': 1, 'oracle_ground_truth_control': False
        }
        with self.assertRaises(SchemaError):
            validate_named(row, 'fdm_checkpoint_metadata.schema.json')


if __name__ == '__main__':
    unittest.main()
