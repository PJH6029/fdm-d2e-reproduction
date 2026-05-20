import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class DocumentationGuardTests(unittest.TestCase):
    def test_non_parity_and_non_commercial_notices_exist(self):
        text = "\n".join(Path(p).read_text().lower() for p in ["README.md", "docs/reproduction_scope.md", "docs/dataset_license.md", "docs/d2e_source_contract.md"])
        self.assertIn("not an fdm-1 parity claim", text)
        self.assertIn("non-commercial", text)
        self.assertNotIn("matches fdm-1", text)
        self.assertNotIn("equivalent to fdm-1", text)


if __name__ == '__main__':
    unittest.main()
