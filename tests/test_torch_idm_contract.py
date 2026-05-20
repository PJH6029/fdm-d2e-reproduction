import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.training.torch_idm import torch_available


class TorchIDMContractTests(unittest.TestCase):
    def test_torch_availability_probe_is_boolean(self):
        self.assertIsInstance(torch_available(), bool)


if __name__ == "__main__":
    unittest.main()
