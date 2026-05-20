import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.launch import DEFAULT_MLXP_REPO_PATH, build_gpu_smoke_matrix, build_torchrun_command, launcher_env, write_gpu_smoke_matrix


class ClusterLaunchTests(unittest.TestCase):
    def test_launcher_env_pins_visible_devices(self):
        env = launcher_env(4)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["UV_LINK_MODE"], "copy")

    def test_single_gpu_uses_uv_python(self):
        cmd = build_torchrun_command(gpu_count=1, script="scripts/cluster_gpu_smoke.py")
        self.assertEqual(cmd.repo_path, DEFAULT_MLXP_REPO_PATH)
        self.assertEqual(cmd.command[:3], ["uv", "run", "python"])
        self.assertIn("--expected-gpus", cmd.command)
        self.assertIn("1", cmd.command)

    def test_multi_gpu_uses_torchrun(self):
        cmd = build_torchrun_command(gpu_count=4, script="scripts/cluster_gpu_smoke.py")
        self.assertEqual(cmd.command[:3], ["uv", "run", "torchrun"])
        self.assertIn("--nproc-per-node", cmd.command)
        self.assertIn("4", cmd.command)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3", cmd.shell())

    def test_matrix_writes_1_2_4_pvc_commands(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "matrix.json"
            payload = write_gpu_smoke_matrix(path)
            self.assertTrue(path.exists())
            self.assertEqual([row.gpu_count for row in build_gpu_smoke_matrix()], [1, 2, 4])
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["schema"], "cluster_gpu_smoke_matrix.v1")
            self.assertEqual([row["gpu_count"] for row in payload["commands"]], [1, 2, 4])
            self.assertTrue(all(row["repo_path"] == DEFAULT_MLXP_REPO_PATH for row in payload["commands"]))


if __name__ == "__main__":
    unittest.main()
