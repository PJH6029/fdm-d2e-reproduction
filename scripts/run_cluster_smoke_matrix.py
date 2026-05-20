#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.cluster.launch import DEFAULT_MLXP_REPO_PATH, build_gpu_smoke_matrix, execute_launch_command, write_gpu_smoke_matrix
from fdm_d2e.io_utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or execute the 1/2/4 GPU MLXP launcher smoke matrix.")
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repo-path", default=DEFAULT_MLXP_REPO_PATH)
    parser.add_argument("--report", default="artifacts/mlxp/cluster_launcher_matrix.local.json")
    parser.add_argument("--execute", action="store_true", help="Run the commands instead of writing the dry-run launcher contract.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU-only execution for local validation of the 1-GPU script path.")
    args = parser.parse_args()

    if not args.execute:
        payload = write_gpu_smoke_matrix(args.report, args.gpu_counts, repo_path=args.repo_path)
        print(f"wrote dry-run cluster launcher matrix: {args.report} commands={len(payload['commands'])}")
        return 0

    results = []
    for command in build_gpu_smoke_matrix(args.gpu_counts, repo_path=args.repo_path):
        results.append(execute_launch_command(command, allow_cpu=args.allow_cpu))
    payload = {"schema": "cluster_gpu_smoke_execution.v1", "repo_path": args.repo_path, "results": results}
    write_json(args.report, payload)
    failed = [row for row in results if row["returncode"] != 0]
    print(f"wrote cluster launcher execution report: {args.report} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
