#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path


def detect_torch() -> dict[str, object]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"torch_imported": False, "torch_error": repr(exc), "cuda_available": False, "cuda_device_count": 0}
    return {
        "torch_imported": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the cluster launcher can see the requested GPU shape.")
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--report", default="outputs/cluster/gpu_smoke.json")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit a local CPU-only dry run while preserving GPU command shape.")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch_info = detect_torch()
    cuda_count = int(torch_info.get("cuda_device_count", 0) or 0)
    pass_gpu = bool(torch_info.get("cuda_available")) and cuda_count >= args.expected_gpus
    pass_cpu_dry = args.allow_cpu and world_size == 1
    status = "passed" if pass_gpu else ("cpu_dry_run" if pass_cpu_dry else "failed")
    payload = {
        "schema": "cluster_gpu_smoke.v1",
        "status": status,
        "expected_gpus": args.expected_gpus,
        "world_size": world_size,
        "rank": rank,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **torch_info,
    }
    report = Path(args.report)
    if rank != 0:
        report = report.with_suffix(report.suffix + f".rank{rank}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if status in {"passed", "cpu_dry_run"} else 2


if __name__ == "__main__":
    sys.exit(main())
