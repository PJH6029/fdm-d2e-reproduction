#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.gidm_exact_pipeline import run_gidm_exact_split_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/resume the released G-IDM exact-split inference and evaluation pipeline.")
    parser.add_argument("--config", default="configs/eval/g006_gidm_exact_split_pipeline.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage", choices=["all", "inference", "finalize"], default="all")
    parser.add_argument("--allow-partial", action="store_true", help="Finalize only existing prediction MCAPs for diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--recording-key", action="append", default=[])
    parser.add_argument("--cuda-devices")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = run_gidm_exact_split_pipeline(
        config,
        root=args.root,
        stage=args.stage,
        allow_partial=args.allow_partial,
        dry_run=args.dry_run,
        max_recordings=args.max_recordings,
        recording_keys=args.recording_key,
        cuda_devices=[item.strip() for item in args.cuda_devices.split(",") if item.strip()] if args.cuda_devices else None,
        workers=args.workers,
        log_wandb=not args.no_wandb,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" or args.allow_partial or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
