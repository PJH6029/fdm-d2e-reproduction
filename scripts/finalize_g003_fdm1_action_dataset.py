#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.fdm1_g003_finalization import finalize_fdm1_g003_action_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize/reset-audit G003 FDM-1 action-slot dataset artifacts.")
    parser.add_argument("--config", default="configs/data/fdm1_g003_action_dataset_finalization.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    summary = finalize_fdm1_g003_action_dataset(load_config(args.config), root=args.root, force=args.force)
    print(
        "finalized FDM-1 G003 action dataset: "
        f"status={summary['status']} audit={summary['completion_audit_status']} errors={summary['completion_audit_error_count']}"
    )
    return 0 if summary["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
