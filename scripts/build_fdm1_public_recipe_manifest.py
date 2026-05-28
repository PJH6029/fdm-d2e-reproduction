#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.reporting.fdm1_recipe_alignment import write_fdm1_public_recipe_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the public FDM-1 recipe constraint manifest used by renewed ultragoal guardrails.")
    parser.add_argument("--output", default="artifacts/reproducibility/fdm1_public_recipe_manifest.json")
    args = parser.parse_args()
    payload = write_fdm1_public_recipe_manifest(args.output)
    print(f"fdm1 public recipe manifest: status={payload['status']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
