#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.reporting.claim_audit import audit_claim_boundaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit report wording against active ultragoal claim boundaries.")
    parser.add_argument("--goals-path", default=".omx/ultragoal/goals.json")
    parser.add_argument("--output", default="artifacts/reproducibility/claim_boundary_audit.json")
    args = parser.parse_args()
    payload = audit_claim_boundaries(goals_path=args.goals_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"claim boundary audit: status={payload['status']} findings={len(payload['findings'])} output={out}")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
