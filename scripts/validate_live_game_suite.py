#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.rollout.live_suite import load_json, run_live_suite_validation
from fdm_d2e.io_utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the G008 live open-source graphical game suite protocol/evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence", help="Live evidence JSON to validate. Omit for protocol-only readiness report.")
    parser.add_argument("--output", help="Override output path from config.")
    parser.add_argument("--root", default=".", help="Root directory for relative evidence artifact paths.")
    args = parser.parse_args()

    config = load_config(args.config)
    evidence = load_json(args.evidence) if args.evidence else None
    result = run_live_suite_validation(config, evidence, root=args.root)
    output_path = args.output or config.get("output_path")
    if output_path:
        write_json(output_path, result)
    gate = result["quality_gate"]
    print(
        "live game suite: "
        f"schema={result['schema']} status={gate.get('status')} "
        f"games={gate.get('games_with_passed_episode', gate.get('planned_games'))}/{gate.get('min_games')} "
        f"tasks={gate.get('passed_tasks', gate.get('planned_tasks'))}/{gate.get('min_tasks')} "
        f"findings={gate.get('findings_count', 0)}"
    )
    # Protocol-only mode is a readiness artifact, not a final pass/fail gate.
    if evidence is None:
        return 0 if gate.get("planned_games", 0) >= gate.get("min_games", 0) else 2
    return 0 if gate.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
