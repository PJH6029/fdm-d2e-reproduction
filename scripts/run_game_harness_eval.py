#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.rollout.game_harness import run_game_harness_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic game/game-adjacent harness tasks against trained FDM predictions.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_game_harness_eval(load_config(args.config))
    gate = result["quality_gate"]
    print(
        "game harness eval: "
        f"status={gate['status']} tasks={gate['tasks_passed']}/{gate['tasks_minimum']} "
        f"envs={gate['environments_passed']}/{gate['environments_minimum']} "
        f"candidate_probes={gate['install_control_pass_count']}/{gate['install_control_minimum']}"
    )
    return 0 if gate["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
