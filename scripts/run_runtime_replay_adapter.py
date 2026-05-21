#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.runtime.sdk import run_replay_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run safe deterministic runtime adapter replay over FDM predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--focus-title", default=None, help="Override active window title for focus-guard validation.")
    args = parser.parse_args()
    focus_provider = (lambda: str(args.focus_title)) if args.focus_title is not None else None
    result = run_replay_adapter(load_config(args.config), focus_title_provider=focus_provider)
    print(
        "runtime replay adapter: "
        f"actions={result['num_actions']} applied={result['applied_actions']} "
        f"blocked={result['blocked_actions']} p95_ms={result['latency']['p95_ms']}"
    )
    return 0 if result["num_actions"] and result["blocked_actions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
