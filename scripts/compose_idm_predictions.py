#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.compose_predictions import compose_idm_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose specialist IDM prediction streams into one evaluable heldout artifact.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = compose_idm_predictions(load_config(args.config))
    print(
        "composed IDM predictions: "
        f"model={summary['model_name']} target={summary['target_records']} "
        f"metrics={summary['metrics_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
