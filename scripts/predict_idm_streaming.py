#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.streaming_idm import predict_streaming_idm_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained streaming IDM checkpoint over a record JSONL without retraining.")
    parser.add_argument("--config", default="configs/model/idm_streaming_d2e_full_compact_predict_fdm_train.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    summary = predict_streaming_idm_checkpoint(config)
    print(
        "streaming idm prediction complete: "
        f"records={summary['records']} pseudo_labels={summary['pseudo_label_path']}"
    )


if __name__ == "__main__":
    main()
