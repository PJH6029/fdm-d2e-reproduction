#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.training.streaming_fdm import recover_streaming_fdm_outputs_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover streaming FDM wrapper metadata/summary from an existing torch_model checkpoint "
            "without rematerializing/retraining unless explicitly requested."
        )
    )
    parser.add_argument("--config", default="configs/model/fdm_streaming_d2e_full_compact.yaml")
    parser.add_argument("--torch-checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--materialize-split-if-missing", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("config_path", args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.torch_checkpoint_path:
        config["torch_checkpoint_path"] = args.torch_checkpoint_path
    config["materialize_split_if_missing"] = bool(args.materialize_split_if_missing)
    config["resume_predictions"] = not args.no_resume_predictions

    summary = recover_streaming_fdm_outputs_from_checkpoint(config)
    print(
        "streaming fdm checkpoint recovery complete: "
        f"status={summary['status']} target_examples={summary['target_examples']} "
        f"metadata={summary['checkpoint_metadata_path']}"
    )


if __name__ == "__main__":
    main()
