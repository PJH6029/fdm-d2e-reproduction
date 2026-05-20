#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.d2e_real import prepare_decoded_sample
from fdm_d2e.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one real D2E video/MCAP sample into v2 training contracts.")
    parser.add_argument("--config", default="configs/data/d2e_real_sample_decode.yaml")
    parser.add_argument(
        "--summary-copy",
        default="artifacts/sources/d2e_decoded_sample_summary.json",
        help="Small source-control-safe summary copy; raw decoded rows remain under ignored outputs/.",
    )
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to extract real D2E frame features")
    result = prepare_decoded_sample(load_config(args.config))
    summary = result["summary"]
    write_json(args.summary_copy, summary)
    print(
        "decoded real D2E sample: "
        f"pair={summary['pair_id']} events={summary['num_decoded_events']} "
        f"frames={summary['num_frame_features']} records={summary['num_window_records']} "
        f"train={summary['splits']['train']} heldout={summary['splits']['heldout']} "
        f"fingerprint={summary['dataset_fingerprint'][:12]}"
    )


if __name__ == "__main__":
    main()
