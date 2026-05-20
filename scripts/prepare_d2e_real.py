#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.d2e_real import prepare_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real D2E manifest/split artifacts from Hugging Face file inventory.")
    parser.add_argument("--config", default="configs/data/d2e_real_sample.yaml")
    args = parser.parse_args()
    prepared = prepare_real_dataset(load_config(args.config))
    manifest = prepared["manifest"]
    print(
        "prepared real D2E manifest: "
        f"repo={manifest['hf_repo_id']} recordings={len(manifest['recordings'])} "
        f"train={manifest['splits'].get('train', 0)} heldout={manifest['splits'].get('heldout', 0)} "
        f"fingerprint={manifest['dataset_fingerprint'][:12]}"
    )


if __name__ == "__main__":
    main()
