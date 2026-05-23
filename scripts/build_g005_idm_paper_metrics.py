#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics


def _list_from_config(config: dict, key: str, fallback_key: str) -> list[str]:
    value = config.get(key)
    if value is None:
        value = config.get(fallback_key)
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is not None:
        return [str(value)]
    raise ValueError(f"missing required config key: {key} or {fallback_key}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-compatible IDM metrics from token JSONL predictions/targets for G005.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics_cfg = dict(config.get("paper_metrics", config))
    root = Path(args.root)
    output = args.output or metrics_cfg.get("output_path") or metrics_cfg.get("metrics_path")
    if not output:
        raise ValueError("paper metrics config requires output_path or metrics_path")
    prediction_paths = _list_from_config(metrics_cfg, "prediction_paths", "predictions_path")
    target_paths = _list_from_config(metrics_cfg, "target_paths", "target_path")
    payload = write_paper_idm_metrics(
        prediction_paths=[str(root / path) if not Path(path).is_absolute() else path for path in prediction_paths],
        target_paths=[str(root / path) if not Path(path).is_absolute() else path for path in target_paths],
        output_path=root / output if not Path(output).is_absolute() else output,
        split_tags=[str(tag) for tag in metrics_cfg.get("split_tags", ["temporal", "heldout_recording", "heldout_game"])],
        model_name=str(metrics_cfg.get("model_name", config.get("model_name", "model"))),
        max_rows=metrics_cfg.get("max_rows"),
        progress_output_path=(
            root / str(metrics_cfg["progress_output_path"])
            if metrics_cfg.get("progress_output_path") and not Path(str(metrics_cfg["progress_output_path"])).is_absolute()
            else metrics_cfg.get("progress_output_path")
        ),
        progress_rows=int(metrics_cfg.get("progress_rows", 1_000_000)),
        empty_bins_as_correct=bool(metrics_cfg.get("empty_bins_as_correct", False)),
    )
    print(f"g005 paper metrics: status={payload['status']} rows={payload['alignment']['rows_seen']} output={output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
