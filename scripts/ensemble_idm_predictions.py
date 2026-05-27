#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.idm_prediction_ensemble import ensemble_idm_predictions


def _parse_sources(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source must be name=path: {value}")
        name, path = value.split("=", 1)
        out[name] = path
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble aligned IDM prediction JSONLs by token group.")
    parser.add_argument("--source", action="append", default=[], help="Named source prediction file, name=path. Repeatable.")
    parser.add_argument("--keyboard-source", required=True)
    parser.add_argument("--mouse-button-source", required=True)
    parser.add_argument("--mouse-move-source", required=True)
    parser.add_argument("--other-source")
    parser.add_argument("--model-name", default="ensemble_idm_predictions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    group_sources = {
        "keyboard": args.keyboard_source,
        "mouse_button": args.mouse_button_source,
        "mouse_move": args.mouse_move_source,
    }
    if args.other_source:
        group_sources["other"] = args.other_source
    payload = ensemble_idm_predictions(
        sources=_parse_sources(args.source),
        group_sources=group_sources,
        output_path=args.output,
        summary_out=args.summary,
        model_name=args.model_name,
        max_rows=args.max_rows,
    )
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "output": payload["output_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
