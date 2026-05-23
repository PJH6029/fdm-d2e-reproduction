#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import write_json
from fdm_d2e.training.g005_aux_prior import run_g005_aux_prior_candidate


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate the G005 D2E+aux action-prior FDM candidate.")
    parser.add_argument("--config", default="configs/model/g005_aux_prior_candidate.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/aux/g005_d2e_aux_train_run.json")
    parser.add_argument("--max-aux-examples-per-source", type=int)
    parser.add_argument("--max-d2e-eval-rows", type=int)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config: dict[str, Any] = load_config(_path(root, args.config))
    if args.max_aux_examples_per_source is not None:
        config["max_aux_examples_per_source"] = args.max_aux_examples_per_source
    if args.max_d2e_eval_rows is not None:
        config["max_d2e_eval_rows"] = args.max_d2e_eval_rows
    config["run_summary"] = args.output
    try:
        payload = run_g005_aux_prior_candidate(config, root=root)
    except Exception as exc:
        failure = {
            "schema": "g005_d2e_aux_train_run.v1",
            "status": "fail",
            "exit_code": 1,
            "error": repr(exc),
            "config": args.config,
            "claim_boundary": "Failed G005 candidate run; do not use for G005 completion.",
        }
        write_json(_path(root, args.output), failure)
        print(f"g005 aux prior candidate: status=fail output={args.output} error={exc}", file=sys.stderr)
        if args.allow_fail:
            return 0
        raise
    print(f"g005 aux prior candidate: status={payload['status']} output={args.output}")
    return 0 if payload.get("status") == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
