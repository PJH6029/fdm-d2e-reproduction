#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdm_d2e.io_utils import write_json
from finalize_g006_evaluation import finalize as finalize_g006
from plan_g006_readiness import build_readiness_plan


DEFAULT_OUTPUT = "artifacts/eval/g006_postrun_watcher_summary.json"


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _base_payload(args: argparse.Namespace, root: Path, *, started_at: float) -> dict[str, Any]:
    return {
        "schema": "g006_postrun_watcher.v1",
        "root": str(root),
        "started_at_unix": started_at,
        "output": args.output,
        "poll_seconds": float(args.poll_seconds),
        "max_wait_seconds": float(args.max_wait_seconds),
        "claim_boundary": "Watches G006 artifact/prerequisite readiness and runs local finalization only when ready; it never checkpoints G006 or mutates OMX/Codex goal state.",
    }


def _planner_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        root=args.root,
        build_config=args.build_config,
        build_summary_out=args.build_summary_out,
        readiness_config=args.readiness_config,
        readiness_output=args.readiness_output,
        g006_completion_config=args.g006_completion_config,
        g006_audit_output=args.g006_audit_output,
        require_existing_final_outputs=args.require_existing_final_outputs,
        allow_precheckpoint=args.allow_precheckpoint,
        output=args.readiness_plan_output,
        allow_fail=True,
    )


def _finalizer_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        root=args.root,
        summary_out=args.g006_finalization_summary,
        allow_fail=True,
        skip_build=args.skip_build,
        build_config=args.build_config,
        build_summary_out=args.build_summary_out,
        readiness_config=args.readiness_config,
        readiness_output=args.readiness_output,
        g006_completion_config=args.g006_completion_config,
        g006_audit_output=args.g006_audit_output,
    )


def _write_summary(root: Path, output: str | Path, payload: dict[str, Any]) -> None:
    write_json(_path(root, output), payload)


def watch(
    args: argparse.Namespace,
    *,
    finalize_func: Callable[[argparse.Namespace], dict[str, Any]] = finalize_g006,
    sleep_func: Callable[[float], None] = time.sleep,
    time_func: Callable[[], float] = time.time,
) -> dict[str, Any]:
    root = Path(args.root).resolve()
    started = time_func()
    base = _base_payload(args, root, started_at=started)
    while True:
        now = time_func()
        elapsed = max(0.0, now - started)
        plan = build_readiness_plan(_planner_args(args))
        write_json(_path(root, args.readiness_plan_output), plan)
        if plan.get("status") != "ready":
            payload = {
                **base,
                "status": "waiting_for_g006_inputs",
                "elapsed_seconds": elapsed,
                "readiness_plan_output": args.readiness_plan_output,
                "readiness_status": plan.get("status"),
                "findings": plan.get("findings", []),
                "warnings": plan.get("warnings", []),
            }
            _write_summary(root, args.output, payload)
            if args.once:
                return payload
            if float(args.max_wait_seconds) >= 0 and elapsed >= float(args.max_wait_seconds):
                payload["status"] = "timeout_waiting_for_g006_inputs"
                payload["findings"] = [*payload["findings"], {"severity": "error", "code": "timeout_waiting_for_g006_inputs", "elapsed_seconds": elapsed}]
                _write_summary(root, args.output, payload)
                return payload
            sleep_func(float(args.poll_seconds))
            continue

        finalization = finalize_func(_finalizer_args(args))
        status = "finalized_pass" if finalization.get("status") == "pass" else "finalized_fail"
        payload = {
            **base,
            "status": status,
            "elapsed_seconds": elapsed,
            "readiness_plan_output": args.readiness_plan_output,
            "readiness_status": plan.get("status"),
            "g006_finalization_summary": args.g006_finalization_summary,
            "g006_finalization_status": finalization.get("status"),
            "readiness_audit_status": finalization.get("readiness_status"),
            "g006_audit_status": finalization.get("g006_audit_status"),
            "g006_audit_error_count": finalization.get("g006_audit_error_count"),
            "findings": [] if status == "finalized_pass" else [{"severity": "error", "code": "g006_finalization_not_pass", "status": finalization.get("status")}],
            "warnings": plan.get("warnings", []),
        }
        _write_summary(root, args.output, payload)
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch G006 prerequisites/artifacts and run non-mutating finalization once ready.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-wait-seconds", type=float, default=-1.0)
    parser.add_argument("--build-config", default="configs/eval/g006_final_artifacts.yaml")
    parser.add_argument("--build-summary-out", default="artifacts/eval/g006_final_artifact_build_summary.json")
    parser.add_argument("--readiness-config", default="configs/eval/g006_evaluation_readiness.yaml")
    parser.add_argument("--readiness-output", default="artifacts/eval/g006_evaluation_readiness_audit.json")
    parser.add_argument("--g006-completion-config", default="configs/eval/g006_completion.yaml")
    parser.add_argument("--g006-audit-output", default="artifacts/eval/g006_completion_audit.json")
    parser.add_argument("--g006-finalization-summary", default="artifacts/eval/g006_finalization_summary.json")
    parser.add_argument("--readiness-plan-output", default="artifacts/eval/g006_readiness_plan.json")
    parser.add_argument("--require-existing-final-outputs", action="store_true")
    parser.add_argument("--allow-precheckpoint", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    payload = watch(args)
    print(f"g006 postrun watcher: status={payload['status']} output={args.output}")
    terminal_ok = payload["status"] in {"waiting_for_g006_inputs", "finalized_pass"}
    return 0 if terminal_ok or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
