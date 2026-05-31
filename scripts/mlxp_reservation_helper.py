#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://147.46.219.248:8000"
TOKEN_ENV_NAMES = ("RESERVATION_API_TOKEN", "MLXP_RESERVATION_API_TOKEN", "MLXP_API_TOKEN", "SNUPI_RESERVATION_API_TOKEN")


class MLXPError(RuntimeError):
    pass


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def token_from_env() -> str:
    for name in TOKEN_ENV_NAMES:
        token = os.environ.get(name)
        if token:
            return token
    raise MLXPError(f"missing API token; set one of: {', '.join(TOKEN_ENV_NAMES)}")


def request_json(base_url: str, path: str, *, token: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 60) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise MLXPError(f"HTTP {exc.code} {exc.reason}: {body}") from exc


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["node_id", "gpu_start", "gpu_count", "start_at", "end_at", "purpose"]
    for key in required:
        if payload.get(key) in (None, ""):
            errors.append(f"missing required field: {key}")
    if payload.get("managed_image_key") and payload.get("image_path"):
        errors.append("managed_image_key and image_path must not both be set")
    if payload.get("registry_profile_key") and not payload.get("image_path"):
        errors.append("registry_profile_key requires image_path")
    if (payload.get("command") or payload.get("args")) and not payload.get("image_path"):
        errors.append("command/args require explicit image_path")
    try:
        gpu_count = int(payload.get("gpu_count", 0))
        gpu_start = int(payload.get("gpu_start", 0))
        if gpu_count <= 0:
            errors.append("gpu_count must be positive")
        if gpu_start < 0:
            errors.append("gpu_start must be non-negative")
        if gpu_start + gpu_count > 8:
            errors.append("production reservations must stay within one 8-GPU node")
    except Exception:
        errors.append("gpu_start/gpu_count must be integers")
    return errors


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": "production",
        "node_id": payload.get("node_id"),
        "gpu_start": payload.get("gpu_start"),
        "gpu_count": payload.get("gpu_count"),
        "start_at": payload.get("start_at"),
        "end_at": payload.get("end_at"),
        "managed_image_key": payload.get("managed_image_key"),
        "image_path": payload.get("image_path"),
        "purpose": payload.get("purpose"),
    }


def summarize_board(board: dict[str, Any]) -> dict[str, Any]:
    nodes = [str(node.get("node_id")) for node in board.get("nodes", [])]
    free_by_node = {node_id: 0 for node_id in nodes}
    for row in board.get("rows", []):
        cells = list(row.get("cells", []))
        for node_index, node_id in enumerate(nodes):
            start = node_index * 8
            free_by_node[node_id] += sum(1 for cell in cells[start : start + 8] if cell is None)
    return {
        "schema": "mlxp_board_summary.v1",
        "project_id": board.get("project_id") or "production",
        "now_iso": board.get("now_iso"),
        "slot_minutes": board.get("slot_minutes"),
        "node_ids": nodes,
        "row_count": len(board.get("rows", [])),
        "free_cells_by_node": free_by_node,
        "default_image_key": board.get("default_image_key"),
        "managed_image_keys": sorted((board.get("managed_images") or {}).keys()) if isinstance(board.get("managed_images"), dict) else [],
        "registry_profile_keys": sorted((board.get("registry_profiles") or {}).keys()) if isinstance(board.get("registry_profiles"), dict) else [],
    }


def command_validate(args: argparse.Namespace) -> int:
    payload = read_json(args.payload)
    errors = validate_payload(payload)
    result = {"schema": "mlxp_reservation_payload_validation.v1", "status": "pass" if not errors else "fail", "errors": errors, "payload_summary": summarize_payload(payload)}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def command_board(args: argparse.Namespace) -> int:
    token = token_from_env()
    result = request_json(args.base_url, f"/api/projects/production/board?days={int(args.days)}", token=token)
    if args.output:
        write_json(args.output, result)
    summary = summarize_board(result)
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(result if args.print_full else summary, ensure_ascii=False, indent=2))
    return 0


def command_create(args: argparse.Namespace) -> int:
    payload = read_json(args.payload)
    errors = validate_payload(payload)
    if errors:
        raise MLXPError("payload validation failed: " + "; ".join(errors))
    if not args.i_confirm_live_production_reservation:
        raise MLXPError("refusing live production reservation without --i-confirm-live-production-reservation")
    token = token_from_env()
    result = request_json(args.base_url, "/api/projects/production/reservations", token=token, method="POST", payload=payload)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    token = token_from_env()
    result = request_json(args.base_url, f"/api/projects/production/reservations/{args.reservation_id}", token=token)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    if not args.i_confirm_cancel_reservation:
        raise MLXPError("refusing cancellation without --i-confirm-cancel-reservation")
    token = token_from_env()
    payload = {"actor_name": args.actor_name}
    result = request_json(args.base_url, f"/api/projects/production/reservations/{args.reservation_id}/cancel", token=token, method="POST", payload=payload)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe helper for MLXP production reservation payload validation/create/status/cancel.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-payload")
    validate.add_argument("--payload", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=command_validate)

    board = sub.add_parser("board")
    board.add_argument("--days", type=int, default=1)
    board.add_argument("--output")
    board.add_argument("--summary-output")
    board.add_argument("--print-full", action="store_true")
    board.set_defaults(func=command_board)

    create = sub.add_parser("create")
    create.add_argument("--payload", required=True)
    create.add_argument("--output")
    create.add_argument("--i-confirm-live-production-reservation", action="store_true")
    create.set_defaults(func=command_create)

    status = sub.add_parser("status")
    status.add_argument("--reservation-id", required=True)
    status.add_argument("--output")
    status.set_defaults(func=command_status)

    cancel = sub.add_parser("cancel")
    cancel.add_argument("--reservation-id", required=True)
    cancel.add_argument("--actor-name", default="jeonghunpark")
    cancel.add_argument("--output")
    cancel.add_argument("--i-confirm-cancel-reservation", action="store_true")
    cancel.set_defaults(func=command_cancel)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MLXPError as exc:
        print(f"mlxp helper error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
