#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mlxp_reservation_helper import summarize_payload, validate_payload


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _node_ids(board: dict[str, Any]) -> list[str]:
    return [str(node["node_id"]) for node in board.get("nodes", [])]


def find_free_window(
    board: dict[str, Any],
    *,
    gpu_count: int,
    duration_hours: float,
    preferred_node_id: str | None = None,
    preferred_gpu_start: int | None = None,
) -> dict[str, Any]:
    rows = list(board.get("rows", []))
    nodes = _node_ids(board)
    if not rows or not nodes:
        raise ValueError("board must include rows and nodes")
    slot_minutes = int(board.get("slot_minutes") or round(float(board.get("slot_hours", 1)) * 60) or 60)
    slots_needed = max(1, int(math.ceil(float(duration_hours) * 60.0 / slot_minutes)))
    now_iso = board.get("now_iso")
    now = _parse_iso(now_iso) if now_iso else None
    node_order = nodes
    if preferred_node_id in nodes:
        node_order = [preferred_node_id] + [node for node in nodes if node != preferred_node_id]

    for row_i in range(0, max(0, len(rows) - slots_needed + 1)):
        if now is not None and _parse_iso(rows[row_i]["slot_end"]) <= now:
            continue
        for node_id in node_order:
            node_index = nodes.index(str(node_id))
            base = node_index * 8
            starts = list(range(0, 8 - int(gpu_count) + 1))
            if preferred_gpu_start is not None and preferred_gpu_start in starts:
                starts = [preferred_gpu_start] + [start for start in starts if start != preferred_gpu_start]
            for gpu_start in starts:
                ok = True
                for row in rows[row_i : row_i + slots_needed]:
                    cells = row.get("cells", [])[base + gpu_start : base + gpu_start + int(gpu_count)]
                    if len(cells) != int(gpu_count) or any(cell is not None for cell in cells):
                        ok = False
                        break
                if ok:
                    return {
                        "node_id": str(node_id),
                        "gpu_start": int(gpu_start),
                        "gpu_count": int(gpu_count),
                        "start_at": rows[row_i]["slot_start"],
                        "end_at": rows[row_i + slots_needed - 1]["slot_end"],
                        "slot_minutes": slot_minutes,
                        "slots": slots_needed,
                    }
    raise ValueError(f"no free {gpu_count}-GPU window for {duration_hours}h found")


def build_payload(board: dict[str, Any], *, gpu_count: int, duration_hours: float, purpose: str, preferred_node_id: str | None, preferred_gpu_start: int | None, actor_name: str, managed_image_key: str | None) -> dict[str, Any]:
    window = find_free_window(
        board,
        gpu_count=gpu_count,
        duration_hours=duration_hours,
        preferred_node_id=preferred_node_id,
        preferred_gpu_start=preferred_gpu_start,
    )
    image_key = managed_image_key or str(board.get("default_image_key") or "base")
    return {
        "node_id": window["node_id"],
        "gpu_start": window["gpu_start"],
        "gpu_count": window["gpu_count"],
        "gpu_indices": [],
        "start_at": window["start_at"],
        "end_at": window["end_at"],
        "purpose": purpose,
        "managed_image_key": image_key,
        "registry_profile_key": "",
        "image_path": "",
        "command": [],
        "args": [],
        "actor_name": actor_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a fresh MLXP production reservation payload from a board JSON snapshot without posting it.")
    parser.add_argument("--board-json", default=".omx/tmp/mlxp_board_latest.json")
    parser.add_argument("--output", default="artifacts/mlxp/g003_action_dataset_reservation_payload_draft.json")
    parser.add_argument("--validation-output", default="artifacts/mlxp/g003_action_dataset_reservation_payload_validation.json")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--duration-hours", type=float, default=12.0)
    parser.add_argument("--preferred-node-id", default="1")
    parser.add_argument("--preferred-gpu-start", type=int, default=2)
    parser.add_argument("--managed-image-key", default=None)
    parser.add_argument("--actor-name", default="jeonghunpark")
    parser.add_argument("--purpose", default="Continuous GUI - FDM reproduction: G003 D2E-480p CPU/IO action-slot materialization and audit; reserve 1xH200 only for managed production workspace/PVC access, cancel promptly if GPU remains idle after setup")
    args = parser.parse_args()

    board = read_json(args.board_json)
    payload = build_payload(
        board,
        gpu_count=args.gpu_count,
        duration_hours=args.duration_hours,
        purpose=args.purpose,
        preferred_node_id=args.preferred_node_id,
        preferred_gpu_start=args.preferred_gpu_start,
        actor_name=args.actor_name,
        managed_image_key=args.managed_image_key,
    )
    errors = validate_payload(payload)
    validation = {
        "schema": "mlxp_reservation_payload_validation.v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "payload_summary": summarize_payload(payload),
        "source_board": str(args.board_json),
        "board_now_iso": board.get("now_iso"),
    }
    write_json(args.output, payload)
    write_json(args.validation_output, validation)
    print(json.dumps({"payload": payload, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
