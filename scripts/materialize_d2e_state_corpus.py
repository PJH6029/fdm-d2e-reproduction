#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.tokenization.actions import state_tokens_from_event_tokens

try:  # pragma: no cover - exercised on the cluster image when present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
    orjson = None


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _dumps(row: dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = _loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
            yield row


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_path(input_path: Path, *, input_root: Path, output_root: Path) -> Path:
    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        rel = Path(input_path.name)
    return output_root / rel


def _materialize_one(
    input_path: Path,
    output_path: Path,
    *,
    key_states: dict[str, set[str]],
    button_states: dict[str, set[str]],
    mouse_emit_mode: str,
    mouse_max_tokens_per_axis: int,
    include_raw_event_tokens: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    token_rows = 0
    key_state_rows = 0
    button_state_rows = 0
    mouse_token_count = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(input_path):
            recording_id = str(row.get("recording_id", ""))
            keys = key_states.setdefault(recording_id, set())
            buttons = button_states.setdefault(recording_id, set())
            state_tokens, next_keys, next_buttons = state_tokens_from_event_tokens(
                row.get("ground_truth_tokens", []) or [],
                pressed_keys=keys,
                pressed_buttons=buttons,
                mouse_emit_mode=mouse_emit_mode,
                mouse_max_tokens_per_axis=mouse_max_tokens_per_axis,
            )
            key_states[recording_id] = next_keys
            button_states[recording_id] = next_buttons
            out_row = dict(row)
            if include_raw_event_tokens:
                out_row["raw_event_tokens"] = list(row.get("ground_truth_tokens", []) or [])
            out_row["ground_truth_tokens"] = state_tokens
            out_row["action_state_schema"] = "d2e_50ms_held_state_tokens.v1"
            out.write(_dumps(out_row) + "\n")
            rows += 1
            token_rows += int(bool(state_tokens))
            key_state_rows += int(any(token.startswith("KEY_") for token in state_tokens))
            button_state_rows += int(any(token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")) for token in state_tokens))
            mouse_token_count += sum(1 for token in state_tokens if token.startswith(("MOUSE_DX_", "MOUSE_DY_")))
    tmp_path.replace(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": rows,
        "token_rows": token_rows,
        "key_state_rows": key_state_rows,
        "button_state_rows": button_state_rows,
        "mouse_token_count": mouse_token_count,
        "sha256": _sha256_file(output_path),
    }


def _materialize_pair_task(payload: dict[str, Any]) -> dict[str, Any]:
    key_states: dict[str, set[str]] = {}
    button_states: dict[str, set[str]] = {}
    train = _materialize_one(
        Path(payload["train_input"]),
        Path(payload["train_output"]),
        key_states=key_states,
        button_states=button_states,
        mouse_emit_mode=str(payload["mouse_emit_mode"]),
        mouse_max_tokens_per_axis=int(payload["mouse_max_tokens_per_axis"]),
        include_raw_event_tokens=bool(payload["include_raw_event_tokens"]),
    )
    target = _materialize_one(
        Path(payload["target_input"]),
        Path(payload["target_output"]),
        key_states=key_states,
        button_states=button_states,
        mouse_emit_mode=str(payload["mouse_emit_mode"]),
        mouse_max_tokens_per_axis=int(payload["mouse_max_tokens_per_axis"]),
        include_raw_event_tokens=bool(payload["include_raw_event_tokens"]),
    )
    return {
        "pair_index": int(payload["pair_index"]),
        "train_output": train,
        "target_output": target,
        "recording_state_count": len(key_states),
    }


def materialize_state_corpus(
    *,
    train_inputs: list[Path],
    target_inputs: list[Path],
    input_root: Path,
    output_root: Path,
    summary_path: Path,
    mouse_emit_mode: str = "decompose",
    mouse_max_tokens_per_axis: int = 32,
    include_raw_event_tokens: bool = False,
    workers: int = 1,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if len(train_inputs) != len(target_inputs):
        raise ValueError("state corpus materialization expects matching train/target shard counts")
    pairs = []
    for idx, (train_path, target_path) in enumerate(zip(train_inputs, target_inputs)):
        pairs.append(
            {
                "pair_index": idx,
                "train_input": str(train_path),
                "target_input": str(target_path),
                "train_output": str(_output_path(train_path, input_root=input_root, output_root=output_root)),
                "target_output": str(_output_path(target_path, input_root=input_root, output_root=output_root)),
                "mouse_emit_mode": mouse_emit_mode,
                "mouse_max_tokens_per_axis": int(mouse_max_tokens_per_axis),
                "include_raw_event_tokens": bool(include_raw_event_tokens),
            }
        )
    if progress_path:
        write_json(
            progress_path,
            {
                "schema": "d2e_state_corpus_materialization_progress.v1",
                "status": "running",
                "pairs_total": len(pairs),
                "pairs_completed": 0,
                "started_at_unix": started,
            },
        )
    completed: list[dict[str, Any]] = []
    if workers > 1 and len(pairs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pairs))) as pool:
            futures = [pool.submit(_materialize_pair_task, payload) for payload in pairs]
            for future in as_completed(futures):
                completed.append(future.result())
                if progress_path:
                    write_json(
                        progress_path,
                        {
                            "schema": "d2e_state_corpus_materialization_progress.v1",
                            "status": "running",
                            "pairs_total": len(pairs),
                            "pairs_completed": len(completed),
                            "last_pair_index": int(completed[-1]["pair_index"]),
                            "updated_at_unix": time.time(),
                        },
                    )
    else:
        for payload in pairs:
            completed.append(_materialize_pair_task(payload))
            if progress_path:
                write_json(
                    progress_path,
                    {
                        "schema": "d2e_state_corpus_materialization_progress.v1",
                        "status": "running",
                        "pairs_total": len(pairs),
                        "pairs_completed": len(completed),
                        "last_pair_index": int(completed[-1]["pair_index"]),
                        "updated_at_unix": time.time(),
                    },
                )
    completed = sorted(completed, key=lambda row: int(row["pair_index"]))
    train_outputs = [row["train_output"] for row in completed]
    target_outputs = [row["target_output"] for row in completed]
    total_train = sum(int(row["rows"]) for row in train_outputs)
    total_target = sum(int(row["rows"]) for row in target_outputs)
    payload = {
        "schema": "d2e_state_corpus_materialization.v1",
        "status": "pass",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mouse_emit_mode": mouse_emit_mode,
        "mouse_max_tokens_per_axis": int(mouse_max_tokens_per_axis),
        "include_raw_event_tokens": bool(include_raw_event_tokens),
        "workers": int(workers),
        "train_inputs": [str(path) for path in train_inputs],
        "target_inputs": [str(path) for path in target_inputs],
        "train_outputs": train_outputs,
        "target_outputs": target_outputs,
        "train_rows": total_train,
        "target_rows": total_target,
        "recording_state_count": sum(int(row.get("recording_state_count") or 0) for row in completed),
        "wall_clock_seconds": time.time() - started,
        "shard_state_assumption": "Each shard pair preserves complete per-recording order; train shard state is carried into the matching target shard.",
        "claim_boundary": "Derived D2E state-token corpus for paper-metric IDM exploration; raw D2E event-token corpus remains unchanged.",
    }
    write_json(summary_path, payload)
    if progress_path:
        progress_payload = {
            "schema": "d2e_state_corpus_materialization_progress.v1",
            "status": "pass",
            "pairs_total": len(pairs),
            "pairs_completed": len(completed),
            "updated_at_unix": time.time(),
            "summary_path": str(summary_path),
        }
        write_json(progress_path, progress_payload)
    return payload


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a D2E held-state token corpus from raw 50 ms event-token JSONL shards.")
    parser.add_argument("--train-input", action="append", required=True, help="Train JSONL path or glob. Repeatable.")
    parser.add_argument("--target-input", action="append", required=True, help="Target JSONL path or glob. Repeatable.")
    parser.add_argument("--input-root", default="outputs/data/d2e_full_corpus_shards_accel64")
    parser.add_argument("--output-root", default="outputs/data/d2e_state_corpus_shards_accel64")
    parser.add_argument("--summary", default="artifacts/idm/g005_idm_state_corpus_materialization_summary.json")
    parser.add_argument("--progress-output", default="artifacts/idm/g005_idm_state_corpus_materialization_progress.json")
    parser.add_argument("--mouse-emit-mode", default="decompose", choices=["single", "decompose"])
    parser.add_argument("--mouse-max-tokens-per-axis", type=int, default=32)
    parser.add_argument("--include-raw-event-tokens", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    payload = materialize_state_corpus(
        train_inputs=_expand(args.train_input),
        target_inputs=_expand(args.target_input),
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        summary_path=Path(args.summary),
        mouse_emit_mode=args.mouse_emit_mode,
        mouse_max_tokens_per_axis=args.mouse_max_tokens_per_axis,
        include_raw_event_tokens=bool(args.include_raw_event_tokens),
        workers=max(1, int(args.workers)),
        progress_path=Path(args.progress_output) if args.progress_output else None,
    )
    print(json.dumps({"status": payload["status"], "train_rows": payload["train_rows"], "target_rows": payload["target_rows"]}, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
