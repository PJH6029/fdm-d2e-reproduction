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

try:  # pragma: no cover - exercised on cluster images when present.
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
                yield _loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


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


def _prior_state_tokens(keys: set[str], buttons: set[str]) -> list[str]:
    tokens = [f"KEY_DOWN_{key}" for key in sorted(keys)]
    tokens.extend(f"MOUSE_{button}_DOWN" for button in sorted(buttons))
    return tokens or ["NOOP"]


def _transition_seen(tokens: list[str], *, prefixes: tuple[str, ...], suffixes: tuple[str, ...] = ()) -> bool:
    for token in tokens:
        if prefixes and token.startswith(prefixes):
            return True
        if suffixes and token.startswith("MOUSE_") and token.endswith(suffixes):
            return True
    return False


def _age_value(value: int | None) -> int | None:
    return None if value is None else max(0, int(value))


def _materialize_one(
    input_path: Path,
    output_path: Path,
    *,
    key_states: dict[str, set[str]],
    button_states: dict[str, set[str]],
    key_hold_bins: dict[str, dict[str, int]],
    button_hold_bins: dict[str, dict[str, int]],
    key_transition_age: dict[str, int | None],
    button_transition_age: dict[str, int | None],
    previous_event_tokens: dict[str, list[str]],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    prior_key_rows = 0
    prior_button_rows = 0
    event_key_rows = 0
    event_button_rows = 0
    with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out:
        for row in _iter_jsonl(input_path):
            recording_id = str(row.get("recording_id", ""))
            keys = key_states.setdefault(recording_id, set())
            buttons = button_states.setdefault(recording_id, set())
            key_durations = key_hold_bins.setdefault(recording_id, {})
            button_durations = button_hold_bins.setdefault(recording_id, {})
            prior_tokens = _prior_state_tokens(keys, buttons)
            prior_key_hold = {key: int(key_durations.get(key, 0)) for key in sorted(keys)}
            prior_button_hold = {button: int(button_durations.get(button, 0)) for button in sorted(buttons)}
            tokens = [str(token) for token in row.get("ground_truth_tokens", []) or []]
            _state_tokens, next_keys, next_buttons = state_tokens_from_event_tokens(
                tokens,
                pressed_keys=keys,
                pressed_buttons=buttons,
                mouse_emit_mode="decompose",
                mouse_max_tokens_per_axis=32,
            )
            key_event = _transition_seen(tokens, prefixes=("KEY_PRESS_", "KEY_RELEASE_", "KEY_DOWN_"))
            button_event = _transition_seen(tokens, prefixes=(), suffixes=("_DOWN", "_UP"))
            next_key_durations = {
                key: int(key_durations.get(key, 0)) + 1 if key in keys else 1
                for key in sorted(next_keys)
            }
            next_button_durations = {
                button: int(button_durations.get(button, 0)) + 1 if button in buttons else 1
                for button in sorted(next_buttons)
            }
            prev_tokens = previous_event_tokens.get(recording_id) or ["NOOP"]
            key_states[recording_id] = next_keys
            button_states[recording_id] = next_buttons
            key_hold_bins[recording_id] = next_key_durations
            button_hold_bins[recording_id] = next_button_durations
            out_row = dict(row)
            out_row["prior_action_tokens"] = prior_tokens
            out_row["prior_action_source"] = "d2e_held_state_before_current_event_bin"
            out_row["prior_key_hold_bins"] = prior_key_hold
            out_row["prior_button_hold_bins"] = prior_button_hold
            out_row["prior_since_key_transition_bins"] = _age_value(key_transition_age.get(recording_id))
            out_row["prior_since_button_transition_bins"] = _age_value(button_transition_age.get(recording_id))
            out_row["previous_event_tokens"] = prev_tokens
            out_row["state_context_schema"] = "d2e_event_target_with_prior_held_state_duration.v1"
            out.write(_dumps(out_row) + "\n")
            key_transition_age[recording_id] = 0 if key_event else (
                None if key_transition_age.get(recording_id) is None else int(key_transition_age[recording_id] or 0) + 1
            )
            button_transition_age[recording_id] = 0 if button_event else (
                None
                if button_transition_age.get(recording_id) is None
                else int(button_transition_age[recording_id] or 0) + 1
            )
            previous_event_tokens[recording_id] = tokens or ["NOOP"]
            rows += 1
            prior_key_rows += int(any(token.startswith("KEY_DOWN_") for token in prior_tokens))
            prior_button_rows += int(
                any(token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")) for token in prior_tokens)
            )
            event_key_rows += int(any(token.startswith("KEY_") for token in tokens))
            event_button_rows += int(any(token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")) for token in tokens))
    tmp_path.replace(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": rows,
        "prior_key_rows": prior_key_rows,
        "prior_button_rows": prior_button_rows,
        "event_key_rows": event_key_rows,
        "event_button_rows": event_button_rows,
        "sha256": _sha256_file(output_path),
    }


def _materialize_pair_task(payload: dict[str, Any]) -> dict[str, Any]:
    key_states: dict[str, set[str]] = {}
    button_states: dict[str, set[str]] = {}
    key_hold_bins: dict[str, dict[str, int]] = {}
    button_hold_bins: dict[str, dict[str, int]] = {}
    key_transition_age: dict[str, int | None] = {}
    button_transition_age: dict[str, int | None] = {}
    previous_event_tokens: dict[str, list[str]] = {}
    train = _materialize_one(
        Path(payload["train_input"]),
        Path(payload["train_output"]),
        key_states=key_states,
        button_states=button_states,
        key_hold_bins=key_hold_bins,
        button_hold_bins=button_hold_bins,
        key_transition_age=key_transition_age,
        button_transition_age=button_transition_age,
        previous_event_tokens=previous_event_tokens,
    )
    target = _materialize_one(
        Path(payload["target_input"]),
        Path(payload["target_output"]),
        key_states=key_states,
        button_states=button_states,
        key_hold_bins=key_hold_bins,
        button_hold_bins=button_hold_bins,
        key_transition_age=key_transition_age,
        button_transition_age=button_transition_age,
        previous_event_tokens=previous_event_tokens,
    )
    return {
        "pair_index": int(payload["pair_index"]),
        "train_output": train,
        "target_output": target,
        "recording_state_count": len(key_states),
    }


def materialize_event_state_context_corpus(
    *,
    train_inputs: list[Path],
    target_inputs: list[Path],
    input_root: Path,
    output_root: Path,
    summary_path: Path,
    workers: int = 1,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if len(train_inputs) != len(target_inputs):
        raise ValueError("event-state context materialization expects matching train/target shard counts")
    pairs = [
        {
            "pair_index": idx,
            "train_input": str(train_path),
            "target_input": str(target_path),
            "train_output": str(_output_path(train_path, input_root=input_root, output_root=output_root)),
            "target_output": str(_output_path(target_path, input_root=input_root, output_root=output_root)),
        }
        for idx, (train_path, target_path) in enumerate(zip(train_inputs, target_inputs))
    ]
    completed: list[dict[str, Any]] = []
    if progress_path:
        write_json(
            progress_path,
            {
                "schema": "d2e_event_state_context_materialization_progress.v1",
                "status": "running",
                "completed_pairs": 0,
                "total_pairs": len(pairs),
            },
        )
    if workers <= 1:
        for payload in pairs:
            completed.append(_materialize_pair_task(payload))
            if progress_path:
                write_json(
                    progress_path,
                    {
                        "schema": "d2e_event_state_context_materialization_progress.v1",
                        "status": "running",
                        "completed_pairs": len(completed),
                        "total_pairs": len(pairs),
                    },
                )
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pairs))) as pool:
            futures = {pool.submit(_materialize_pair_task, payload): payload for payload in pairs}
            for future in as_completed(futures):
                completed.append(future.result())
                if progress_path:
                    write_json(
                        progress_path,
                        {
                            "schema": "d2e_event_state_context_materialization_progress.v1",
                            "status": "running",
                            "completed_pairs": len(completed),
                            "total_pairs": len(pairs),
                        },
                    )
    completed = sorted(completed, key=lambda item: int(item["pair_index"]))
    train_rows = sum(int(row["train_output"]["rows"]) for row in completed)
    target_rows = sum(int(row["target_output"]["rows"]) for row in completed)
    payload = {
        "schema": "d2e_event_state_context_materialization.v1",
        "status": "pass",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "train_rows": train_rows,
        "target_rows": target_rows,
        "pair_count": len(completed),
        "recording_state_count": sum(int(row.get("recording_state_count") or 0) for row in completed),
        "train_outputs": [row["train_output"] for row in completed],
        "target_outputs": [row["target_output"] for row in completed],
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Adds causal prior held-state context while preserving original D2E event-token targets.",
    }
    write_json(summary_path, payload)
    if progress_path:
        write_json(
            progress_path,
            {
                "schema": "d2e_event_state_context_materialization_progress.v1",
                "status": "pass",
                "completed_pairs": len(completed),
                "total_pairs": len(pairs),
                "train_rows": train_rows,
                "target_rows": target_rows,
            },
        )
    return payload


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize D2E event targets with causal prior held-state context features.")
    parser.add_argument("--train-input", action="append", required=True)
    parser.add_argument("--target-input", action="append", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", default="artifacts/idm/g005_idm_event_state_context_materialization_summary.json")
    parser.add_argument("--progress-output", default="artifacts/idm/g005_idm_event_state_context_materialization_progress.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    payload = materialize_event_state_context_corpus(
        train_inputs=_expand(args.train_input),
        target_inputs=_expand(args.target_input),
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        summary_path=Path(args.summary),
        progress_path=Path(args.progress_output) if args.progress_output else None,
        workers=max(1, int(args.workers)),
    )
    print(json.dumps({"status": payload["status"], "train_rows": payload["train_rows"], "target_rows": payload["target_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
