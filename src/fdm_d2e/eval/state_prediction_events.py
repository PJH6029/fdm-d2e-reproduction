from __future__ import annotations

import glob
import json
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import write_json

try:  # pragma: no cover - exercised on cluster images when installed.
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


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        matches = sorted(glob.glob(text))
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(text)
        if path.exists():
            paths.append(path)
    return paths


def _recording_id(row: dict[str, Any]) -> str:
    value = row.get("recording_id")
    if isinstance(value, str) and value:
        return value
    sequence_id = row.get("sequence_id")
    if isinstance(sequence_id, str) and "#" in sequence_id:
        return sequence_id.rsplit("#", 1)[0]
    if isinstance(sequence_id, str) and sequence_id:
        return sequence_id
    return "__unknown_recording__"


def _tokens(row: dict[str, Any]) -> list[str]:
    value = row.get("predicted_tokens")
    if value is None:
        value = row.get("tokens")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _prior_tokens(row: dict[str, Any]) -> list[str]:
    value = row.get("prior_action_tokens")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _held_keys(tokens: Sequence[str]) -> set[str]:
    keys: set[str] = set()
    for token in tokens:
        if token.startswith("KEY_DOWN_"):
            keys.add(token.removeprefix("KEY_DOWN_"))
        elif token.startswith("KEY_PRESS_"):
            keys.add(token.removeprefix("KEY_PRESS_"))
    return keys


def _held_buttons(tokens: Sequence[str]) -> set[str]:
    buttons: set[str] = set()
    for token in tokens:
        if token.startswith("MOUSE_") and token.endswith("_DOWN") and not token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            buttons.add(token[len("MOUSE_") : -len("_DOWN")])
    return buttons


def _mouse_motion_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(token) for token in tokens if str(token).startswith(("MOUSE_DX_", "MOUSE_DY_"))]


class _DebouncedSetDiffer:
    def __init__(self, *, press_rows: int = 1, release_rows: int = 1) -> None:
        self.press_rows = max(1, int(press_rows))
        self.release_rows = max(1, int(release_rows))
        self.committed: set[str] = set()
        self.present_counts: dict[str, int] = {}
        self.absent_counts: dict[str, int] = {}

    def seed(self, committed: Iterable[str]) -> None:
        self.committed = {str(item) for item in committed if str(item)}
        self.present_counts = {}
        self.absent_counts = {}

    def update(self, observed: set[str]) -> tuple[list[str], list[str]]:
        presses: list[str] = []
        releases: list[str] = []
        for item in sorted(observed | set(self.present_counts) | self.committed):
            if item in self.committed:
                self.present_counts.pop(item, None)
                if item in observed:
                    self.absent_counts.pop(item, None)
                    continue
                count = self.absent_counts.get(item, 0) + 1
                if count >= self.release_rows:
                    self.committed.remove(item)
                    self.absent_counts.pop(item, None)
                    releases.append(item)
                else:
                    self.absent_counts[item] = count
                continue
            self.absent_counts.pop(item, None)
            if item not in observed:
                self.present_counts.pop(item, None)
                continue
            count = self.present_counts.get(item, 0) + 1
            if count >= self.press_rows:
                self.committed.add(item)
                self.present_counts.pop(item, None)
                presses.append(item)
            else:
                self.present_counts[item] = count
        return presses, releases


class _RecordingState:
    def __init__(
        self,
        *,
        key_press_rows: int,
        key_release_rows: int,
        button_press_rows: int,
        button_release_rows: int,
    ) -> None:
        self.keys = _DebouncedSetDiffer(press_rows=key_press_rows, release_rows=key_release_rows)
        self.buttons = _DebouncedSetDiffer(press_rows=button_press_rows, release_rows=button_release_rows)

    def seed_from_prior_tokens(self, tokens: Sequence[str]) -> None:
        self.keys.seed(_held_keys(tokens))
        self.buttons.seed(_held_buttons(tokens))


def _iter_prediction_seed_pairs(
    *,
    prediction_paths: Sequence[str | Path],
    seed_prior_paths: Sequence[str | Path] | None,
    max_rows: int | None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any] | None]]:
    prediction_iter = _iter_loaded_rows(_expand_paths(prediction_paths))
    seed_iter = _iter_loaded_rows(_expand_paths(seed_prior_paths or [])) if seed_prior_paths else None
    rows = 0
    if seed_iter is None:
        for prediction_row in prediction_iter:
            if max_rows is not None and rows >= max_rows:
                return
            yield prediction_row, None
            rows += 1
        return
    for prediction_row, seed_row in zip_longest(prediction_iter, seed_iter):
        if max_rows is not None and rows >= max_rows:
            return
        if prediction_row is None or seed_row is None:
            raise ValueError(
                "state prediction prior seeding requires prediction and target-prior streams "
                f"to have the same row count before max_rows; stopped at row {rows}"
            )
        yield prediction_row, seed_row
        rows += 1


def _iter_loaded_rows(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


def convert_state_prediction_tokens(
    tokens: Sequence[str],
    state: _RecordingState,
    *,
    include_mouse_motion: bool = True,
) -> list[str]:
    out: list[str] = []
    if include_mouse_motion:
        out.extend(_mouse_motion_tokens(tokens))
    key_presses, key_releases = state.keys.update(_held_keys(tokens))
    button_presses, button_releases = state.buttons.update(_held_buttons(tokens))
    out.extend(f"KEY_PRESS_{key}" for key in key_presses)
    out.extend(f"KEY_RELEASE_{key}" for key in key_releases)
    out.extend(f"MOUSE_{button}_DOWN" for button in button_presses)
    out.extend(f"MOUSE_{button}_UP" for button in button_releases)
    return out or ["NOOP"]


def convert_state_prediction_file(
    *,
    prediction_paths: Sequence[str | Path],
    output_path: str | Path,
    seed_prior_paths: Sequence[str | Path] | None = None,
    key_press_rows: int = 1,
    key_release_rows: int = 1,
    button_press_rows: int = 1,
    button_release_rows: int = 1,
    include_mouse_motion: bool = True,
    include_state_prediction_tokens: bool = False,
    max_rows: int | None = None,
    progress_output_path: str | Path | None = None,
    progress_rows: int = 1_000_000,
) -> dict[str, Any]:
    paths = _expand_paths(prediction_paths)
    seed_paths = _expand_paths(seed_prior_paths or [])
    findings: list[dict[str, Any]] = []
    if not paths:
        findings.append({"severity": "error", "code": "missing_prediction_paths", "patterns": [str(path) for path in prediction_paths]})
    if seed_prior_paths and not seed_paths:
        findings.append({"severity": "error", "code": "missing_seed_prior_paths", "patterns": [str(path) for path in seed_prior_paths]})
    started = time.time()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    states: dict[str, _RecordingState] = {}
    seeded_recordings: set[str] = set()
    rows = 0
    seed_rows = 0
    seed_sequence_mismatches = 0
    key_event_rows = 0
    button_event_rows = 0
    mouse_motion_rows = 0
    if paths:
        with tmp_output.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
            for row, seed_row in _iter_prediction_seed_pairs(
                prediction_paths=paths,
                seed_prior_paths=seed_paths or None,
                max_rows=max_rows,
            ):
                recording_id = _recording_id(row)
                state = states.setdefault(
                    recording_id,
                    _RecordingState(
                        key_press_rows=key_press_rows,
                        key_release_rows=key_release_rows,
                        button_press_rows=button_press_rows,
                        button_release_rows=button_release_rows,
                    ),
                )
                if seed_row is not None:
                    if str(seed_row.get("sequence_id")) != str(row.get("sequence_id")):
                        seed_sequence_mismatches += 1
                    if recording_id not in seeded_recordings:
                        state.seed_from_prior_tokens(_prior_tokens(seed_row))
                        seeded_recordings.add(recording_id)
                    seed_rows += 1
                converted = convert_state_prediction_tokens(
                    _tokens(row),
                    state,
                    include_mouse_motion=include_mouse_motion,
                )
                out_row = {
                    "schema": "state_prediction_eventified.v1",
                    "sequence_id": row.get("sequence_id"),
                    "recording_id": row.get("recording_id"),
                    "predicted_tokens": converted,
                    "conversion": {
                        "key_press_rows": int(key_press_rows),
                        "key_release_rows": int(key_release_rows),
                        "button_press_rows": int(button_press_rows),
                        "button_release_rows": int(button_release_rows),
                        "include_mouse_motion": bool(include_mouse_motion),
                        "include_state_prediction_tokens": bool(include_state_prediction_tokens),
                        "seed_prior_from_target": bool(seed_paths),
                    },
                }
                if include_state_prediction_tokens:
                    out_row["state_prediction_tokens"] = _tokens(row)
                handle.write(_dumps(out_row) + "\n")
                rows += 1
                key_event_rows += int(any(token.startswith("KEY_") for token in converted))
                button_event_rows += int(
                    any(token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")) for token in converted)
                )
                mouse_motion_rows += int(any(token.startswith(("MOUSE_DX_", "MOUSE_DY_")) for token in converted))
                if progress_output_path and progress_rows > 0 and rows % progress_rows == 0:
                    write_json(
                        progress_output_path,
                        {
                            "schema": "state_prediction_eventification_progress.v1",
                            "status": "running",
                            "rows": rows,
                            "recordings": len(states),
                            "seeded_recordings": len(seeded_recordings),
                            "output_path": str(output),
                        },
                    )
        tmp_output.replace(output)
    errors = [item for item in findings if item.get("severity") == "error"]
    if seed_sequence_mismatches:
        findings.append(
            {
                "severity": "warning",
                "code": "seed_prior_sequence_id_mismatches",
                "count": seed_sequence_mismatches,
            }
        )
    return {
        "schema": "state_prediction_eventification_summary.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "prediction_paths": [str(path) for path in paths],
        "seed_prior_paths": [str(path) for path in seed_paths],
        "output_path": str(output),
        "rows": rows,
        "recordings": len(states),
        "seed_rows": seed_rows,
        "seeded_recordings": len(seeded_recordings),
        "seed_sequence_mismatches": seed_sequence_mismatches,
        "key_event_rows": key_event_rows,
        "button_event_rows": button_event_rows,
        "mouse_motion_rows": mouse_motion_rows,
        "max_rows": max_rows,
        "conversion": {
            "key_press_rows": int(key_press_rows),
            "key_release_rows": int(key_release_rows),
            "button_press_rows": int(button_press_rows),
            "button_release_rows": int(button_release_rows),
            "include_mouse_motion": bool(include_mouse_motion),
            "include_state_prediction_tokens": bool(include_state_prediction_tokens),
            "seed_prior_from_target": bool(seed_paths),
        },
        "wall_clock_seconds": time.time() - started,
        "findings": findings,
        "claim_boundary": "Postprocesses held-state IDM predictions into D2E event-token predictions; it does not train or change the checkpoint.",
    }
