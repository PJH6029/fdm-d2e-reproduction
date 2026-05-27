from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, TextIO

from fdm_d2e.io_utils import write_json


def _loads(line: str) -> dict[str, Any]:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("prediction row must be a JSON object")
    return payload


def _token_group(token: str) -> str:
    if token.startswith("KEY_"):
        return "keyboard"
    if token.startswith("MOUSE_DX_") or token.startswith("MOUSE_DY_"):
        return "mouse_move"
    if token.startswith("MOUSE_"):
        return "mouse_button"
    return "other"


def _tokens_for_group(row: dict[str, Any], group: str) -> list[str]:
    tokens = [str(token) for token in row.get("predicted_tokens", []) or []]
    if group == "all":
        return tokens
    return [token for token in tokens if _token_group(token) == group]


def _dedupe_preserve_order(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _open_sources(sources: dict[str, str | Path]) -> dict[str, TextIO]:
    return {name: Path(path).open("r", encoding="utf-8", buffering=1024 * 1024) for name, path in sources.items()}


def ensemble_idm_predictions(
    *,
    sources: dict[str, str | Path],
    group_sources: dict[str, str],
    output_path: str | Path,
    summary_out: str | Path,
    model_name: str = "ensemble_idm_predictions",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Combine aligned IDM prediction JSONLs by token group.

    This is a deterministic post-hoc ensemble over already trained/evaluated IDM
    candidates.  It never reads target labels and therefore can be used as a
    leakage-safe scaling probe when source prediction files are aligned by
    sequence_id.
    """

    if not sources:
        raise ValueError("at least one source prediction file is required")
    allowed_groups = {"keyboard", "mouse_button", "mouse_move", "other"}
    unknown_groups = set(group_sources) - allowed_groups
    if unknown_groups:
        raise ValueError(f"unsupported token groups: {sorted(unknown_groups)}")
    missing_sources = set(group_sources.values()) - set(sources)
    if missing_sources:
        raise ValueError(f"group_sources reference missing sources: {sorted(missing_sources)}")

    started = time.time()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    token_counts = {group: 0 for group in sorted(allowed_groups)}
    first_sequence_id: str | None = None
    last_sequence_id: str | None = None
    handles = _open_sources(sources)
    try:
        with output.open("w", encoding="utf-8", buffering=1024 * 1024) as out:
            while True:
                raw_rows: dict[str, str] = {name: handle.readline() for name, handle in handles.items()}
                present = {name: bool(line) for name, line in raw_rows.items()}
                if not any(present.values()):
                    break
                if not all(present.values()):
                    raise ValueError(f"source row-count mismatch after {row_count} rows: {present}")
                rows = {name: _loads(line) for name, line in raw_rows.items()}
                sequence_ids = {str(row.get("sequence_id")) for row in rows.values()}
                if len(sequence_ids) != 1:
                    raise ValueError(f"sequence_id mismatch after {row_count} rows: {sorted(sequence_ids)[:5]}")
                sequence_id = next(iter(sequence_ids))
                base_name = next(iter(sources))
                base = dict(rows[base_name])
                tokens: list[str] = []
                for group in ("keyboard", "mouse_button", "mouse_move", "other"):
                    source_name = group_sources.get(group)
                    if source_name is None:
                        continue
                    group_tokens = _tokens_for_group(rows[source_name], group)
                    tokens.extend(group_tokens)
                    token_counts[group] += len(group_tokens)
                base["model"] = model_name
                base["predicted_tokens"] = _dedupe_preserve_order(tokens)
                base["ensemble_sources"] = {group: group_sources[group] for group in group_sources}
                out.write(json.dumps(base, ensure_ascii=False, sort_keys=True) + "\n")
                row_count += 1
                first_sequence_id = first_sequence_id or sequence_id
                last_sequence_id = sequence_id
                if max_rows is not None and row_count >= int(max_rows):
                    break
    finally:
        for handle in handles.values():
            handle.close()

    payload = {
        "schema": "idm_prediction_ensemble_summary.v1",
        "status": "pass",
        "model_name": model_name,
        "sources": {name: str(path) for name, path in sources.items()},
        "group_sources": dict(group_sources),
        "output_path": str(output),
        "rows": row_count,
        "max_rows": max_rows,
        "first_sequence_id": first_sequence_id,
        "last_sequence_id": last_sequence_id,
        "token_counts": token_counts,
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Post-hoc ensemble over aligned prediction files; not new training evidence by itself.",
    }
    write_json(summary_out, payload)
    return payload
