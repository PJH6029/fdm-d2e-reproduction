from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import read_json, stable_hash_json, write_json
from fdm_d2e.tokenization.fdm1_actions import fit_signed_exponential_boundaries_from_histogram


def _event_delta(event: dict[str, Any]) -> tuple[float, float]:
    etype = str(event.get("type", event.get("topic", ""))).lower()
    if etype not in {"mouse_move", "mouse/raw", "mouse_raw", "raw_mouse"}:
        return 0.0, 0.0
    return float(event.get("dx", event.get("last_x", 0)) or 0), float(event.get("dy", event.get("last_y", 0)) or 0)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def collect_mouse_magnitude_histogram(
    input_paths: Sequence[str | Path],
    *,
    split: str = "train_core",
    max_records: int | None = None,
) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    records_seen = 0
    records_used = 0
    mouse_events = 0
    zero_events = 0
    for path in input_paths:
        for row in iter_jsonl(path):
            records_seen += 1
            if split and str(row.get("split")) != split:
                continue
            records_used += 1
            for event in row.get("events", []) or []:
                dx, dy = _event_delta(event)
                if dx == 0 and dy == 0:
                    zero_events += 1
                    continue
                for value in (dx, dy):
                    mag = int(round(abs(value)))
                    if mag > 0:
                        histogram[mag] += 1
                mouse_events += 1
            if max_records is not None and records_used >= int(max_records):
                break
        if max_records is not None and records_used >= int(max_records):
            break
    return {
        "histogram": dict(sorted(histogram.items())),
        "records_seen": records_seen,
        "records_used": records_used,
        "mouse_events": mouse_events,
        "zero_events": zero_events,
    }


def build_fitted_mouse_bins(
    input_paths: Sequence[str | Path],
    *,
    base_tokenization_config: str | Path = "configs/tokenization/fdm1_action_slots.json",
    bins_output_path: str | Path = "artifacts/sources/fdm1_g003_fitted_mouse_bins.json",
    fitted_config_path: str | Path = "artifacts/sources/fdm1_action_slots_fitted_config.json",
    split: str = "train_core",
    max_records: int | None = None,
) -> dict[str, Any]:
    collected = collect_mouse_magnitude_histogram(input_paths, split=split, max_records=max_records)
    boundaries = fit_signed_exponential_boundaries_from_histogram(collected["histogram"])
    base = read_json(base_tokenization_config)
    fitted = dict(base)
    mouse = dict(fitted.get("mouse_move", {}))
    mouse["positive_boundaries_default"] = [int(v) if float(v).is_integer() else v for v in boundaries]
    mouse["fit_policy"] = f"fitted globally on split={split} from decoded D2E 50ms rows"
    fitted["mouse_move"] = mouse
    fitted["fitted_from"] = {
        "input_paths": [str(path) for path in input_paths],
        "split": split,
        "max_records": max_records,
        "base_tokenization_config": str(base_tokenization_config),
    }
    fitted["fingerprint"] = stable_hash_json({"base": base, "boundaries": list(boundaries), "fitted_from": fitted["fitted_from"]})
    summary = {
        "schema": "fdm1_g003_fitted_mouse_bins.v1",
        "canonical_roadmap": "ROADMAP.md",
        "status": "pass" if len(boundaries) == 24 and all(a < b for a, b in zip(boundaries, boundaries[1:])) else "fail",
        "input_paths": [str(path) for path in input_paths],
        "split": split,
        "max_records": max_records,
        "records_seen": collected["records_seen"],
        "records_used": collected["records_used"],
        "mouse_events": collected["mouse_events"],
        "zero_events": collected["zero_events"],
        "unique_magnitudes": len(collected["histogram"]),
        "positive_boundaries": list(boundaries),
        "base_tokenization_config": str(base_tokenization_config),
        "fitted_tokenization_config": str(fitted_config_path),
        "fingerprint": fitted["fingerprint"],
    }
    write_json(bins_output_path, summary)
    write_json(fitted_config_path, fitted)
    return {"summary": summary, "config": fitted}


__all__ = ["build_fitted_mouse_bins", "collect_mouse_magnitude_histogram", "iter_jsonl"]
