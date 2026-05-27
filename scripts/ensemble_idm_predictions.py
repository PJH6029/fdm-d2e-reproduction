#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, write_json

MOUSE_PREFIXES = ("MOUSE_DX_", "MOUSE_DY_")
BUTTON_PREFIXES = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")
KEYBOARD_PREFIXES = ("KEY_",)
ENDPOINT_PREFIXES = {
    "mouse": MOUSE_PREFIXES,
    "button": BUTTON_PREFIXES,
    "keyboard": KEYBOARD_PREFIXES,
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            yield row


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _tokens(row: dict[str, Any]) -> list[str]:
    return [str(token) for token in row.get("predicted_tokens", []) or []]


def _endpoint_tokens(tokens: Iterable[str], endpoint: str) -> list[str]:
    prefixes = ENDPOINT_PREFIXES[endpoint]
    return [token for token in tokens if token.startswith(prefixes)]


def _stable_unique(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _select_endpoint_tokens(
    *,
    endpoint: str,
    policy: dict[str, Any],
    rows_by_source: dict[str, dict[str, Any]],
) -> list[str]:
    mode = str(policy.get("mode", "source"))
    source = str(policy.get("source", ""))
    sources = [str(item) for item in policy.get("sources", [])]
    if source and not sources:
        sources = [source]
    if mode == "source":
        if not source:
            raise ValueError(f"{endpoint} source policy requires source")
        return _endpoint_tokens(_tokens(rows_by_source[source]), endpoint)
    if mode == "union":
        if not sources:
            raise ValueError(f"{endpoint} union policy requires sources")
        merged: list[str] = []
        for name in sources:
            merged.extend(_endpoint_tokens(_tokens(rows_by_source[name]), endpoint))
        return _stable_unique(merged)
    if mode == "intersection":
        if not sources:
            raise ValueError(f"{endpoint} intersection policy requires sources")
        token_sets = [set(_endpoint_tokens(_tokens(rows_by_source[name]), endpoint)) for name in sources]
        if not token_sets:
            return []
        common = set.intersection(*token_sets)
        # Preserve order from the first source.
        return [token for token in _endpoint_tokens(_tokens(rows_by_source[sources[0]]), endpoint) if token in common]
    if mode == "source_with_endpoint_gate":
        if not source:
            raise ValueError(f"{endpoint} gated policy requires source")
        gate_sources = [str(item) for item in policy.get("gate_sources", [])]
        if not gate_sources:
            raise ValueError(f"{endpoint} gated policy requires gate_sources")
        source_tokens = _endpoint_tokens(_tokens(rows_by_source[source]), endpoint)
        if not source_tokens:
            return []
        gate_positive = all(_endpoint_tokens(_tokens(rows_by_source[name]), endpoint) for name in gate_sources)
        return source_tokens if gate_positive else []
    raise ValueError(f"unsupported {endpoint} ensemble policy mode: {mode}")


def _other_tokens(row: dict[str, Any]) -> list[str]:
    endpoint_prefixes = MOUSE_PREFIXES + BUTTON_PREFIXES + KEYBOARD_PREFIXES
    return [token for token in _tokens(row) if not token.startswith(endpoint_prefixes)]


def _write_ensemble(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    prediction_paths = {str(name): root / str(path) for name, path in dict(config["prediction_paths"]).items()}
    for name, path in prediction_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"prediction source {name} missing: {path}")
    output_path = root / str(config["output_path"])
    ensure_dir(output_path.parent)
    metadata_source = str(config.get("metadata_source") or next(iter(prediction_paths)))
    if metadata_source not in prediction_paths:
        raise ValueError(f"metadata_source {metadata_source!r} not in prediction_paths")
    policies = dict(config.get("policies", {}))
    for endpoint in ENDPOINT_PREFIXES:
        if endpoint not in policies:
            policies[endpoint] = {"mode": "source", "source": metadata_source}
    include_other = bool(config.get("include_other_tokens", False))
    other_source = str(config.get("other_source", metadata_source))
    max_rows = config.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None
    iterators = {name: _iter_jsonl(path) for name, path in prediction_paths.items()}
    rows_written = 0
    mismatches: list[dict[str, Any]] = []
    endpoint_counts = {endpoint: 0 for endpoint in ENDPOINT_PREFIXES}
    source_counts = {name: _line_count(path) for name, path in prediction_paths.items()}
    started = time.time()
    with output_path.open("w", encoding="utf-8") as out:
        while max_rows is None or rows_written < max_rows:
            rows_by_source: dict[str, dict[str, Any]] = {}
            exhausted: list[str] = []
            for name, iterator in iterators.items():
                try:
                    rows_by_source[name] = next(iterator)
                except StopIteration:
                    exhausted.append(name)
            if exhausted:
                if len(exhausted) != len(iterators):
                    mismatches.append({"code": "source_length_mismatch", "exhausted": exhausted, "row_index": rows_written})
                break
            sequence_ids = {name: str(row.get("sequence_id")) for name, row in rows_by_source.items()}
            if len(set(sequence_ids.values())) != 1:
                mismatches.append({"code": "sequence_id_mismatch", "row_index": rows_written, "sequence_ids": sequence_ids})
                if len(mismatches) >= int(config.get("max_mismatch_examples", 20)):
                    break
            base = dict(rows_by_source[metadata_source])
            tokens: list[str] = []
            tokens.extend(_select_endpoint_tokens(endpoint="mouse", policy=dict(policies["mouse"]), rows_by_source=rows_by_source))
            tokens.extend(_select_endpoint_tokens(endpoint="keyboard", policy=dict(policies["keyboard"]), rows_by_source=rows_by_source))
            tokens.extend(_select_endpoint_tokens(endpoint="button", policy=dict(policies["button"]), rows_by_source=rows_by_source))
            if include_other:
                tokens.extend(_other_tokens(rows_by_source[other_source]))
            tokens = _stable_unique(tokens) or ["NOOP"]
            for endpoint in ENDPOINT_PREFIXES:
                endpoint_counts[endpoint] += int(bool(_endpoint_tokens(tokens, endpoint)))
            base["predicted_tokens"] = tokens
            base["ensemble_sources"] = sorted(prediction_paths)
            base["ensemble_policy_id"] = str(config.get("policy_id", "endpoint_mixture"))
            out.write(json.dumps(base, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
    output_sha = _sha256_file(output_path) if output_path.exists() else None
    summary = {
        "schema": "idm_endpoint_prediction_ensemble.v1",
        "status": "pass" if not mismatches and rows_written > 0 else "fail",
        "policy_id": str(config.get("policy_id", "endpoint_mixture")),
        "prediction_paths": {name: str(path.relative_to(root) if path.is_relative_to(root) else path) for name, path in prediction_paths.items()},
        "source_counts": source_counts,
        "metadata_source": metadata_source,
        "policies": policies,
        "output_path": str(output_path.relative_to(root) if output_path.is_relative_to(root) else output_path),
        "output_rows": rows_written,
        "output_sha256": output_sha,
        "endpoint_positive_rows": endpoint_counts,
        "mismatches": mismatches,
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Post-hoc endpoint ensemble diagnostic only; no G005 completion claim without paper-target validation.",
    }
    summary_path = config.get("summary_path")
    if summary_path:
        write_json(root / str(summary_path), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a post-hoc endpoint ensemble from aligned IDM prediction JSONLs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(args.config)
    summary = _write_ensemble(config, root=root)
    print(json.dumps({"status": summary["status"], "output_rows": summary["output_rows"], "output_path": summary["output_path"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
