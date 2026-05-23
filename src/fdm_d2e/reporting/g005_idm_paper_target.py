from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.io_utils import read_json, write_json


def _path(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(root: Path, value: str | Path | None) -> dict[str, Any] | None:
    p = _path(root, value)
    if p is None or not p.exists() or not p.is_file():
        return None
    payload = read_json(p)
    return payload if isinstance(payload, dict) else None


def _file_metadata(root: Path, value: str | Path | None) -> dict[str, Any]:
    p = _path(root, value)
    if p is None:
        return {"path": None, "exists": False, "bytes": 0, "sha256": None}
    if not p.exists() or not p.is_file():
        return {"path": str(p), "exists": False, "bytes": 0, "sha256": None}
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "exists": True, "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def _get(data: dict[str, Any] | None, path: Sequence[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_monitor_status(path: Path | None, expected_gpus: int) -> dict[str, Any]:
    status: dict[str, Any] = {
        "rows": 0,
        "sample_count": 0,
        "unique_gpu_indices": [],
        "expected_gpus": expected_gpus,
        "covers_expected_gpus": False,
    }
    if path is None or not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return status
    index_col = None
    sample_col = None
    samples: set[str] = set()
    gpu_indices: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for raw_row in csv.reader(handle):
            row = [cell.strip() for cell in raw_row]
            if not row:
                continue
            lowered = [cell.lower() for cell in row]
            if "index" in lowered:
                index_col = lowered.index("index")
                sample_col = lowered.index("sample_unix") if "sample_unix" in lowered else None
                continue
            if index_col is None:
                index_col = 1 if len(row) > 1 else 0
            if index_col < len(row):
                gpu_indices.add(row[index_col])
            if sample_col is not None and sample_col < len(row):
                samples.add(row[sample_col])
            status["rows"] += 1
    status["unique_gpu_indices"] = sorted(gpu_indices)
    status["sample_count"] = len(samples) if samples else (max(1, status["rows"] // max(1, len(gpu_indices))) if gpu_indices else 0)
    status["covers_expected_gpus"] = len(gpu_indices) >= expected_gpus
    return status


def _paper_target_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    targets = dict(_get(contract, ["target_sequence", "phase_1", "primary_targets"]) or {})
    return [
        {
            "name": "pearson_x",
            "path": ["paper_compatible", "mouse_move", "pearson_x"],
            "direction": "higher",
            "target": targets.get("pearson_x"),
        },
        {
            "name": "pearson_y",
            "path": ["paper_compatible", "mouse_move", "pearson_y"],
            "direction": "higher",
            "target": targets.get("pearson_y"),
        },
        {
            "name": "keyboard_accuracy",
            "path": ["paper_compatible", "keyboard", "key_accuracy"],
            "direction": "higher",
            "target": targets.get("keyboard_accuracy"),
        },
        {
            "name": "mouse_button_accuracy",
            "path": ["paper_compatible", "mouse_button", "button_accuracy"],
            "direction": "higher",
            "target": targets.get("mouse_button_accuracy"),
        },
        {
            "name": "scale_ratio_x",
            "path": ["paper_compatible", "mouse_move", "scale_ratio_x"],
            "direction": "lower",
            "target": targets.get("scale_ratio_x_max"),
        },
        {
            "name": "scale_ratio_y",
            "path": ["paper_compatible", "mouse_move", "scale_ratio_y"],
            "direction": "lower",
            "target": targets.get("scale_ratio_y_max"),
        },
    ]


def _check_target(group_metrics: dict[str, Any], row: dict[str, Any], *, margin: float = 0.0) -> dict[str, Any]:
    actual = _as_float(_get(group_metrics, row["path"]))
    target = _as_float(row.get("target"))
    if actual is None or target is None:
        passed = False
    elif row.get("direction") == "lower":
        passed = actual <= target - margin
    else:
        passed = actual >= target + margin
    return {
        "name": row["name"],
        "path": row["path"],
        "direction": row.get("direction", "higher"),
        "target": target,
        "actual": actual,
        "margin": margin,
        "passed": passed,
    }


def _strict_target_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in config.get("strict_local_targets", []):
        item = dict(row)
        item["path"] = [str(part) for part in item.get("path", [])]
        rows.append(item)
    return rows


def _check_strict_target(group_metrics: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    actual = _as_float(_get(group_metrics, row["path"]))
    baseline = _as_float(row.get("baseline"))
    target = _as_float(row.get("target"))
    min_delta = float(row.get("min_delta", 0.0) or 0.0)
    direction = str(row.get("direction", "higher"))
    if target is None and baseline is not None:
        target = baseline - min_delta if direction == "lower" else baseline + min_delta
    if actual is None or target is None:
        passed = False
    elif direction == "lower":
        passed = actual <= target
    else:
        passed = actual >= target
    return {
        "name": str(row.get("name", ".".join(row["path"]))),
        "path": row["path"],
        "direction": direction,
        "baseline": baseline,
        "target": target,
        "actual": actual,
        "min_delta": min_delta,
        "passed": passed,
    }


def validate_g005_idm_paper_target(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    paths = dict(config.get("paths", {}))
    expected_gpus = int(config.get("expected_gpus", 4))
    required_splits = [str(tag) for tag in config.get("required_splits", ["temporal", "heldout_recording", "heldout_game"])]
    min_train_records = int(config.get("min_train_records", 0))
    min_target_records = int(config.get("min_target_records", 0))
    paper_margin = float(config.get("paper_target_margin", 0.0) or 0.0)
    require_all_splits_beat_targets = bool(config.get("require_all_splits_beat_targets", False))

    contract = _load_json(root_path, paths.get("gidm_baseline_contract"))
    paper_metrics = _load_json(root_path, paths.get("paper_metrics"))
    metadata = _load_json(root_path, paths.get("checkpoint_metadata"))
    train_summary = _load_json(root_path, paths.get("train_summary"))
    run_summary = _load_json(root_path, paths.get("run_summary"))
    split_stats_summary = _load_json(root_path, paths.get("split_stats_summary"))
    gpu_monitor_path = _path(root_path, paths.get("gpu_monitor"))
    gpu_monitor_status = _gpu_monitor_status(gpu_monitor_path, expected_gpus)

    findings: list[dict[str, Any]] = []
    for name, payload in [
        ("gidm_baseline_contract", contract),
        ("paper_metrics", paper_metrics),
        ("checkpoint_metadata", metadata),
        ("train_summary", train_summary),
        ("run_summary", run_summary),
        ("split_stats_summary", split_stats_summary),
    ]:
        if payload is None:
            findings.append({"severity": "error", "code": f"missing_{name}", "path": paths.get(name)})

    if contract and contract.get("status") != "pass":
        findings.append({"severity": "error", "code": "gidm_baseline_contract_not_pass", "status": contract.get("status")})
    if paper_metrics and paper_metrics.get("status") != "pass":
        findings.append({"severity": "error", "code": "paper_metrics_not_pass", "status": paper_metrics.get("status"), "error_count": paper_metrics.get("error_count")})
    if run_summary and int(run_summary.get("exit_code", 999)) != 0:
        findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "exit_code": run_summary.get("exit_code")})
    if run_summary and int(run_summary.get("nproc_per_node", 0) or 0) != expected_gpus:
        findings.append({"severity": "error", "code": "run_summary_nproc_mismatch", "actual": run_summary.get("nproc_per_node"), "expected": expected_gpus})
    run_gpu_status = run_summary.get("gpu_monitor_status") if isinstance(run_summary, dict) else None
    if isinstance(run_gpu_status, dict) and bool(config.get("require_gpu_monitor_covers_expected_gpus", True)):
        if run_gpu_status.get("covers_expected_gpus") is not True:
            findings.append({"severity": "error", "code": "run_summary_gpu_monitor_missing_expected_gpus", "actual": run_gpu_status})
    if gpu_monitor_status["rows"] < int(config.get("min_gpu_monitor_rows", expected_gpus)):
        findings.append({"severity": "error", "code": "gpu_monitor_too_few_rows", "actual": gpu_monitor_status["rows"]})
    if bool(config.get("require_gpu_monitor_covers_expected_gpus", True)) and not gpu_monitor_status["covers_expected_gpus"]:
        findings.append({"severity": "error", "code": "gpu_monitor_does_not_cover_expected_gpus", "actual": gpu_monitor_status})

    train_records = int(metadata.get("train_records", 0) or 0) if metadata else 0
    target_records = int(metadata.get("target_records", 0) or 0) if metadata else 0
    if min_train_records and train_records < min_train_records:
        findings.append({"severity": "error", "code": "train_records_below_full_corpus_min", "actual": train_records, "expected_min": min_train_records})
    if min_target_records and target_records < min_target_records:
        findings.append({"severity": "error", "code": "target_records_below_full_corpus_min", "actual": target_records, "expected_min": min_target_records})
    if metadata and int(_get(metadata, ["distributed", "world_size"]) or 0) != expected_gpus:
        findings.append({"severity": "error", "code": "metadata_world_size_mismatch", "actual": _get(metadata, ["distributed", "world_size"]), "expected": expected_gpus})

    checkpoint_meta = _file_metadata(root_path, paths.get("checkpoint"))
    if not checkpoint_meta["exists"]:
        findings.append({"severity": "error", "code": "missing_checkpoint", "path": paths.get("checkpoint")})

    split_outputs = split_stats_summary.get("outputs", []) if isinstance(split_stats_summary, dict) else []
    split_by_name = {str(row.get("split")): row for row in split_outputs if isinstance(row, dict)}
    if split_stats_summary and split_stats_summary.get("status") != "pass":
        findings.append({"severity": "error", "code": "split_stats_summary_not_pass", "status": split_stats_summary.get("status")})
    for split in required_splits:
        row = split_by_name.get(split)
        if not row or row.get("status") != "pass":
            findings.append({"severity": "error", "code": "missing_or_failed_split_stats", "split": split, "row": row})

    aggregate_target_results: list[dict[str, Any]] = []
    split_target_results: dict[str, list[dict[str, Any]]] = {}
    strict_target_results: list[dict[str, Any]] = []
    if contract and paper_metrics:
        groups = paper_metrics.get("groups", {}) if isinstance(paper_metrics.get("groups"), dict) else {}
        all_metrics = groups.get("all", {}) if isinstance(groups.get("all"), dict) else {}
        for row in _paper_target_rows(contract):
            result = _check_target(all_metrics, row, margin=paper_margin)
            aggregate_target_results.append(result)
            if not result["passed"]:
                findings.append({"severity": "error", "code": "paper_target_not_met", **result})
        for split in required_splits:
            split_metrics = groups.get(f"eval_split:{split}", {}) if isinstance(groups.get(f"eval_split:{split}"), dict) else {}
            split_results = [_check_target(split_metrics, row, margin=paper_margin) for row in _paper_target_rows(contract)]
            split_target_results[split] = split_results
            if require_all_splits_beat_targets:
                for result in split_results:
                    if not result["passed"]:
                        findings.append({"severity": "error", "code": "split_paper_target_not_met", "split": split, **result})
        for row in _strict_target_rows(config):
            result = _check_strict_target(all_metrics, row)
            strict_target_results.append(result)
            if not result["passed"]:
                findings.append({"severity": "error", "code": "strict_local_target_not_met", **result})

    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g005_idm_paper_target_audit.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "goal_id": str(config.get("goal_id", "G005-idm-full-paper-target")),
        "model_name": str(config.get("model_name", "")),
        "paths": paths,
        "expected_gpus": expected_gpus,
        "min_train_records": min_train_records,
        "min_target_records": min_target_records,
        "paper_target_margin": paper_margin,
        "require_all_splits_beat_targets": require_all_splits_beat_targets,
        "train_records": train_records,
        "target_records": target_records,
        "checkpoint": checkpoint_meta,
        "gpu_monitor_status": gpu_monitor_status,
        "aggregate_target_results": aggregate_target_results,
        "split_target_results": split_target_results,
        "strict_target_results": strict_target_results,
        "split_stats_outputs": split_outputs,
        "findings": findings,
        "claim_boundary": "G005 passes only when our full-corpus 4xH200 IDM run beats paper-reported G-IDM targets under the local paper-compatible token metric and records split/GPU evidence. It is not the exact released G-IDM comparison; that remains G006.",
    }
    return payload


def write_g005_idm_paper_target_audit(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = validate_g005_idm_paper_target(config, root=root)
    out = output_path or config.get("output_path", "artifacts/idm/g005_idm_paper_target_audit.json")
    write_json(Path(root) / out if not Path(out).is_absolute() else out, payload)
    return payload
