from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _get(data: dict[str, Any] | None, path: str | list[str]) -> Any:
    cur: Any = data
    parts = path if isinstance(path, list) else path.split(".")
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(str(part))
    return cur


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = _as_float(row.get(metric))
    if value is None:
        return None
    if metric in {"scale_ratio_x", "scale_ratio_y"}:
        return value
    return value / 100.0 if value > 1.0 else value


def _paper_rows(config: dict[str, Any], model: str) -> list[dict[str, Any]]:
    rows = []
    for row in config.get("paper_reported", {}).get("in_distribution_rows", []):
        if str(row.get("model")) == model:
            rows.append(dict(row))
    return rows


def _aggregate_paper_targets(config: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "pearson_x",
        "pearson_y",
        "scale_ratio_x",
        "scale_ratio_y",
        "keyboard_accuracy",
        "mouse_button_accuracy",
    ]
    rows = _paper_rows(config, "G-IDM")
    per_game: list[dict[str, Any]] = []
    for row in rows:
        normalized = {"game": row.get("game"), "model": row.get("model")}
        for metric in metrics:
            normalized[metric] = _round(_metric_value(row, metric))
        per_game.append(normalized)
    aggregate = {
        metric: _round(_mean([value for value in (_metric_value(row, metric) for row in rows) if value is not None]))
        for metric in metrics
    }
    derived = {
        "source": "D2E paper Tables 4 and 5, six in-distribution G-IDM rows",
        "row_count": len(rows),
        "per_game": per_game,
        "aggregate": aggregate,
        "unreported_metrics": {
            "mouse_button_f1": {
                "status": "not_paper_reported",
                "required_source": "local exact-split postprocessor over released G-IDM predictions",
            },
            "no_button_false_positive_rate": {
                "status": "not_paper_reported",
                "required_source": "local exact-split postprocessor over released G-IDM predictions",
            },
        },
    }
    return derived


def _compare_aggregate(derived: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    aggregate = derived.get("aggregate", {})
    tolerance = _as_float(expected.get("tolerance", 0.0006)) or 0.0006
    for metric, expected_value in dict(expected.get("metrics", {})).items():
        actual = _as_float(aggregate.get(metric))
        exp = _as_float(expected_value)
        if actual is None or exp is None:
            findings.append({"severity": "error", "code": "missing_paper_target_metric", "metric": metric, "actual": actual, "expected": exp})
            continue
        if math.fabs(actual - exp) > tolerance:
            findings.append(
                {
                    "severity": "error",
                    "code": "paper_target_metric_mismatch",
                    "metric": metric,
                    "actual": actual,
                    "expected": exp,
                    "tolerance": tolerance,
                }
            )
    return findings


def _source_evidence(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = dict(config.get("source_evidence", {}))
    local_paths = {}
    for key, rel_path in dict(source.get("local_paths", {})).items():
        rel = str(rel_path)
        local_paths[str(key)] = _artifact(root / rel, rel)
    source["local_paths"] = local_paths
    return source


def _external_entry(manifest: dict[str, Any] | None, path: str) -> dict[str, Any] | None:
    entries = (manifest or {}).get("entries", [])
    if isinstance(entries, dict):
        entry = entries.get(path)
        return dict(entry) if isinstance(entry, dict) else None
    for row in entries:
        if isinstance(row, dict) and row.get("path") == path:
            return dict(row)
    return None


def _split_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    split_config = dict(config.get("exact_split_contract", {}))
    manifest = _load_json(root / str(split_config.get("external_artifact_manifest", "")))
    split_artifacts = {}
    for name, rel_path in dict(split_config.get("split_artifacts", {})).items():
        rel = str(rel_path)
        split_artifacts[str(name)] = _artifact(root / rel, rel)
    required_external = {}
    for name, rel_path in dict(split_config.get("required_external_artifacts", {})).items():
        entry = _external_entry(manifest, str(rel_path))
        required_external[str(name)] = {
            "path": str(rel_path),
            "exists": bool((entry or {}).get("exists")),
            "bytes": (entry or {}).get("bytes"),
            "sha256": (entry or {}).get("sha256"),
            "fingerprint": (entry or {}).get("fingerprint"),
            "storage_uri": (entry or {}).get("storage_uri"),
            "proof": (entry or {}).get("proof"),
        }
    split_config["external_artifact_manifest_artifact"] = _artifact(root / str(split_config.get("external_artifact_manifest", "")), str(split_config.get("external_artifact_manifest", "")))
    split_config["split_artifacts"] = split_artifacts
    split_config["required_external_artifacts"] = required_external
    return split_config


def _validate_contract(payload: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    source = payload["source_evidence"]
    if not source.get("d2e_repo_commit"):
        findings.append({"severity": "error", "code": "missing_d2e_repo_commit"})
    if not source.get("hf_model", {}).get("revision"):
        findings.append({"severity": "error", "code": "missing_hf_model_revision"})
    if source.get("hf_model", {}).get("gated") not in (False, "false"):
        findings.append({"severity": "error", "code": "hf_model_is_gated_or_unknown", "value": source.get("hf_model", {}).get("gated")})
    siblings = source.get("hf_model", {}).get("siblings", [])
    if "model.safetensors" not in siblings:
        findings.append({"severity": "error", "code": "missing_hf_model_weight_sibling", "required": "model.safetensors"})
    for name, artifact in source.get("local_paths", {}).items():
        if not artifact.get("exists"):
            findings.append({"severity": "error", "code": "missing_source_artifact", "name": name, "path": artifact.get("path")})

    paper = payload["paper_reported_targets"]
    if paper.get("row_count") != 6:
        findings.append({"severity": "error", "code": "unexpected_gidm_paper_row_count", "row_count": paper.get("row_count")})
    findings.extend(_compare_aggregate(paper, config.get("paper_reported", {}).get("expected_aggregate", {})))
    if _get(paper, "unreported_metrics.mouse_button_f1.status") != "not_paper_reported":
        findings.append({"severity": "error", "code": "mouse_button_f1_must_not_be_paper_fabricated"})

    protocol = payload["official_metric_protocol"]
    if protocol.get("bin_ms") != 50:
        findings.append({"severity": "error", "code": "metric_protocol_bin_ms_mismatch", "actual": protocol.get("bin_ms")})
    if protocol.get("empty_bins_as_correct") is not False:
        findings.append({"severity": "error", "code": "metric_protocol_empty_bins_mismatch", "actual": protocol.get("empty_bins_as_correct")})
    if protocol.get("autoregressive") is not True or protocol.get("teacher_forcing") is not False:
        findings.append({"severity": "error", "code": "metric_protocol_must_be_autoregressive_no_teacher_forcing"})

    inference = payload["official_inference_defaults"]
    if _as_float(inference.get("time_shift_seconds")) != 0.1:
        findings.append({"severity": "error", "code": "inference_time_shift_mismatch", "actual": inference.get("time_shift_seconds")})
    if inference.get("video_filter") != "fps=60,scale=448:448":
        findings.append({"severity": "error", "code": "inference_video_filter_mismatch", "actual": inference.get("video_filter")})

    split_contract = payload["exact_split_contract"]
    for name, artifact in split_contract.get("split_artifacts", {}).items():
        if not artifact.get("exists") or not artifact.get("sha256"):
            findings.append({"severity": "error", "code": "missing_split_artifact_hash", "name": name, "path": artifact.get("path")})
    for name, artifact in split_contract.get("required_external_artifacts", {}).items():
        if not artifact.get("exists") or not (artifact.get("sha256") or artifact.get("fingerprint")):
            findings.append({"severity": "error", "code": "missing_external_artifact_proof", "name": name, "path": artifact.get("path")})

    fallback = payload["fallback_semantics"]
    if fallback.get("released_gidm_unavailable") != "block_exact_split_goal":
        findings.append({"severity": "error", "code": "fallback_must_block_exact_split_goal"})
    if fallback.get("paper_metrics_substitute_for_exact_split") is not False:
        findings.append({"severity": "error", "code": "paper_metrics_must_not_substitute_exact_split"})
    if fallback.get("requires_error_log") is not True:
        findings.append({"severity": "error", "code": "fallback_must_require_error_log"})
    return findings


def build_gidm_baseline_contract(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    paper_targets = _aggregate_paper_targets(config)
    payload: dict[str, Any] = {
        "schema": "gidm_baseline_contract.v1",
        "status": "pending",
        "error_count": 0,
        "source_evidence": _source_evidence(config, root_path),
        "paper_reported_targets": paper_targets,
        "official_metric_protocol": config.get("official_metric_protocol", {}),
        "official_inference_defaults": config.get("official_inference_defaults", {}),
        "target_sequence": config.get("target_sequence", {}),
        "exact_split_contract": _split_contract(config, root_path),
        "exact_split_inference_plan": config.get("exact_split_inference_plan", {}),
        "environment_pinning": config.get("environment_pinning", {}),
        "fallback_semantics": config.get("fallback_semantics", {}),
        "claim_boundaries": config.get("claim_boundaries", []),
    }
    findings = _validate_contract(payload, config)
    payload["findings"] = findings
    payload["error_count"] = sum(1 for finding in findings if finding.get("severity") == "error")
    payload["status"] = "pass" if payload["error_count"] == 0 else "fail"
    return payload


def write_gidm_baseline_contract(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_gidm_baseline_contract(config, root=root)
    output = output_path or config.get("output_path")
    if not output:
        raise ValueError("output_path is required")
    write_json(Path(root) / output, payload)
    return payload
