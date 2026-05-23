from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _get(data: dict[str, Any] | None, path: str | list[str]) -> Any:
    cur: Any = data
    parts = path if isinstance(path, list) else path.split(".")
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(str(part))
    return cur


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_snapshot(metrics: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "num_examples": _get(metrics, "num_examples"),
        "keyboard_accuracy": _get(metrics, "keyboard.accuracy"),
        "mouse_button_accuracy": _get(metrics, "mouse_button.accuracy"),
        "mouse_button_f1": _get(metrics, "mouse_button.f1"),
        "no_button_false_positive_rate": _get(metrics, "mouse_button.no_button_false_positive_rate"),
        "mouse_move_pearson": _get(metrics, "mouse_move.pearson"),
        "mouse_move_scale_ratio": _get(metrics, "mouse_move.scale_ratio"),
    }


def _score(metrics: dict[str, Any]) -> float:
    keyboard = _float(metrics.get("keyboard_accuracy")) or 0.0
    button_f1 = _float(metrics.get("mouse_button_f1")) or 0.0
    mouse = _float(metrics.get("mouse_move_pearson")) or 0.0
    fpr = _float(metrics.get("no_button_false_positive_rate"))
    fpr_penalty = 0.0 if fpr is None else max(0.0, fpr - 0.10)
    return keyboard + button_f1 + mouse - fpr_penalty


def _top_sweep_row(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("top_variants") or payload.get("rows") or []
    if not isinstance(candidates, list) or not candidates:
        return None
    row = dict(candidates[0])
    metrics_snapshot = _metric_snapshot(row.get("metrics"))
    metadata = dict(row.get("metadata") or {})
    return {
        "variant": row.get("variant"),
        "model_name": row.get("model_name"),
        "config": row.get("config"),
        "metrics_snapshot": metrics_snapshot,
        "score": _score(metrics_snapshot),
        "metadata": {
            "train_records": metadata.get("train_records"),
            "target_records": metadata.get("target_records"),
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "checkpoint_path": metadata.get("checkpoint_path"),
            "metrics_path": metadata.get("metrics_path"),
        },
    }


def _portfolio_row(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "model_name": payload.get("model_name"),
        "metrics": payload.get("metrics"),
        "metrics_snapshot": _metric_snapshot(payload.get("metrics")),
        "score": _score(_metric_snapshot(payload.get("metrics"))),
        "schema": payload.get("schema"),
    }


def _config_summary(path: Path, rel_path: str) -> dict[str, Any]:
    artifact = _artifact(path, rel_path)
    cfg = load_config(path) if path.exists() else {}
    required = {
        "model_name": cfg.get("model_name"),
        "feature_mode": cfg.get("feature_mode"),
        "model_arch": cfg.get("model_arch", "mlp"),
        "epochs": cfg.get("epochs"),
        "category_threshold_mode": cfg.get("category_threshold_mode"),
        "category_calibration_beta": cfg.get("category_calibration_beta"),
        "category_calibration_max_examples": cfg.get("category_calibration_max_examples"),
        "mouse_target_mode": cfg.get("mouse_target_mode", "mean"),
        "mouse_head_mode": cfg.get("mouse_head_mode"),
        "mouse_emit_mode": cfg.get("mouse_emit_mode", "single"),
        "mouse_max_tokens_per_axis": cfg.get("mouse_max_tokens_per_axis"),
        "mouse_output_gain_mode": cfg.get("mouse_output_gain_mode"),
        "mouse_gain_calibration_max_examples": cfg.get("mouse_gain_calibration_max_examples"),
        "training_cache_shard_assignment": cfg.get("training_cache_shard_assignment"),
        "prediction_workers": cfg.get("prediction_workers"),
        "train_records_glob": cfg.get("train_records_glob"),
        "target_records_glob": cfg.get("target_records_glob"),
    }
    return {"artifact": artifact, "config": required}


def build_idm_exploration_summary(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []

    prerequisites = {}
    for name, rel_path in dict(config.get("prerequisites", {})).items():
        rel = str(rel_path)
        payload = _load_json(root_path / rel)
        prerequisites[name] = {"artifact": _artifact(root_path / rel, rel), "status": (payload or {}).get("status"), "error_count": (payload or {}).get("error_count")}
        if (payload or {}).get("status") != "pass":
            findings.append({"severity": "error", "code": "prerequisite_not_pass", "name": name, "path": rel, "status": (payload or {}).get("status")})

    current_metrics_path = str(config.get("current_full_idm_metrics"))
    current_metrics = _load_json(root_path / current_metrics_path)
    current_snapshot = _metric_snapshot(current_metrics)
    if current_metrics is None:
        findings.append({"severity": "error", "code": "missing_current_full_idm_metrics", "path": current_metrics_path})

    sweep_rows = []
    for row in config.get("sweep_evidence", []):
        rel = str(row["path"])
        payload = _load_json(root_path / rel)
        top = _top_sweep_row(payload)
        item = {
            "id": str(row["id"]),
            "scale": row.get("scale"),
            "path": rel,
            "artifact": _artifact(root_path / rel, rel),
            "num_runs": (payload or {}).get("num_runs"),
            "top_variant": top,
        }
        sweep_rows.append(item)
        if not item["artifact"]["exists"] or not top:
            findings.append({"severity": "error", "code": "missing_or_empty_sweep_evidence", "id": item["id"], "path": rel})

    portfolio_rows = []
    for row in config.get("portfolio_evidence", []):
        rel = str(row["path"])
        payload = _load_json(root_path / rel)
        portfolio = _portfolio_row(payload)
        item = {
            "id": str(row["id"]),
            "scale": row.get("scale"),
            "path": rel,
            "artifact": _artifact(root_path / rel, rel),
            "portfolio": portfolio,
        }
        portfolio_rows.append(item)
        if not item["artifact"]["exists"] or not portfolio:
            findings.append({"severity": "error", "code": "missing_or_empty_portfolio_evidence", "id": item["id"], "path": rel})

    selected_rows: list[dict[str, Any]] = []
    for item in sweep_rows:
        if item.get("top_variant"):
            selected_rows.append({"source": item["id"], **item["top_variant"]})
    for item in portfolio_rows:
        if item.get("portfolio"):
            selected_rows.append({"source": item["id"], **item["portfolio"]})
    ranked = sorted(selected_rows, key=lambda row: _float(row.get("score")) or -999.0, reverse=True)

    criteria = dict(config.get("progress_criteria", {}))
    best = ranked[0] if ranked else {}
    best_metrics = dict(best.get("metrics_snapshot", {}))
    if bool(criteria.get("requires_medium_candidate", False)):
        medium_sources = {
            str(item["id"])
            for item in sweep_rows
            if str(item.get("scale")) == "medium_h200" and item.get("top_variant")
        }
        if not any(str(row.get("source")) in medium_sources for row in ranked):
            findings.append({"severity": "error", "code": "missing_medium_h200_candidate"})
    if (_float(best_metrics.get("mouse_button_f1")) or 0.0) < float(criteria.get("min_small_medium_mouse_button_f1", 0.0)):
        findings.append({"severity": "error", "code": "small_medium_mouse_button_f1_gate_not_met", "best": best_metrics})
    fpr = _float(best_metrics.get("no_button_false_positive_rate"))
    if fpr is not None and fpr > float(criteria.get("max_small_medium_no_button_fpr", 1.0)):
        findings.append({"severity": "error", "code": "small_medium_no_button_fpr_gate_not_met", "best": best_metrics})
    if (_float(best_metrics.get("mouse_move_pearson")) or 0.0) < float(criteria.get("min_small_medium_mouse_pearson", 0.0)):
        findings.append({"severity": "error", "code": "small_medium_mouse_pearson_gate_not_met", "best": best_metrics})

    candidate_configs = []
    for rel_path in config.get("full_corpus_candidate_configs", []):
        summary = _config_summary(root_path / str(rel_path), str(rel_path))
        candidate_configs.append(summary)
        cfg = summary["config"]
        if not summary["artifact"]["exists"]:
            findings.append({"severity": "error", "code": "missing_candidate_config", "path": str(rel_path)})
        if cfg.get("category_threshold_mode") != "group_fbeta_calibrated":
            findings.append({"severity": "error", "code": "candidate_missing_streaming_category_calibration", "path": str(rel_path)})
        if cfg.get("mouse_output_gain_mode") != "train_abs_ratio":
            findings.append({"severity": "error", "code": "candidate_missing_mouse_gain_calibration", "path": str(rel_path)})
        if cfg.get("training_cache_shard_assignment") != "greedy_rows":
            findings.append({"severity": "error", "code": "candidate_missing_greedy_cache_assignment", "path": str(rel_path)})

    implementation_hooks = dict(config.get("implementation_hooks", {}))
    for hook, expected in implementation_hooks.items():
        rel = str(expected.get("path"))
        patterns = [str(pattern) for pattern in expected.get("patterns", [])]
        text = (root_path / rel).read_text(encoding="utf-8") if (root_path / rel).exists() else ""
        for pattern in patterns:
            if pattern not in text:
                findings.append({"severity": "error", "code": "missing_implementation_hook", "hook": hook, "path": rel, "pattern": pattern})

    status = "pass" if not any(finding.get("severity") == "error" for finding in findings) else "fail"
    return {
        "schema": "idm_exploration_summary.v1",
        "status": status,
        "error_count": sum(1 for finding in findings if finding.get("severity") == "error"),
        "findings": findings,
        "prerequisites": prerequisites,
        "diagnosed_failure_links": config.get("diagnosed_failure_links", []),
        "current_full_idm": {
            "metrics_path": current_metrics_path,
            "artifact": _artifact(root_path / current_metrics_path, current_metrics_path),
            "metrics": current_snapshot,
        },
        "sweep_evidence": sweep_rows,
        "portfolio_evidence": portfolio_rows,
        "ranked_candidates": ranked[:10],
        "progress_criteria": criteria,
        "full_corpus_candidate_configs": candidate_configs,
        "selected_g005_order": config.get("selected_g005_order", []),
        "budget": config.get("budget", {}),
        "claim_boundary": "G004 ranks bounded IDM exploration evidence and publishes full-corpus candidates; it does not claim paper-target/full-corpus success.",
    }


def write_idm_exploration_summary(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_idm_exploration_summary(config, root=root)
    output = output_path or config.get("output_path")
    if not output:
        raise ValueError("output_path is required")
    write_json(Path(root) / output, payload)
    return payload
