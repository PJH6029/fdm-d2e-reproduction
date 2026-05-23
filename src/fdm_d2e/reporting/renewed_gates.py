from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_evidence(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _get_nested(data: dict[str, Any] | None, path: str | list[str]) -> Any:
    cur: Any = data
    parts = path if isinstance(path, list) else path.split(".")
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(str(part))
    return cur


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_from_gate(gate: dict[str, Any]) -> float | None:
    explicit = _as_float(gate.get("target"))
    if explicit is not None:
        return explicit
    baseline = _as_float(gate.get("baseline"))
    maximum = _as_float(gate.get("maximum", 1.0))
    remaining_error_reduction = _as_float(gate.get("remaining_error_reduction"))
    if baseline is None or maximum is None or remaining_error_reduction is None:
        return None
    return baseline + remaining_error_reduction * (maximum - baseline)


def _check_threshold(actual: float | None, gate: dict[str, Any], *, source: str) -> dict[str, Any]:
    name = str(gate["name"])
    direction = str(gate.get("direction", "higher"))
    target = _target_from_gate(gate)
    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "direction": direction,
        "actual": actual,
        "target": target,
        "status": "fail",
    }
    if actual is None:
        row["reason"] = "missing_or_non_numeric_metric"
        return row
    if target is None:
        row["reason"] = "missing_or_non_numeric_target"
        return row
    if direction == "higher":
        passed = actual >= target
    elif direction == "lower":
        passed = actual <= target
    else:
        row["reason"] = "unsupported_direction"
        row["direction"] = direction
        return row
    row["status"] = "pass" if passed else "fail"
    row["margin"] = actual - target if direction == "higher" else target - actual
    return row


def _glob_matches(root: Path, pattern: str) -> list[str]:
    matches = sorted(glob.glob(str(root / pattern)))
    rel_matches = []
    for match in matches:
        path = Path(match)
        try:
            rel_matches.append(str(path.relative_to(root)))
        except ValueError:
            rel_matches.append(str(path))
    return rel_matches


def _comparison_rows(path: Path, split: str, model_name: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("comparisons", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("model")) != model_name:
            continue
        endpoint = row.get("endpoint")
        if endpoint:
            rows[str(endpoint)] = {"split": split, **row}
    return rows


def _split_outputs(root: Path, split_summary_path: str | None) -> list[dict[str, Any]]:
    if not split_summary_path:
        return []
    payload = _load_json(root / split_summary_path)
    if not isinstance(payload, dict):
        return []
    outputs = payload.get("outputs", [])
    return [dict(row) for row in outputs if isinstance(row, dict)]


def _old_goal_statuses(root: Path, goals_path: str | None) -> dict[str, str]:
    if not goals_path:
        return {}
    payload = _load_json(root / goals_path)
    if not isinstance(payload, dict):
        return {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", []) if isinstance(goal, dict)}


def _select_old_archive_goals_path(root: Path, pattern: str | None, required_goal_ids: list[str]) -> tuple[str | None, dict[str, str]]:
    if not pattern:
        return None, {}
    fallback_path: str | None = None
    fallback_statuses: dict[str, str] = {}
    for rel_path in reversed(_glob_matches(root, pattern)):
        statuses = _old_goal_statuses(root, rel_path)
        if fallback_path is None:
            fallback_path = rel_path
            fallback_statuses = statuses
        if required_goal_ids and all(statuses.get(goal_id) == "complete" for goal_id in required_goal_ids):
            return rel_path, statuses
    return fallback_path, fallback_statuses


def validate_renewed_metric_gates(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    gate_failures: list[dict[str, Any]] = []
    old_evidence_findings: list[dict[str, Any]] = []

    paths = {str(key): str(value) for key, value in dict(config.get("paths", {})).items()}
    candidate_metrics_path = paths.get("candidate_metrics")
    candidate_metrics = _load_json(root_path / candidate_metrics_path) if candidate_metrics_path else None
    candidate_artifact = _file_evidence(root_path / candidate_metrics_path, candidate_metrics_path) if candidate_metrics_path else None
    if candidate_metrics_path and candidate_metrics is None:
        findings.append({"severity": "error", "code": "missing_candidate_metrics", "path": candidate_metrics_path})

    aux_metrics_path = paths.get("aux_metrics")
    aux_metrics = _load_json(root_path / aux_metrics_path) if aux_metrics_path else None
    aux_artifact = _file_evidence(root_path / aux_metrics_path, aux_metrics_path) if aux_metrics_path else None

    evidence_files = []
    for rel_path in config.get("required_evidence_files", []):
        rel = str(rel_path)
        evidence = _file_evidence(root_path / rel, rel)
        evidence_files.append(evidence)
        if not evidence["exists"]:
            old_evidence_findings.append({"severity": "error", "code": "missing_required_evidence_file", "path": rel})

    old_audits = []
    for audit in config.get("old_completion_audits", []):
        rel = str(audit["path"])
        expected_status = str(audit.get("expected_status", "pass"))
        payload = _load_json(root_path / rel)
        evidence = _file_evidence(root_path / rel, rel)
        actual_status = payload.get("status") if isinstance(payload, dict) else None
        row = {"id": str(audit.get("id", rel)), "path": rel, "expected_status": expected_status, "actual_status": actual_status, "artifact": evidence}
        old_audits.append(row)
        if actual_status != expected_status:
            old_evidence_findings.append(
                {
                    "severity": "error",
                    "code": "old_completion_audit_status_mismatch",
                    "id": row["id"],
                    "path": rel,
                    "expected": expected_status,
                    "actual": actual_status,
                }
            )

    old_goal_ids = [str(goal_id) for goal_id in config.get("old_goal_ids", [])]
    archive_glob = config.get("old_ultragoal_archive_glob")
    archive_goals_path, old_goal_statuses = _select_old_archive_goals_path(
        root_path,
        str(archive_glob) if archive_glob else None,
        old_goal_ids,
    )
    for goal_id in old_goal_ids:
        actual = old_goal_statuses.get(goal_id)
        if actual != "complete":
            old_evidence_findings.append({"severity": "error", "code": "old_goal_not_complete_in_archive", "goal_id": goal_id, "actual": actual})

    hard_gates = list(config.get("hard_gates", []))
    aggregate_gate_rows: list[dict[str, Any]] = []
    for gate in hard_gates:
        metric_path = gate.get("metric_path", gate.get("name"))
        actual = _as_float(_get_nested(candidate_metrics, metric_path))
        row = _check_threshold(actual, dict(gate), source="aggregate")
        aggregate_gate_rows.append(row)
        if row["status"] != "pass":
            gate_failures.append(row)

    split_gate_rows: list[dict[str, Any]] = []
    model_name = str(config.get("model_name", "streaming_fdm_d2e_full_compact"))
    required_splits = {str(value) for value in config.get("required_splits", [])}
    split_outputs = _split_outputs(root_path, paths.get("split_stats_summary"))
    split_output_by_name = {str(row.get("split")): row for row in split_outputs}
    if required_splits and set(split_output_by_name) != required_splits:
        missing = sorted(required_splits - set(split_output_by_name))
        if missing:
            findings.append({"severity": "error", "code": "missing_required_split_outputs", "missing": missing})
    for split in sorted(required_splits or set(split_output_by_name)):
        split_output = split_output_by_name.get(split)
        if not split_output:
            continue
        rel_path = str(split_output.get("path"))
        rows = _comparison_rows(root_path / rel_path, split, model_name)
        for gate in hard_gates:
            endpoint = str(gate.get("endpoint", gate.get("name")))
            if not bool(gate.get("require_all_splits", True)):
                continue
            comparison = rows.get(endpoint)
            actual = _as_float((comparison or {}).get("candidate_value"))
            source = f"split:{split}"
            row = _check_threshold(actual, dict(gate), source=source)
            row["endpoint"] = endpoint
            row["split"] = split
            row["split_comparison_path"] = rel_path
            row["comparison_status"] = (comparison or {}).get("status")
            split_gate_rows.append(row)
            if row["status"] != "pass":
                gate_failures.append(row)

    report_metrics = {}
    for item in config.get("report_metrics", []):
        name = str(item["name"])
        report_metrics[name] = _get_nested(candidate_metrics, item.get("metric_path", name))

    aux_delta_report = None
    if aux_metrics is not None and candidate_metrics is not None:
        aux_delta_report = {}
        for gate in hard_gates:
            name = str(gate["name"])
            metric_path = gate.get("metric_path", name)
            base = _as_float(_get_nested(candidate_metrics, metric_path))
            aux = _as_float(_get_nested(aux_metrics, metric_path))
            aux_delta_report[name] = {
                "d2e_only": base,
                "d2e_aux": aux,
                "delta": None if base is None or aux is None else aux - base,
            }

    gate_status = "pass" if not gate_failures and not findings else "fail"
    expected_gate_status = config.get("expected_gate_status")
    status_findings = list(findings) + list(old_evidence_findings)
    if expected_gate_status is not None and gate_status != str(expected_gate_status):
        status_findings.append(
            {
                "severity": "error",
                "code": "gate_status_expectation_mismatch",
                "expected_gate_status": str(expected_gate_status),
                "actual_gate_status": gate_status,
            }
        )
    status = "pass" if not status_findings and (expected_gate_status is None or gate_status == str(expected_gate_status)) else "fail"
    if expected_gate_status is None and gate_status == "fail":
        status = "fail"

    return {
        "schema": "renewed_metric_gate_audit.v1",
        "status": status,
        "error_count": sum(1 for item in status_findings if item.get("severity") == "error"),
        "gate_status": gate_status,
        "expected_gate_status": expected_gate_status,
        "gate_error_count": len(gate_failures),
        "gate_failures": gate_failures,
        "findings": status_findings,
        "target_context": config.get("target_context", {}),
        "candidate_model": config.get("candidate_model", "d2e_only_full_fdm_current"),
        "model_name": model_name,
        "candidate_metrics_artifact": candidate_artifact,
        "aux_metrics_artifact": aux_artifact,
        "aggregate_gates": aggregate_gate_rows,
        "split_gates": split_gate_rows,
        "report_metrics": report_metrics,
        "aux_delta_report": aux_delta_report,
        "old_evidence": {
            "required_files": evidence_files,
            "old_completion_audits": old_audits,
            "old_ultragoal_archive_goals_path": archive_goals_path,
            "old_goal_statuses": old_goal_statuses,
        },
    }


def write_renewed_metric_gate_audit(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = validate_renewed_metric_gates(config, root=root)
    output = output_path or config.get("output_path")
    if not output:
        raise ValueError("output_path is required")
    write_json(Path(root) / output, payload)
    return payload
