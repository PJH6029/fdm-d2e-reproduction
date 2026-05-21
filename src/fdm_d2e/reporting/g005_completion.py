from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_status(path: Path, rel_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": rel_path, "exists": False, "bytes": 0, "sha256": None}
    return {"path": rel_path, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _jsonl_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                item_id = item.get("id") or item.get("source_id") or item.get("dataset_id")
                if item_id is not None:
                    result.add(str(item_id))
            else:
                result.add(str(item))
        return result
    return {str(value)}


def _get(data: dict[str, Any] | None, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _goal_statuses(root: Path, goals_path: str) -> dict[str, str]:
    payload = _load_json(root / goals_path) or {}
    return {str(goal.get("id")): str(goal.get("status")) for goal in payload.get("goals", [])}


def _assert_json_expectations(
    payload: dict[str, Any] | None,
    expectations: dict[str, Any],
    *,
    source_name: str,
    findings: list[dict[str, Any]],
) -> None:
    for dotted, expected in expectations.items():
        actual = _get(payload, dotted)
        if actual != expected:
            findings.append(
                {
                    "severity": "error",
                    "code": "json_expectation_mismatch",
                    "source": source_name,
                    "json_path": dotted,
                    "expected": expected,
                    "actual": actual,
                }
            )


def _selected_aux_candidate_ids(aux_candidates: dict[str, Any] | None) -> set[str]:
    if aux_candidates is None:
        return set()
    return {
        str(row.get("id"))
        for row in aux_candidates.get("candidates", []) or []
        if isinstance(row, dict) and row.get("selection_status") == "selected_candidate" and row.get("id") is not None
    }


def _namespace_source_rows(namespace_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if namespace_manifest is None:
        return []
    rows = namespace_manifest.get("aux_sources")
    if rows is None:
        rows = namespace_manifest.get("sources")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _validate_namespace_manifest(
    namespace_manifest: dict[str, Any] | None,
    *,
    selected_aux_ids: set[str],
    required_splits: set[str],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the G005 source namespace/provenance contract.

    This deliberately checks more than artifact existence: D2E+aux is allowed
    to become the best model only if auxiliary sources remain in source-specific
    namespaces, keep action heads separate, and prove the D2E eval manifests are
    byte-identical to the D2E-only eval manifests used by G003/G004.
    """
    if namespace_manifest is None:
        return {"aux_source_ids": [], "d2e_eval_splits": [], "selected_aux_ids": sorted(selected_aux_ids)}

    source_rows = _namespace_source_rows(namespace_manifest)
    source_ids = {str(row.get("id")) for row in source_rows if row.get("id") is not None}
    missing_selected = sorted(selected_aux_ids - source_ids)
    if missing_selected:
        findings.append({"severity": "error", "code": "namespace_missing_selected_aux_sources", "missing": missing_selected})
    unselected = sorted(source_ids - selected_aux_ids)
    if unselected:
        findings.append({"severity": "error", "code": "namespace_contains_unselected_aux_sources", "unselected": unselected})

    for row in source_rows:
        source_id = str(row.get("id") or "")
        missing_fields = [field for field in ("id", "namespace", "source_url", "license_id", "provenance_sha256") if not row.get(field)]
        if missing_fields:
            findings.append(
                {
                    "severity": "error",
                    "code": "namespace_source_missing_required_fields",
                    "source_id": source_id,
                    "missing": missing_fields,
                }
            )
        namespace = str(row.get("namespace") or "")
        expected_prefix = f"outputs/aux/{source_id}/"
        if source_id and not namespace.startswith(expected_prefix):
            findings.append(
                {
                    "severity": "error",
                    "code": "namespace_source_path_mismatch",
                    "source_id": source_id,
                    "expected_prefix": expected_prefix,
                    "actual": namespace,
                }
            )
        action_head = row.get("action_head")
        if not isinstance(action_head, dict) or not action_head.get("type") or not action_head.get("namespace"):
            findings.append({"severity": "error", "code": "namespace_source_missing_action_head", "source_id": source_id})
        elif source_id and str(action_head.get("namespace")) != source_id:
            findings.append(
                {
                    "severity": "error",
                    "code": "namespace_action_head_namespace_mismatch",
                    "source_id": source_id,
                    "actual": action_head.get("namespace"),
                }
            )
        overlap_count = row.get("d2e_heldout_overlap_count", 0)
        overlap_ids = row.get("d2e_heldout_overlap_recording_ids") or []
        try:
            overlap_count_int = int(overlap_count)
        except (TypeError, ValueError):
            overlap_count_int = -1
        if overlap_count_int != 0 or overlap_ids:
            findings.append(
                {
                    "severity": "error",
                    "code": "namespace_aux_overlap_with_d2e_heldout",
                    "source_id": source_id,
                    "overlap_count": overlap_count,
                    "overlap_recording_ids": overlap_ids,
                }
            )

    eval_splits_payload = _get(namespace_manifest, "d2e_eval_manifests.splits")
    eval_splits: dict[str, Any]
    if isinstance(eval_splits_payload, dict):
        eval_splits = eval_splits_payload
    elif isinstance(eval_splits_payload, list):
        eval_splits = {str(row.get("split")): row for row in eval_splits_payload if isinstance(row, dict) and row.get("split") is not None}
    else:
        eval_splits = {}
    missing_eval_splits = sorted(required_splits - set(eval_splits))
    if missing_eval_splits:
        findings.append({"severity": "error", "code": "namespace_missing_required_eval_splits", "missing": missing_eval_splits})
    for split in sorted(required_splits & set(eval_splits)):
        row = eval_splits[split]
        if not isinstance(row, dict):
            findings.append({"severity": "error", "code": "namespace_eval_split_malformed", "split": split})
            continue
        if row.get("same_hash") is not True:
            findings.append({"severity": "error", "code": "namespace_eval_split_hash_not_equal", "split": split, "actual": row.get("same_hash")})
        missing_hash_fields = [
            field
            for field in ("d2e_only_manifest_sha256", "d2e_aux_manifest_sha256")
            if not row.get(field)
        ]
        if missing_hash_fields:
            findings.append(
                {
                    "severity": "error",
                    "code": "namespace_eval_split_missing_hashes",
                    "split": split,
                    "missing": missing_hash_fields,
                }
            )
    return {
        "aux_source_ids": sorted(source_ids),
        "selected_aux_ids": sorted(selected_aux_ids),
        "d2e_eval_splits": sorted(eval_splits),
    }


def _validate_aux_ablation_details(
    ablation: dict[str, Any] | None,
    *,
    required_splits: set[str],
    namespace_manifest: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> None:
    if ablation is None:
        return
    eval_splits_payload = _get(namespace_manifest, "d2e_eval_manifests.splits") if namespace_manifest is not None else {}
    namespace_hash_by_split: dict[str, str] = {}
    if isinstance(eval_splits_payload, dict):
        for split, row in eval_splits_payload.items():
            if isinstance(row, dict) and row.get("d2e_aux_manifest_sha256"):
                namespace_hash_by_split[str(split)] = str(row["d2e_aux_manifest_sha256"])
    elif isinstance(eval_splits_payload, list):
        for row in eval_splits_payload:
            if isinstance(row, dict) and row.get("split") is not None and row.get("d2e_aux_manifest_sha256"):
                namespace_hash_by_split[str(row["split"])] = str(row["d2e_aux_manifest_sha256"])

    by_split = {
        str(item.get("split")): item
        for item in ablation.get("split_results", []) or []
        if isinstance(item, dict) and item.get("split") is not None
    }
    for split in sorted(required_splits):
        item = by_split.get(split)
        if item is None:
            continue
        missing_run_fields = [field for field in ("d2e_only_run_id", "d2e_aux_run_id") if not item.get(field)]
        if missing_run_fields:
            findings.append({"severity": "error", "code": "ablation_split_missing_run_ids", "split": split, "missing": missing_run_fields})
        if item.get("same_d2e_eval_manifest") is not True:
            findings.append(
                {
                    "severity": "error",
                    "code": "ablation_split_not_same_d2e_eval_manifest",
                    "split": split,
                    "actual": item.get("same_d2e_eval_manifest"),
                }
            )
        expected_hash = namespace_hash_by_split.get(split)
        actual_hash = item.get("d2e_eval_manifest_sha256")
        if expected_hash and actual_hash != expected_hash:
            findings.append(
                {
                    "severity": "error",
                    "code": "ablation_split_eval_manifest_hash_mismatch",
                    "split": split,
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )


def _validate_action_registry(
    action_registry: dict[str, Any] | None,
    *,
    selected_aux_ids: set[str],
    namespace_manifest: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if action_registry is None:
        return {"action_head_ids": [], "selected_aux_ids": sorted(selected_aux_ids)}
    heads = action_registry.get("action_heads", [])
    if not isinstance(heads, list):
        findings.append({"severity": "error", "code": "action_registry_heads_malformed"})
        heads = []
    if action_registry.get("source_specific_action_heads") is not True:
        findings.append({"severity": "error", "code": "action_registry_not_source_specific", "actual": action_registry.get("source_specific_action_heads")})
    if action_registry.get("no_cross_source_action_collapse") is not True:
        findings.append({"severity": "error", "code": "action_registry_allows_cross_source_collapse", "actual": action_registry.get("no_cross_source_action_collapse")})
    if _get(action_registry, "d2e_endpoint_claim_boundary.no_aux_source_directly_claims_d2e_keyboard_mouse") is not True:
        findings.append({"severity": "error", "code": "action_registry_missing_d2e_claim_boundary"})
    by_id = {str(row.get("id")): row for row in heads if isinstance(row, dict) and row.get("id") is not None}
    missing = sorted(selected_aux_ids - set(by_id))
    extra = sorted(set(by_id) - selected_aux_ids)
    if missing:
        findings.append({"severity": "error", "code": "action_registry_missing_selected_aux_sources", "missing": missing})
    if extra:
        findings.append({"severity": "error", "code": "action_registry_contains_unselected_aux_sources", "extra": extra})

    namespace_heads = {
        str(row.get("id")): row.get("action_head")
        for row in _namespace_source_rows(namespace_manifest)
        if row.get("id") is not None and isinstance(row.get("action_head"), dict)
    }
    for source_id, row in sorted(by_id.items()):
        if str(row.get("namespace") or "") != source_id:
            findings.append({"severity": "error", "code": "action_registry_namespace_mismatch", "source_id": source_id, "actual": row.get("namespace")})
        if not row.get("type"):
            findings.append({"severity": "error", "code": "action_registry_missing_type", "source_id": source_id})
        if row.get("d2e_endpoint_claims_allowed") not in ([], None):
            findings.append({"severity": "error", "code": "action_registry_aux_allows_d2e_endpoint_claims", "source_id": source_id, "actual": row.get("d2e_endpoint_claims_allowed")})
        namespace_head = namespace_heads.get(source_id)
        if isinstance(namespace_head, dict) and namespace_head.get("type") and row.get("type") and namespace_head.get("type") != row.get("type"):
            findings.append(
                {
                    "severity": "error",
                    "code": "action_registry_namespace_type_mismatch",
                    "source_id": source_id,
                    "registry_type": row.get("type"),
                    "namespace_type": namespace_head.get("type"),
                }
            )
    return {"action_head_ids": sorted(by_id), "selected_aux_ids": sorted(selected_aux_ids)}


def _validate_aux_examples(
    aux_examples: dict[str, Any] | None,
    *,
    selected_aux_ids: set[str],
    required_splits: set[str],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if aux_examples is None:
        return {"source_ids": [], "selected_aux_ids": sorted(selected_aux_ids), "required_splits": sorted(required_splits)}
    if aux_examples.get("status") != "pass":
        findings.append(
            {
                "severity": "error",
                "code": "aux_examples_not_pass",
                "status": aux_examples.get("status"),
                "error_count": aux_examples.get("error_count"),
            }
        )
    rows = aux_examples.get("sources", [])
    if not isinstance(rows, list):
        findings.append({"severity": "error", "code": "aux_examples_sources_malformed"})
        rows = []
    by_id = {str(row.get("source_id")): row for row in rows if isinstance(row, dict) and row.get("source_id") is not None}
    missing = sorted(selected_aux_ids - set(by_id))
    extra = sorted(set(by_id) - selected_aux_ids)
    if missing:
        findings.append({"severity": "error", "code": "aux_examples_missing_selected_sources", "missing": missing})
    if extra:
        findings.append({"severity": "error", "code": "aux_examples_contains_unselected_sources", "extra": extra})
    for source_id, row in sorted(by_id.items()):
        if row.get("status") != "pass":
            findings.append({"severity": "error", "code": "aux_examples_source_not_pass", "source_id": source_id, "status": row.get("status")})
        split_counts = row.get("split_counts")
        if not isinstance(split_counts, dict):
            findings.append({"severity": "error", "code": "aux_examples_split_counts_malformed", "source_id": source_id})
            split_counts = {}
        split_files = row.get("split_files") if isinstance(row.get("split_files"), dict) else {}
        for split in sorted(required_splits):
            count = int(split_counts.get(split) or 0)
            if count <= 0:
                findings.append({"severity": "error", "code": "aux_examples_split_empty", "source_id": source_id, "split": split})
            file_row = split_files.get(split) if isinstance(split_files, dict) else None
            if not isinstance(file_row, dict) or file_row.get("exists") is not True or not file_row.get("sha256"):
                findings.append({"severity": "error", "code": "aux_examples_split_file_missing", "source_id": source_id, "split": split})
            elif int(file_row.get("rows") or 0) != count:
                findings.append(
                    {
                        "severity": "error",
                        "code": "aux_examples_split_file_count_mismatch",
                        "source_id": source_id,
                        "split": split,
                        "expected": count,
                        "actual": file_row.get("rows"),
                    }
                )
    return {
        "source_ids": sorted(by_id),
        "selected_aux_ids": sorted(selected_aux_ids),
        "required_splits": sorted(required_splits),
        "total_examples": aux_examples.get("total_examples"),
    }


def validate_g005_aux_completion(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    findings: list[dict[str, Any]] = []
    goals_path = str(config.get("goals_path", ".omx/ultragoal/goals.json"))
    goal_id = str(config.get("goal_id", "G005-aux-data-best-model"))
    statuses = _goal_statuses(root_path, goals_path)
    goal_status = statuses.get(goal_id, "missing")
    require_goal_checkpoint = bool(config.get("require_goal_checkpoint_complete", True))
    if require_goal_checkpoint and goal_status != "complete":
        findings.append({"severity": "error", "code": "goal_not_checkpointed_complete", "goal_id": goal_id, "actual": goal_status})
    prereq_report = {}
    for prereq in config.get("prerequisite_goals", []):
        actual = statuses.get(str(prereq), "missing")
        prereq_report[str(prereq)] = actual
        if actual != "complete":
            findings.append({"severity": "error", "code": "prerequisite_goal_not_complete", "goal_id": str(prereq), "actual": actual})

    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    artifacts = {key: _file_status(root_path / rel_path, rel_path) for key, rel_path in paths.items()}
    for key, evidence in artifacts.items():
        if not evidence["exists"]:
            findings.append({"severity": "error", "code": "missing_required_artifact", "artifact_key": key, "path": evidence["path"]})

    aux_candidates = _load_json(root_path / paths.get("aux_candidates", "")) if paths.get("aux_candidates") else None
    namespace_manifest = _load_json(root_path / paths.get("namespace_manifest", "")) if paths.get("namespace_manifest") else None
    action_registry = _load_json(root_path / paths.get("action_registry", "")) if paths.get("action_registry") else None
    runtime_env = _load_json(root_path / paths.get("runtime_env", "")) if paths.get("runtime_env") else None
    aux_examples = _load_json(root_path / paths.get("aux_examples_summary", "")) if paths.get("aux_examples_summary") else None
    ablation = _load_json(root_path / paths.get("ablation_summary", "")) if paths.get("ablation_summary") else None
    metadata = _load_json(root_path / paths.get("checkpoint_metadata", "")) if paths.get("checkpoint_metadata") else None
    run_summary = _load_json(root_path / paths.get("run_summary", "")) if paths.get("run_summary") else None

    _assert_json_expectations(aux_candidates, dict(config.get("aux_candidate_expectations", {})), source_name="aux_candidates", findings=findings)
    _assert_json_expectations(namespace_manifest, dict(config.get("namespace_manifest_expectations", {})), source_name="namespace_manifest", findings=findings)
    _assert_json_expectations(action_registry, dict(config.get("action_registry_expectations", {})), source_name="action_registry", findings=findings)
    _assert_json_expectations(runtime_env, dict(config.get("runtime_env_expectations", {})), source_name="runtime_env", findings=findings)
    _assert_json_expectations(ablation, dict(config.get("ablation_expectations", {})), source_name="ablation_summary", findings=findings)
    _assert_json_expectations(metadata, dict(config.get("metadata_expectations", {})), source_name="checkpoint_metadata", findings=findings)

    selected_aux_ids = _selected_aux_candidate_ids(aux_candidates)
    if aux_candidates is not None:
        selected = [row for row in aux_candidates.get("candidates", []) if row.get("selection_status") == "selected_candidate"]
        if not selected:
            findings.append({"severity": "error", "code": "no_selected_aux_candidates"})
        selected_total = float(_get(aux_candidates, "storage_policy.selected_plus_d2e_gib") or 0.0)
        cap = float(_get(aux_candidates, "storage_policy.cap_gib") or 0.0)
        if selected_total and cap and selected_total > cap:
            findings.append({"severity": "error", "code": "aux_storage_over_cap", "selected_plus_d2e_gib": selected_total, "cap_gib": cap})

    required_splits = set(config.get("required_splits", []))
    namespace_report = _validate_namespace_manifest(
        namespace_manifest,
        selected_aux_ids=selected_aux_ids,
        required_splits=required_splits,
        findings=findings,
    )
    action_registry_report = _validate_action_registry(
        action_registry,
        selected_aux_ids=selected_aux_ids,
        namespace_manifest=namespace_manifest,
        findings=findings,
    )
    aux_example_report = _validate_aux_examples(
        aux_examples,
        selected_aux_ids=selected_aux_ids,
        required_splits=set(config.get("required_aux_example_splits", ["train", "val", "test"])),
        findings=findings,
    )
    ablation_splits: set[str] = set()
    if ablation is not None:
        for item in ablation.get("split_results", []) or []:
            if isinstance(item, dict) and item.get("split") is not None:
                ablation_splits.add(str(item.get("split")))
        missing = sorted(required_splits - ablation_splits)
        if missing:
            findings.append({"severity": "error", "code": "ablation_missing_required_splits", "missing": missing})
        if bool(config.get("require_d2e_only_baseline", True)) and not bool(ablation.get("d2e_only_baseline_present", False)):
            findings.append({"severity": "error", "code": "ablation_missing_d2e_only_baseline"})
        if bool(config.get("require_d2e_aux_candidate", True)) and not bool(ablation.get("d2e_aux_candidate_present", False)):
            findings.append({"severity": "error", "code": "ablation_missing_d2e_aux_candidate"})
    _validate_aux_ablation_details(ablation, required_splits=required_splits, namespace_manifest=namespace_manifest, findings=findings)

    if metadata is not None:
        aux_sources = _as_string_set(metadata.get("aux_sources") or metadata.get("source_aux_datasets") or [])
        if not aux_sources:
            findings.append({"severity": "error", "code": "metadata_missing_aux_sources"})
        if selected_aux_ids and aux_sources and aux_sources != selected_aux_ids:
            findings.append(
                {
                    "severity": "error",
                    "code": "metadata_aux_sources_mismatch",
                    "expected": sorted(selected_aux_ids),
                    "actual": sorted(aux_sources),
                }
            )
        required_tags = set(config.get("required_target_eval_split_tags", []))
        actual_tags = set(str(tag) for tag in metadata.get("target_eval_split_tags", []) or [])
        missing_tags = sorted(required_tags - actual_tags)
        if missing_tags:
            findings.append({"severity": "error", "code": "metadata_missing_target_eval_split_tags", "missing": missing_tags})

    if run_summary is not None:
        if run_summary.get("exit_code") != 0:
            findings.append({"severity": "error", "code": "run_summary_exit_nonzero", "actual": run_summary.get("exit_code")})
        expected_gpus = int(config.get("expected_gpus", 4))
        if int(run_summary.get("expected_gpus", -1)) != expected_gpus:
            findings.append({"severity": "error", "code": "run_summary_expected_gpus_mismatch", "expected": expected_gpus, "actual": run_summary.get("expected_gpus")})

    target_count = _jsonl_count(root_path / paths.get("target_records", "")) if paths.get("target_records") else None
    pred_count = _jsonl_count(root_path / paths.get("predictions", "")) if paths.get("predictions") else None
    count_report = {"target_records": target_count, "predictions": pred_count}
    if target_count is not None and pred_count is not None and pred_count != target_count:
        findings.append({"severity": "error", "code": "predictions_count_mismatch", "expected": target_count, "actual": pred_count})

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "g005_aux_completion_audit.v1",
        "status": "pass" if not errors else "fail",
        "goal_id": goal_id,
        "goal_status": goal_status,
        "require_goal_checkpoint_complete": require_goal_checkpoint,
        "prerequisite_goal_statuses": prereq_report,
        "required_splits": sorted(required_splits),
        "ablation_splits": sorted(ablation_splits),
        "namespace_report": namespace_report,
        "action_registry_report": action_registry_report,
        "aux_example_report": aux_example_report,
        "artifacts": artifacts,
        "counts": count_report,
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "This audit is required before checkpointing G005 complete; it proves D2E-only prerequisites, aux provenance/storage policy, separated namespaces, D2E-only vs D2E+aux ablations, target split tags, and run evidence.",
    }


def write_g005_aux_completion_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_g005_aux_completion(config, root=root)
    out = output_path or config.get("output_path")
    if out:
        write_json(Path(root) / str(out), payload)
    return payload
