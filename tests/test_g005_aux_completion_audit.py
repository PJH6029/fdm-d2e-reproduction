from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.g005_completion import validate_g005_aux_completion


SPLITS = ["temporal", "heldout_recording", "heldout_game"]


def _config() -> dict:
    paths = {
        "aux_candidates": "artifacts/sources/aux.json",
        "aux_plan_doc": "docs/aux.md",
        "action_registry": "artifacts/aux/action_registry.json",
        "namespace_manifest": "artifacts/aux/namespace.json",
        "ablation_summary": "artifacts/aux/ablation.json",
        "checkpoint_metadata": "outputs/fdm_aux/best/checkpoint_metadata.json",
        "resolved_config": "outputs/fdm_aux/best/resolved_config.json",
        "checkpoint": "outputs/fdm_aux/best/checkpoint.pt",
        "target_records": "outputs/fdm_aux/best/target.jsonl",
        "predictions": "outputs/fdm_aux/best/predictions.jsonl",
        "metrics": "outputs/fdm_aux/best/metrics.json",
        "statistical_comparison": "outputs/fdm_aux/best/stats.json",
        "run_summary": "artifacts/aux/run.json",
    }
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "goal_id": "G005",
        "prerequisite_goals": ["G003", "G004"],
        "expected_gpus": 4,
        "required_splits": SPLITS,
        "required_target_eval_split_tags": SPLITS,
        "paths": paths,
        "aux_candidate_expectations": {
            "user_decision.d2e_aux_may_be_primary": True,
            "claim_boundary.no_d2e_aux_claim_before_d2e_only_gates": True,
            "storage_policy.fits_cap_with_selected_candidates": True,
        },
        "ablation_expectations": {
            "status": "pass",
            "same_d2e_eval_manifests": True,
            "no_aux_in_d2e_heldout": True,
            "claim_boundary.d2e_only_separately_reported": True,
        },
        "metadata_expectations": {
            "source_namespace": "d2e_aux",
            "d2e_eval_split_contract.exists": True,
            "data_universe.exists": True,
            "claim_boundary.no_aux_in_d2e_heldout": True,
        },
        "namespace_manifest_expectations": {
            "schema": "g005_aux_namespace_manifest.v1",
            "source_namespace": "d2e_aux",
            "completion_ready": True,
            "claim_boundary.no_aux_in_d2e_heldout": True,
            "claim_boundary.no_d2e_aux_claim_before_d2e_only_gates": True,
            "training_policy.source_specific_action_heads": True,
            "d2e_eval_manifests.same_as_d2e_only": True,
        },
        "action_registry_expectations": {
            "schema": "g005_aux_action_registry.v1",
            "status": "pass",
            "source_specific_action_heads": True,
            "no_cross_source_action_collapse": True,
            "d2e_endpoint_claim_boundary.no_aux_source_directly_claims_d2e_keyboard_mouse": True,
        },
    }


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x": %d}\n' % i for i in range(n)))


def _complete_fixture(root: Path) -> None:
    cfg = _config()
    write_json(
        root / ".omx/ultragoal/goals.json",
        {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}, {"id": "G005", "status": "complete"}]},
    )
    write_json(
        root / cfg["paths"]["aux_candidates"],
        {
            "user_decision": {"d2e_aux_may_be_primary": True},
            "claim_boundary": {"no_d2e_aux_claim_before_d2e_only_gates": True},
            "storage_policy": {"fits_cap_with_selected_candidates": True, "selected_plus_d2e_gib": 100.0, "cap_gib": 5120.0},
            "candidates": [{"id": "aux_a", "selection_status": "selected_candidate"}],
        },
    )
    plan = root / cfg["paths"]["aux_plan_doc"]
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("aux plan")
    split_hashes = {
        split: {
            "d2e_only_manifest_sha256": f"{split}-hash",
            "d2e_aux_manifest_sha256": f"{split}-hash",
            "same_hash": True,
        }
        for split in SPLITS
    }
    write_json(
        root / cfg["paths"]["namespace_manifest"],
        {
            "schema": "g005_aux_namespace_manifest.v1",
            "source_namespace": "d2e_aux",
            "completion_ready": True,
            "claim_boundary": {
                "no_aux_in_d2e_heldout": True,
                "no_d2e_aux_claim_before_d2e_only_gates": True,
            },
            "training_policy": {"source_specific_action_heads": True},
            "d2e_eval_manifests": {"same_as_d2e_only": True, "splits": split_hashes},
            "aux_sources": [
                {
                    "id": "aux_a",
                    "namespace": "outputs/aux/aux_a/train/",
                    "source_url": "https://example.invalid/aux-a",
                    "license_id": "cc-by-4.0",
                    "provenance_sha256": "abc123",
                    "action_head": {"type": "discrete", "namespace": "aux_a"},
                    "d2e_heldout_overlap_count": 0,
                    "d2e_heldout_overlap_recording_ids": [],
                }
            ],
        },
    )
    write_json(
        root / cfg["paths"]["action_registry"],
        {
            "schema": "g005_aux_action_registry.v1",
            "status": "pass",
            "selected_aux_source_ids": ["aux_a"],
            "source_specific_action_heads": True,
            "no_cross_source_action_collapse": True,
            "d2e_endpoint_claim_boundary": {"no_aux_source_directly_claims_d2e_keyboard_mouse": True},
            "action_heads": [{"id": "aux_a", "namespace": "aux_a", "type": "discrete", "d2e_endpoint_claims_allowed": []}],
        },
    )
    write_json(
        root / cfg["paths"]["ablation_summary"],
        {
            "status": "pass",
            "same_d2e_eval_manifests": True,
            "no_aux_in_d2e_heldout": True,
            "d2e_only_baseline_present": True,
            "d2e_aux_candidate_present": True,
            "claim_boundary": {"d2e_only_separately_reported": True},
            "split_results": [
                {
                    "split": split,
                    "status": "pass",
                    "d2e_only_run_id": f"d2e-only-{split}",
                    "d2e_aux_run_id": f"d2e-aux-{split}",
                    "same_d2e_eval_manifest": True,
                    "d2e_eval_manifest_sha256": f"{split}-hash",
                }
                for split in SPLITS
            ],
        },
    )
    write_json(
        root / cfg["paths"]["checkpoint_metadata"],
        {
            "source_namespace": "d2e_aux",
            "aux_sources": [{"id": "aux_a", "namespace": "outputs/aux/aux_a/train/"}],
            "d2e_eval_split_contract": {"exists": True},
            "data_universe": {"exists": True},
            "claim_boundary": {"no_aux_in_d2e_heldout": True},
            "target_eval_split_tags": SPLITS,
        },
    )
    for key in ["resolved_config", "metrics", "statistical_comparison"]:
        write_json(root / cfg["paths"][key], {"status": "ok"})
    checkpoint = root / cfg["paths"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    _write_jsonl(root / cfg["paths"]["target_records"], 2)
    _write_jsonl(root / cfg["paths"]["predictions"], 2)
    write_json(root / cfg["paths"]["run_summary"], {"exit_code": 0, "expected_gpus": 4})


def test_g005_aux_completion_audit_passes_on_full_fixture(tmp_path: Path):
    _complete_fixture(tmp_path)
    payload = validate_g005_aux_completion(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert set(payload["ablation_splits"]) == set(SPLITS)
    assert payload["namespace_report"]["aux_source_ids"] == ["aux_a"]


def test_g005_aux_completion_audit_fails_on_prereq_leakage_and_counts(tmp_path: Path):
    _complete_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "pending"}, {"id": "G005", "status": "pending"}]})
    ablation_path = tmp_path / _config()["paths"]["ablation_summary"]
    import json

    payload = json.loads(ablation_path.read_text())
    payload["no_aux_in_d2e_heldout"] = False
    payload["split_results"][0].pop("d2e_aux_run_id")
    write_json(ablation_path, payload)
    namespace_path = tmp_path / _config()["paths"]["namespace_manifest"]
    namespace = json.loads(namespace_path.read_text())
    namespace["aux_sources"][0]["d2e_heldout_overlap_count"] = 1
    namespace["d2e_eval_manifests"]["splits"]["temporal"]["same_hash"] = False
    write_json(namespace_path, namespace)
    _write_jsonl(tmp_path / _config()["paths"]["predictions"], 1)
    result = validate_g005_aux_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in result["findings"]}
    assert result["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "json_expectation_mismatch" in codes
    assert "predictions_count_mismatch" in codes
    assert "ablation_split_missing_run_ids" in codes
    assert "namespace_aux_overlap_with_d2e_heldout" in codes
    assert "namespace_eval_split_hash_not_equal" in codes


def test_g005_aux_completion_audit_rejects_unselected_namespace_and_hash_mismatch(tmp_path: Path):
    _complete_fixture(tmp_path)
    namespace_path = tmp_path / _config()["paths"]["namespace_manifest"]
    import json

    namespace = json.loads(namespace_path.read_text())
    namespace["aux_sources"].append(
        {
            "id": "aux_b",
            "namespace": "outputs/aux/aux_b/train/",
            "source_url": "https://example.invalid/aux-b",
            "license_id": "mit",
            "provenance_sha256": "def456",
            "action_head": {"type": "continuous", "namespace": "wrong_namespace"},
            "d2e_heldout_overlap_count": 0,
            "d2e_heldout_overlap_recording_ids": [],
        }
    )
    namespace["d2e_eval_manifests"]["splits"]["heldout_game"].pop("d2e_aux_manifest_sha256")
    write_json(namespace_path, namespace)
    ablation_path = tmp_path / _config()["paths"]["ablation_summary"]
    ablation = json.loads(ablation_path.read_text())
    for item in ablation["split_results"]:
        if item["split"] == "heldout_recording":
            item["same_d2e_eval_manifest"] = False
        if item["split"] == "temporal":
            item["d2e_eval_manifest_sha256"] = "wrong-hash"
    write_json(ablation_path, ablation)

    result = validate_g005_aux_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in result["findings"]}
    assert result["status"] == "fail"
    assert "namespace_contains_unselected_aux_sources" in codes
    assert "namespace_action_head_namespace_mismatch" in codes
    assert "namespace_eval_split_missing_hashes" in codes
    assert "ablation_split_not_same_d2e_eval_manifest" in codes
    assert "ablation_split_eval_manifest_hash_mismatch" in codes


def test_g005_aux_completion_audit_rejects_collapsed_or_mismatched_action_registry(tmp_path: Path):
    _complete_fixture(tmp_path)
    import json

    registry_path = tmp_path / _config()["paths"]["action_registry"]
    registry = json.loads(registry_path.read_text())
    registry["no_cross_source_action_collapse"] = False
    registry["action_heads"][0]["namespace"] = "shared_aux"
    registry["action_heads"][0]["d2e_endpoint_claims_allowed"] = ["keyboard_accuracy"]
    registry["action_heads"].append({"id": "unselected", "namespace": "unselected", "type": "discrete", "d2e_endpoint_claims_allowed": []})
    write_json(registry_path, registry)

    result = validate_g005_aux_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in result["findings"]}
    assert result["status"] == "fail"
    assert "action_registry_allows_cross_source_collapse" in codes
    assert "action_registry_namespace_mismatch" in codes
    assert "action_registry_aux_allows_d2e_endpoint_claims" in codes
    assert "action_registry_contains_unselected_aux_sources" in codes
