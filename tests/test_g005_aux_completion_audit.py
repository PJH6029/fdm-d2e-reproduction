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
    write_json(
        root / cfg["paths"]["ablation_summary"],
        {
            "status": "pass",
            "same_d2e_eval_manifests": True,
            "no_aux_in_d2e_heldout": True,
            "d2e_only_baseline_present": True,
            "d2e_aux_candidate_present": True,
            "claim_boundary": {"d2e_only_separately_reported": True},
            "split_results": [{"split": split, "status": "pass"} for split in SPLITS],
        },
    )
    write_json(
        root / cfg["paths"]["checkpoint_metadata"],
        {
            "source_namespace": "d2e_aux",
            "aux_sources": ["aux_a"],
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


def test_g005_aux_completion_audit_fails_on_prereq_leakage_and_counts(tmp_path: Path):
    _complete_fixture(tmp_path)
    write_json(tmp_path / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "pending"}, {"id": "G005", "status": "pending"}]})
    ablation_path = tmp_path / _config()["paths"]["ablation_summary"]
    import json

    payload = json.loads(ablation_path.read_text())
    payload["no_aux_in_d2e_heldout"] = False
    write_json(ablation_path, payload)
    _write_jsonl(tmp_path / _config()["paths"]["predictions"], 1)
    result = validate_g005_aux_completion(_config(), root=tmp_path)
    codes = {item["code"] for item in result["findings"]}
    assert result["status"] == "fail"
    assert "goal_not_checkpointed_complete" in codes
    assert "prerequisite_goal_not_complete" in codes
    assert "json_expectation_mismatch" in codes
    assert "predictions_count_mismatch" in codes
