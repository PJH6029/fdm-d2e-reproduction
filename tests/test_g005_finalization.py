from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from finalize_g005_aux_best_model import finalize


SPLITS = ["temporal", "heldout_recording", "heldout_game"]


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "summary_out": "artifacts/aux/finalize.json",
        "allow_fail": False,
        "g005_completion_config": "configs/eval/g005_completion.json",
        "g005_audit_output": "artifacts/aux/g005_audit.json",
        "run_summary": "artifacts/aux/run.json",
        "namespace_manifest_output": "artifacts/aux/namespace.json",
        "aux_candidates": "artifacts/sources/aux.json",
        "source_evidence": [],
        "eval_manifest_hashes": None,
        "completion_ready": False,
        "allow_template_namespace": False,
        "skip_namespace_build": False,
        "force_namespace": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _write_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"x": %d}\n' % i for i in range(n)))


def _write_fixture(root: Path) -> None:
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": "complete"}, {"id": "G004", "status": "complete"}, {"id": "G005", "status": "pending"}]})
    write_json(
        root / "artifacts/sources/aux.json",
        {
            "user_decision": {"d2e_aux_may_be_primary": True},
            "claim_boundary": {"no_d2e_aux_claim_before_d2e_only_gates": True},
            "storage_policy": {"fits_cap_with_selected_candidates": True, "selected_plus_d2e_gib": 100.0, "cap_gib": 5120.0},
            "candidates": [
                {
                    "id": "aux_a",
                    "selection_status": "selected_candidate",
                    "source_url": "https://example.invalid/aux-a",
                    "license_id": "mit",
                    "domain": "Minecraft human demonstrations",
                }
            ],
        },
    )
    doc = root / "docs/aux.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("aux plan")
    write_json(
        root / "artifacts/aux/source_evidence.json",
        {
            "aux_sources": [
                {
                    "id": "aux_a",
                    "namespace": "outputs/aux/aux_a/train/",
                    "source_url": "https://example.invalid/aux-a",
                    "license_id": "mit",
                    "provenance_sha256": "abc123",
                    "action_head": {"type": "minecraft_keyboard_mouse", "namespace": "aux_a"},
                    "d2e_heldout_overlap_count": 0,
                    "d2e_heldout_overlap_recording_ids": [],
                    "split_hashes": {"train": {"sha256": "train-hash"}, "val": {"sha256": "val-hash"}, "test": {"sha256": "test-hash"}},
                }
            ]
        },
    )
    hashes = {split: {"sha256": f"{split}-hash"} for split in SPLITS}
    write_json(root / "artifacts/aux/eval_hashes.json", hashes)
    write_json(
        root / "artifacts/aux/ablation.json",
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
        root / "outputs/fdm_aux/best/checkpoint_metadata.json",
        {
            "source_namespace": "d2e_aux",
            "aux_sources": [{"id": "aux_a", "namespace": "outputs/aux/aux_a/train/"}],
            "d2e_eval_split_contract": {"exists": True},
            "data_universe": {"exists": True},
            "claim_boundary": {"no_aux_in_d2e_heldout": True},
            "target_eval_split_tags": SPLITS,
        },
    )
    for rel in ["outputs/fdm_aux/best/resolved_config.json", "outputs/fdm_aux/best/metrics.json", "outputs/fdm_aux/best/stats.json"]:
        write_json(root / rel, {"status": "ok"})
    checkpoint = root / "outputs/fdm_aux/best/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    _write_jsonl(root / "outputs/fdm_aux/best/target.jsonl", 2)
    _write_jsonl(root / "outputs/fdm_aux/best/predictions.jsonl", 2)
    write_json(root / "artifacts/aux/run.json", {"exit_code": 0, "expected_gpus": 4})
    write_json(
        root / "configs/eval/g005_completion.json",
        {
            "goals_path": ".omx/ultragoal/goals.json",
            "goal_id": "G005",
            "prerequisite_goals": ["G003", "G004"],
            "require_goal_checkpoint_complete": False,
            "expected_gpus": 4,
            "required_splits": SPLITS,
            "required_target_eval_split_tags": SPLITS,
            "paths": {
                "aux_candidates": "artifacts/sources/aux.json",
                "aux_plan_doc": "docs/aux.md",
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
            },
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
        },
    )


def test_finalize_reports_missing_run_and_namespace_inputs(tmp_path: Path):
    write_json(tmp_path / "artifacts/sources/aux.json", {"candidates": [{"id": "aux_a", "selection_status": "selected_candidate"}]})
    write_json(tmp_path / "configs/eval/g005_completion.json", {"paths": {"aux_candidates": "artifacts/sources/aux.json", "run_summary": "artifacts/aux/run.json"}})
    payload = finalize(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "namespace_manifest_not_ready" in codes
    assert "missing_run_summary" in codes
    assert "g005_completion_audit_not_pass" in codes


def test_finalize_builds_namespace_manifest_and_g005_audit(tmp_path: Path):
    _write_fixture(tmp_path)
    payload = finalize(
        _args(
            tmp_path,
            source_evidence=["artifacts/aux/source_evidence.json"],
            eval_manifest_hashes="artifacts/aux/eval_hashes.json",
            completion_ready=True,
        )
    )
    assert payload["status"] == "pass"
    assert payload["namespace_completion_ready"] is True
    assert payload["g005_audit_status"] == "pass"
    assert json.loads((tmp_path / "artifacts/aux/namespace.json").read_text())["completion_ready"] is True
    assert json.loads((tmp_path / "artifacts/aux/g005_audit.json").read_text())["status"] == "pass"
