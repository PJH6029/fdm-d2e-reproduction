import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_manifest() -> dict:
    return json.loads(Path("artifacts/sources/aux_game_action_dataset_candidates.json").read_text())


def test_aux_plan_preserves_d2e_only_gate_and_user_decision():
    payload = load_manifest()
    assert payload["schema"] == "aux_game_action_dataset_candidates.v1"
    assert payload["user_decision"]["d2e_aux_may_be_primary"] is True
    assert payload["claim_boundary"]["no_d2e_aux_claim_before_d2e_only_gates"] is True
    assert payload["claim_boundary"]["no_training_started_by_this_artifact"] is True
    doc = Path("docs/auxiliary_data_plan.md").read_text()
    assert "D2E-only hard gates remain mandatory" in doc
    assert "This artifact is source-selection/storage evidence only" in doc


def test_aux_selected_candidates_fit_storage_and_have_usable_license_status():
    payload = load_manifest()
    storage = payload["storage_policy"]
    assert storage["fits_cap_with_selected_candidates"] is True
    assert storage["fits_cap_if_high_value_review_passes"] is True
    assert storage["selected_plus_d2e_gib"] < storage["cap_gib"]

    selected = [c for c in payload["candidates"] if c["selection_status"] == "selected_candidate"]
    assert {c["id"] for c in selected} == {
        "minerl_2019_zenodo_v2",
        "atari_head_zenodo_v4",
        "p_doom_atari_breakout_hf",
    }
    for candidate in selected:
        assert candidate["valid_for_training_now"] is True
        assert "review_required" not in candidate["license_id"]
        assert candidate["size_gib"] > 0
        assert candidate["source_url"].startswith("https://")
        assert candidate["source_evidence"]


def test_high_value_vpt_candidate_requires_license_review_before_selection():
    payload = load_manifest()
    [vpt] = [c for c in payload["candidates"] if c["id"] == "openai_vpt_basalt_2022"]
    assert vpt["selection_status"] == "high_value_review_required_not_selected"
    assert vpt["valid_for_training_now"] is False
    assert vpt["license_id"] == "review_required"
    assert vpt["size_gib"] > 500


def test_aux_plan_builder_is_deterministic_for_core_manifest_fields():
    current = load_manifest()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "aux.json"
        doc = Path(tmp) / "aux.md"
        subprocess.run(
            [sys.executable, "scripts/build_aux_dataset_plan.py", "--output", str(out), "--doc-output", str(doc)],
            check=True,
        )
        rebuilt = json.loads(out.read_text())
        assert rebuilt == current
        assert "D2E+aux may become the best/primary final model" in doc.read_text()
