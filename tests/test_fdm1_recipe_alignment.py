from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.fdm1_recipe_alignment import (
    build_fdm1_public_recipe_manifest,
    validate_fdm1_recipe_alignment,
    write_fdm1_public_recipe_manifest,
)


def _idm_candidate() -> dict:
    return {
        "fdm1_recipe": {
            "stage": "idm",
            "video_encoder": {"enabled": True, "input": "D2E screen-video frame windows"},
            "idm_objective": {
                "type": "masked_diffusion_action_tokens",
                "conditioning": "all_window_frames_plus_masked_action_tokens",
                "noncausal_frame_conditioning": True,
            },
            "inference_schedule": {"type": "iterative_unmasking", "steps": 16},
            "action_tokenization": {
                "keyboard_press_release_tokens": True,
                "mouse_delta_bins_per_axis": 49,
            },
            "fidelity": {
                "public_vs_inferred_recorded": True,
                "claims_fdm1_parity": False,
            },
        }
    }


def _fdm_candidate() -> dict:
    return {
        "fdm1_recipe": {
            "stage": "fdm",
            "video_encoder": {"enabled": True, "input": "D2E screen-video frame/action streams"},
            "fdm_objective": {
                "type": "autoregressive_next_action_prediction",
                "interleaved_frame_action_tokens": True,
                "direct_video_action_modeling": True,
                "uses_vlm_cot_or_tool_proxy": False,
            },
            "action_tokenization": {
                "keyboard_press_release_tokens": True,
                "mouse_delta_bins_per_axis": 49,
            },
            "fidelity": {"public_vs_inferred_recorded": True, "claims_fdm1_parity": False},
        }
    }


def test_public_recipe_manifest_records_primary_sources():
    manifest = build_fdm1_public_recipe_manifest()
    source_ids = {source["id"] for source in manifest["sources"]}
    assert manifest["status"] == "pass"
    assert "fdm1_technical_report" in source_ids
    assert "d2e_project_page" in source_ids
    assert "masked-diffusion" in " ".join(manifest["public_recipe_constraints"]["idm"])
    assert "parity" in manifest["claim_boundary"].lower()


def test_recipe_alignment_accepts_idm_and_fdm_candidates(tmp_path):
    write_fdm1_public_recipe_manifest(tmp_path / "manifest.json")
    write_json(tmp_path / "idm.json", _idm_candidate())
    write_json(tmp_path / "fdm.json", _fdm_candidate())
    payload = validate_fdm1_recipe_alignment(
        {
            "recipe_manifest": "manifest.json",
            "candidate_configs": [
                {"stage": "idm", "path": "idm.json"},
                {"stage": "fdm", "path": "fdm.json"},
            ],
        },
        root=tmp_path,
    )
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert [row["status"] for row in payload["candidate_configs"]] == ["pass", "pass"]


def test_recipe_alignment_rejects_arbitrary_supervised_idm(tmp_path):
    write_json(tmp_path / "manifest.json", build_fdm1_public_recipe_manifest())
    write_json(
        tmp_path / "candidate.json",
        {
            "objective": "supervised_cross_entropy",
            "fdm1_recipe": {
                "stage": "idm",
                "video_encoder": {"enabled": False, "input": "summary features"},
                "idm_objective": {"type": "supervised_cross_entropy", "noncausal_frame_conditioning": False},
                "inference_schedule": {"type": "argmax", "steps": 1},
                "action_tokenization": {"keyboard_press_release_tokens": False, "mouse_delta_bins_per_axis": 3},
                "fidelity": {"claims_fdm1_parity": True},
            },
        },
    )
    payload = validate_fdm1_recipe_alignment(
        {
            "recipe_manifest": "manifest.json",
            "candidate_configs": [{"stage": "idm", "path": "candidate.json"}],
        },
        root=tmp_path,
    )
    codes = {finding["code"] for finding in payload["findings"] if finding["severity"] == "error"}
    assert payload["status"] == "fail"
    assert "missing_video_encoder" in codes
    assert "idm_objective_not_masked_diffusion" in codes
    assert "missing_iterative_unmasking_schedule" in codes
    assert "forbidden_fdm1_parity_claim" in codes


def test_recipe_alignment_fails_without_manifest(tmp_path):
    write_json(tmp_path / "candidate.json", _idm_candidate())
    payload = validate_fdm1_recipe_alignment(
        {
            "recipe_manifest": "missing.json",
            "candidate_configs": [{"stage": "idm", "path": "candidate.json"}],
        },
        root=tmp_path,
    )
    assert payload["status"] == "fail"
    assert "missing_recipe_manifest" in {finding["code"] for finding in payload["findings"]}
