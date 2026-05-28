from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_json, write_json


PUBLIC_RECIPE_MANIFEST: dict[str, Any] = {
    "schema": "fdm1_public_recipe_manifest.v1",
    "status": "pass",
    "sources": [
        {
            "id": "fdm1_technical_report",
            "url": "https://si.inc/posts/fdm1/",
            "accessed_date": "2026-05-29",
            "role": "primary_public_recipe_source",
        },
        {
            "id": "d2e_project_page",
            "url": "https://worv-ai.github.io/d2e/",
            "accessed_date": "2026-05-29",
            "role": "D2E dataset/model/metric context",
        },
    ],
    "public_recipe_constraints": {
        "global": [
            "Use D2E data to reproduce the public FDM-1 IDM/FDM training recipe shape, not arbitrary easier objectives.",
            "Record public-vs-inferred-vs-novel choices and never claim closed-source FDM-1 parity.",
        ],
        "video_encoder": [
            "Operate on screen video rather than isolated screenshots where feasible.",
            "Use masked/self-supervised compression-style representation learning or an explicitly documented approximation.",
        ],
        "idm": [
            "Condition on all frames in the labeling window and masked action-token slots.",
            "Use a non-causal masked-diffusion/iterative-unmask objective over action tokens.",
            "Use an iterative denoising/unmasking inference schedule, targeting the public 16-step schedule when feasible.",
        ],
        "fdm": [
            "Train an autoregressive next-action model on interleaved frame and action tokens.",
            "Use direct video/action-token modeling rather than VLM screenshot CoT/tool-use proxies.",
        ],
        "action_tokenization": [
            "Represent key press/release events as tokens.",
            "Represent mouse movement as discrete binned X/Y components; prefer 49 exponential bins per public FDM-1 description where feasible.",
            "Consider click-position/trajectory auxiliary prediction for mouse movements when feasible.",
        ],
    },
    "claim_boundary": "This manifest captures only publicly described recipe constraints; it is not evidence of FDM-1 internal parity or metric success.",
}


def _path(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(data: dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for part in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _recipe(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("fdm1_recipe")
    return value if isinstance(value, dict) else {}


def build_fdm1_public_recipe_manifest() -> dict[str, Any]:
    return copy.deepcopy(PUBLIC_RECIPE_MANIFEST)


def write_fdm1_public_recipe_manifest(output_path: str | Path) -> dict[str, Any]:
    payload = build_fdm1_public_recipe_manifest()
    write_json(output_path, payload)
    return payload


def _check_idm_recipe(config: dict[str, Any]) -> list[dict[str, Any]]:
    recipe = _recipe(config)
    findings: list[dict[str, Any]] = []

    stage = str(recipe.get("stage") or config.get("fdm1_recipe_stage") or "")
    if stage != "idm":
        findings.append({"severity": "error", "code": "idm_stage_not_declared", "actual": stage, "expected": "idm"})

    video = recipe.get("video_encoder") if isinstance(recipe.get("video_encoder"), dict) else {}
    if video.get("enabled") is not True:
        findings.append({"severity": "error", "code": "missing_video_encoder", "expected": "fdm1_recipe.video_encoder.enabled=true"})
    if "video" not in str(video.get("input", "")).lower() and "frame" not in str(video.get("input", "")).lower():
        findings.append({"severity": "warning", "code": "video_encoder_input_not_video_or_frame", "actual": video.get("input")})

    objective = recipe.get("idm_objective") if isinstance(recipe.get("idm_objective"), dict) else {}
    objective_type = str(objective.get("type") or config.get("objective") or "")
    if objective_type != "masked_diffusion_action_tokens":
        findings.append(
            {
                "severity": "error",
                "code": "idm_objective_not_masked_diffusion",
                "actual": objective_type,
                "expected": "masked_diffusion_action_tokens",
            }
        )
    if objective.get("noncausal_frame_conditioning") is not True:
        findings.append({"severity": "error", "code": "idm_not_noncausal_frame_conditioned"})
    if str(objective.get("conditioning", "")) != "all_window_frames_plus_masked_action_tokens":
        findings.append(
            {
                "severity": "error",
                "code": "idm_conditioning_not_masked_action_tokens",
                "actual": objective.get("conditioning"),
            }
        )

    schedule = recipe.get("inference_schedule") if isinstance(recipe.get("inference_schedule"), dict) else {}
    if str(schedule.get("type") or "") != "iterative_unmasking":
        findings.append({"severity": "error", "code": "missing_iterative_unmasking_schedule", "actual": schedule})
    steps = int(schedule.get("steps") or 0)
    if steps < 2:
        findings.append({"severity": "error", "code": "iterative_schedule_too_short", "actual": steps, "expected_min": 2})
    elif steps != 16:
        findings.append({"severity": "warning", "code": "iterative_schedule_not_public_16_step", "actual": steps, "public_target": 16})

    tok = recipe.get("action_tokenization") if isinstance(recipe.get("action_tokenization"), dict) else {}
    if tok.get("keyboard_press_release_tokens") is not True:
        findings.append({"severity": "error", "code": "missing_keyboard_press_release_tokens"})
    mouse_bins = int(tok.get("mouse_delta_bins_per_axis") or 0)
    if mouse_bins < 11:
        findings.append({"severity": "error", "code": "mouse_delta_bins_too_small", "actual": mouse_bins, "expected_min": 11})
    if mouse_bins != 49:
        findings.append({"severity": "warning", "code": "mouse_delta_bins_not_public_49", "actual": mouse_bins, "public_target": 49})

    fidelity = recipe.get("fidelity") if isinstance(recipe.get("fidelity"), dict) else {}
    if not fidelity.get("public_vs_inferred_recorded"):
        findings.append({"severity": "error", "code": "missing_public_vs_inferred_fidelity_record"})
    if fidelity.get("claims_fdm1_parity") is True:
        findings.append({"severity": "error", "code": "forbidden_fdm1_parity_claim"})
    return findings


def _check_fdm_recipe(config: dict[str, Any]) -> list[dict[str, Any]]:
    recipe = _recipe(config)
    findings: list[dict[str, Any]] = []
    stage = str(recipe.get("stage") or config.get("fdm1_recipe_stage") or "")
    if stage != "fdm":
        findings.append({"severity": "error", "code": "fdm_stage_not_declared", "actual": stage, "expected": "fdm"})

    video = recipe.get("video_encoder") if isinstance(recipe.get("video_encoder"), dict) else {}
    if video.get("enabled") is not True:
        findings.append({"severity": "error", "code": "fdm_missing_video_encoder", "expected": "fdm1_recipe.video_encoder.enabled=true"})

    objective = recipe.get("fdm_objective") if isinstance(recipe.get("fdm_objective"), dict) else {}
    if str(objective.get("type") or "") != "autoregressive_next_action_prediction":
        findings.append({"severity": "error", "code": "fdm_objective_not_autoregressive_next_action"})
    if objective.get("interleaved_frame_action_tokens") is not True:
        findings.append({"severity": "error", "code": "fdm_missing_interleaved_frame_action_tokens"})
    if objective.get("direct_video_action_modeling") is not True:
        findings.append({"severity": "error", "code": "fdm_not_direct_video_action_modeling"})
    if objective.get("uses_vlm_cot_or_tool_proxy") is True:
        findings.append({"severity": "error", "code": "fdm_uses_forbidden_vlm_cot_or_tool_proxy"})

    tok = recipe.get("action_tokenization") if isinstance(recipe.get("action_tokenization"), dict) else {}
    if tok.get("keyboard_press_release_tokens") is not True:
        findings.append({"severity": "error", "code": "fdm_missing_keyboard_press_release_tokens"})
    mouse_bins = int(tok.get("mouse_delta_bins_per_axis") or 0)
    if mouse_bins < 11:
        findings.append({"severity": "error", "code": "fdm_mouse_delta_bins_too_small", "actual": mouse_bins, "expected_min": 11})
    if mouse_bins != 49:
        findings.append({"severity": "warning", "code": "fdm_mouse_delta_bins_not_public_49", "actual": mouse_bins, "public_target": 49})

    fidelity = recipe.get("fidelity") if isinstance(recipe.get("fidelity"), dict) else {}
    if not fidelity.get("public_vs_inferred_recorded"):
        findings.append({"severity": "error", "code": "fdm_missing_public_vs_inferred_fidelity_record"})
    if fidelity.get("claims_fdm1_parity") is True:
        findings.append({"severity": "error", "code": "forbidden_fdm1_parity_claim"})
    return findings


def validate_fdm1_recipe_alignment(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    manifest_path = _path(root_path, config.get("recipe_manifest"))
    manifest = read_json(manifest_path) if manifest_path and manifest_path.exists() else None
    findings: list[dict[str, Any]] = []
    if not isinstance(manifest, dict):
        findings.append({"severity": "error", "code": "missing_recipe_manifest", "path": str(manifest_path) if manifest_path else None})
    elif manifest.get("status") != "pass":
        findings.append({"severity": "error", "code": "recipe_manifest_not_pass", "status": manifest.get("status")})

    candidate_rows: list[dict[str, Any]] = []
    for row in config.get("candidate_configs", []):
        candidate_path = _path(root_path, row.get("path") if isinstance(row, dict) else row)
        stage_hint = str(row.get("stage", "") if isinstance(row, dict) else "")
        candidate_findings: list[dict[str, Any]] = []
        candidate_config: dict[str, Any] | None = None
        if candidate_path is None or not candidate_path.exists():
            candidate_findings.append({"severity": "error", "code": "missing_candidate_config", "path": str(candidate_path) if candidate_path else None})
        else:
            candidate_config = load_config(candidate_path)
            stage = stage_hint or str(_get(candidate_config, ["fdm1_recipe", "stage"], candidate_config.get("fdm1_recipe_stage", "")))
            if stage == "idm":
                candidate_findings.extend(_check_idm_recipe(candidate_config))
            elif stage == "fdm":
                candidate_findings.extend(_check_fdm_recipe(candidate_config))
            else:
                candidate_findings.append({"severity": "error", "code": "unknown_recipe_stage", "actual": stage})
        candidate_errors = [item for item in candidate_findings if item.get("severity") == "error"]
        candidate_rows.append(
            {
                "path": str(candidate_path) if candidate_path else None,
                "stage": stage_hint or (str(_get(candidate_config or {}, ["fdm1_recipe", "stage"], "")) if candidate_config else ""),
                "status": "pass" if not candidate_errors else "fail",
                "error_count": len(candidate_errors),
                "findings": candidate_findings,
                "sha256": _file_sha256(candidate_path) if candidate_path else None,
            }
        )
        findings.extend({**item, "candidate_path": str(candidate_path) if candidate_path else None} for item in candidate_findings)

    errors = [item for item in findings if item.get("severity") == "error"]
    return {
        "schema": "fdm1_recipe_alignment_audit.v1",
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "recipe_manifest": str(manifest_path) if manifest_path else None,
        "candidate_configs": candidate_rows,
        "findings": findings,
        "claim_boundary": "Recipe-alignment audit only; metric success and full D2E training evidence remain separate G005/G009 gates.",
    }


def write_fdm1_recipe_alignment_audit(config: dict[str, Any], *, root: str | Path = ".", output_path: str | Path | None = None) -> dict[str, Any]:
    payload = validate_fdm1_recipe_alignment(config, root=root)
    out = output_path or config.get("output_path", "artifacts/reproducibility/fdm1_recipe_alignment_audit.json")
    write_json(_path(Path(root), out) or out, payload)
    return payload
