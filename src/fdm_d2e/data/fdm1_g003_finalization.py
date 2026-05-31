from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.data.fdm1_action_dataset import write_action_slot_dataset_streaming_from_jsonl
from fdm_d2e.data.fdm1_alignment_report import build_alignment_report_from_jsonl
from fdm_d2e.data.fdm1_mouse_bins import build_fitted_mouse_bins
from fdm_d2e.io_utils import read_json, write_json
from fdm_d2e.reporting.fdm1_g003_completion import write_fdm1_g003_action_dataset_completion_audit
from fdm_d2e.tokenization.fdm1_actions import ActionSlotTokenizer, MouseMoveBinner


def _exists(root: Path, rel: str | None) -> bool:
    return bool(rel) and (root / str(rel)).exists()


def _tokenizer_from_config(path: Path, *, bin_ms: int, k_event_slots: int | None = None) -> ActionSlotTokenizer:
    config = read_json(path)
    mouse = config.get("mouse_move", {}) if isinstance(config.get("mouse_move"), dict) else {}
    boundaries = tuple(float(v) for v in mouse.get("positive_boundaries_default", MouseMoveBinner().boundaries))
    compound = str(mouse.get("default", "compound")).lower() == "compound"
    configured_k = int(config.get("k_event_slots_default", 8))
    return ActionSlotTokenizer(
        k_event_slots=int(k_event_slots if k_event_slots is not None else configured_k),
        mouse_binner=MouseMoveBinner(boundaries=boundaries, compound=compound),
        bin_ms=int(bin_ms),
    )


def finalize_fdm1_g003_action_dataset(config: dict[str, Any], *, root: str | Path = ".", force: bool = False) -> dict[str, Any]:
    root_path = Path(root)
    paths = {key: str(value) for key, value in dict(config.get("paths", {})).items()}
    decoded_records = str(config.get("decoded_records", "outputs/data/fdm1_d2e_480p_window_records/all_records.jsonl"))
    decoded_path = root_path / decoded_records
    if not decoded_path.exists():
        raise FileNotFoundError(f"decoded records missing: {decoded_records}")

    steps: list[dict[str, Any]] = []
    bin_ms = int(config.get("bin_ms", 50))
    frame_fps = int(config.get("frame_fps", 20))
    k_event_slots = int(config.get("k_event_slots", 8))
    click_horizon_seconds = float(config.get("click_horizon_seconds", 1.0))
    click_grid = (int(config.get("click_grid_width", 32)), int(config.get("click_grid_height", 18)))
    screen_size = (int(config.get("screen_width", 854)), int(config.get("screen_height", 480)))

    fitted_bins = paths.get("fitted_mouse_bins", "artifacts/sources/fdm1_g003_fitted_mouse_bins.json")
    fitted_config = paths.get("fitted_tokenization_config", "artifacts/sources/fdm1_action_slots_fitted_config.json")
    base_config = str(config.get("base_tokenization_config", "configs/tokenization/fdm1_action_slots.json"))
    if force or not (_exists(root_path, fitted_bins) and _exists(root_path, fitted_config)):
        build_fitted_mouse_bins(
            [decoded_path],
            base_tokenization_config=root_path / base_config,
            bins_output_path=root_path / fitted_bins,
            fitted_config_path=root_path / fitted_config,
            split=str(config.get("mouse_fit_split", "train_core")),
        )
        steps.append({"step": "fit_mouse_bins", "status": "ran"})
    else:
        steps.append({"step": "fit_mouse_bins", "status": "skipped_existing"})

    action_slots = paths.get("action_slots", "outputs/data/fdm1_action_slots/action_slots.jsonl")
    dataset_summary = paths.get("dataset_summary", "outputs/data/fdm1_action_slots/dataset_summary.json")
    action_output_dir = str(config.get("action_output_dir", str(Path(action_slots).parent)))
    if force or not (_exists(root_path, action_slots) and _exists(root_path, dataset_summary)):
        tokenizer = _tokenizer_from_config(root_path / fitted_config, bin_ms=bin_ms, k_event_slots=k_event_slots)
        write_action_slot_dataset_streaming_from_jsonl(
            [decoded_path],
            output_dir=root_path / action_output_dir,
            tokenization_config_path=fitted_config,
            tokenizer=tokenizer,
            bin_ms=bin_ms,
            frame_fps=frame_fps,
            click_horizon_seconds=click_horizon_seconds,
            click_grid=click_grid,
            screen_size=screen_size,
        )
        steps.append({"step": "materialize_action_slots", "status": "ran"})
    else:
        steps.append({"step": "materialize_action_slots", "status": "skipped_existing"})

    visual_audit = paths.get("visual_alignment_audit", "artifacts/sources/fdm1_g003_action_alignment_visual_check.json")
    visual_report = paths.get("visual_alignment_report", "artifacts/reports/fdm1_g003_action_alignment_visual_check.md")
    if force or not (_exists(root_path, visual_audit) and _exists(root_path, visual_report)):
        build_alignment_report_from_jsonl(
            root_path / action_slots,
            markdown_path=root_path / visual_report,
            audit_path=root_path / visual_audit,
            expected_bin_ms=bin_ms,
            max_rows=int(config.get("alignment_max_rows", 24)),
            recording_id=config.get("alignment_recording_id"),
            game=config.get("alignment_game"),
        )
        steps.append({"step": "visual_alignment", "status": "ran"})
    else:
        steps.append({"step": "visual_alignment", "status": "skipped_existing"})

    completion_config_path = str(config.get("completion_config", "configs/eval/fdm1_g003_action_dataset_completion.yaml"))
    completion_config = read_json(root_path / completion_config_path)
    audit = write_fdm1_g003_action_dataset_completion_audit(completion_config, root=root_path)
    steps.append({"step": "completion_audit", "status": audit["status"], "error_count": audit["error_count"]})
    summary = {
        "schema": "fdm1_g003_action_dataset_finalization_summary.v1",
        "status": "pass" if audit["status"] == "pass" else "fail",
        "steps": steps,
        "decoded_records": decoded_records,
        "completion_config": completion_config_path,
        "completion_audit_status": audit["status"],
        "completion_audit_error_count": audit["error_count"],
        "claim_boundary": "G003 finalization covers action-slot dataset evidence only; it does not prove model training, metric wins, harness control, or FDM-1 parity.",
    }
    out = config.get("output_path")
    if out:
        write_json(root_path / str(out), summary)
    return summary


__all__ = ["finalize_fdm1_g003_action_dataset"]
