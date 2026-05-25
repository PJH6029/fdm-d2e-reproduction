from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.eval import gidm_exact_pipeline as pipeline


def _base_config(tmp_path: Path) -> dict:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"recordings": []}), encoding="utf-8")
    return {
        "goal_id": "G006-g-idm-exact-split",
        "model_name": "released_generalist_idm_1b_exact_split",
        "manifest_path": str(manifest),
        "d2e_repo": str(tmp_path / "D2E"),
        "cuda_devices": ["0", "1"],
        "workers": 2,
        "inference_summary": "artifacts/inference_summary.json",
        "inference_log_dir": "artifacts/logs",
        "by_recording_roots": [str(tmp_path / "by_recording")],
        "target_records": "outputs/targets.jsonl",
        "target_summary": "artifacts/target_summary.json",
        "predictions": "outputs/predictions.jsonl",
        "conversion_summary": "artifacts/conversion_summary.json",
        "paper_metrics": {
            "model_name": "released_generalist_idm_1b_exact_split",
            "output_path": "artifacts/paper_metrics.json",
            "prediction_paths": ["outputs/predictions.jsonl"],
            "target_paths": ["outputs/targets.jsonl"],
            "split_tags": ["temporal", "heldout_recording", "heldout_game"],
        },
        "summary_out": "artifacts/pipeline_summary.json",
        "wandb": {"enabled": False},
    }


def test_gidm_exact_pipeline_inference_stage_delegates_resume_run(tmp_path: Path, monkeypatch):
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return {
            "dry_run": False,
            "planned_recordings": 3,
            "completed_recordings": 3,
            "failed_recordings": 0,
        }

    monkeypatch.setattr(pipeline, "run_gidm_manifest_inference", fake_run)

    payload = pipeline.run_gidm_exact_split_pipeline(
        _base_config(tmp_path),
        root=tmp_path,
        stage="inference",
        cuda_devices=["2", "3"],
        workers=2,
        max_recordings=3,
        log_wandb=False,
    )

    assert payload["status"] == "pass"
    assert payload["statuses"]["inference"]["completed_recordings"] == 3
    assert calls["cuda_devices"] == ["2", "3"]
    assert calls["max_recordings"] == 3
    assert calls["resume"] is True
    assert (tmp_path / "artifacts/pipeline_summary.json").exists()


def test_gidm_exact_pipeline_finalize_stage_is_fail_closed(tmp_path: Path, monkeypatch):
    def fake_extract(**_kwargs):
        return {"status": "pass", "rows_written": 2, "recording_count": 1, "error_count": 0}

    def fake_convert(**_kwargs):
        return {"rows_written": 2, "recording_count": 1, "missing_prediction_count": 1}

    def fake_metrics(**_kwargs):
        return {"status": "pass", "error_count": 0, "alignment": {"rows_seen": 2}}

    monkeypatch.setattr(pipeline, "extract_gidm_target_records", fake_extract)
    monkeypatch.setattr(pipeline, "convert_gidm_mcap_predictions", fake_convert)
    monkeypatch.setattr(pipeline, "write_paper_idm_metrics", fake_metrics)

    strict = pipeline.run_gidm_exact_split_pipeline(
        _base_config(tmp_path),
        root=tmp_path,
        stage="finalize",
        allow_partial=False,
        log_wandb=False,
    )
    partial = pipeline.run_gidm_exact_split_pipeline(
        _base_config(tmp_path),
        root=tmp_path,
        stage="finalize",
        allow_partial=True,
        log_wandb=False,
    )

    assert strict["status"] == "fail"
    assert strict["statuses"]["conversion"]["missing_prediction_count"] == 1
    assert any(item["code"] == "conversion_missing_predictions" for item in strict["findings"])
    assert partial["status"] == "pass"
