from __future__ import annotations

import json
from pathlib import Path

from fdm_d2e.io_utils import write_json
from fdm_d2e.training.g005_aux_prior import run_g005_aux_prior_candidate, train_aux_action_priors


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_train_aux_action_priors_keeps_source_specific_heads(tmp_path: Path) -> None:
    minerl = tmp_path / "outputs/aux_examples/minerl_2019_zenodo_v2/train.jsonl"
    atari = tmp_path / "outputs/aux_examples/atari_head_zenodo_v4/train.jsonl"
    _write_jsonl(
        minerl,
        [
            {"action": {"type": "minecraft_keyboard_mouse", "raw_action": {"attack": 1, "forward": 1, "camera": [1.0, 0.0]}}},
            {"action": {"type": "minecraft_keyboard_mouse", "raw_action": {"attack": 0, "left": 1, "camera": [0.0, -1.0]}}},
        ],
    )
    _write_jsonl(atari, [{"action": {"type": "atari_discrete", "action_id": 2}}, {"action": {"type": "atari_discrete", "action_id": 2}}])
    write_json(
        tmp_path / "artifacts/aux/g005_aux_examples_summary.json",
        {
            "schema": "g005_aux_examples.v1",
            "status": "pass",
            "sources": [
                {"source_id": "minerl_2019_zenodo_v2", "split_files": {"train": {"path": str(minerl.relative_to(tmp_path))}}},
                {"source_id": "atari_head_zenodo_v4", "split_files": {"train": {"path": str(atari.relative_to(tmp_path))}}},
            ],
        },
    )

    payload = train_aux_action_priors(root=tmp_path, aux_examples_summary="artifacts/aux/g005_aux_examples_summary.json")

    assert payload["status"] == "pass"
    assert payload["total_rows_consumed"] == 4
    sources = {row["source_id"]: row for row in payload["sources"]}
    assert sources["minerl_2019_zenodo_v2"]["minecraft_rates"]["attack"] == 0.5
    assert sources["atari_head_zenodo_v4"]["top_actions"][0]["action_key"] == "atari:2"


def test_run_g005_aux_prior_candidate_writes_completion_artifacts(tmp_path: Path) -> None:
    aux_train = tmp_path / "outputs/aux_examples/minerl_2019_zenodo_v2/train.jsonl"
    _write_jsonl(
        aux_train,
        [
            {"action": {"type": "minecraft_keyboard_mouse", "raw_action": {"attack": 1, "forward": 1, "camera": [1.0, 0.0]}}},
            {"action": {"type": "minecraft_keyboard_mouse", "raw_action": {"attack": 0, "left": 1, "camera": [0.0, -1.0]}}},
        ],
    )
    write_json(
        tmp_path / "artifacts/aux/g005_aux_examples_summary.json",
        {
            "schema": "g005_aux_examples.v1",
            "status": "pass",
            "sources": [
                {"source_id": "minerl_2019_zenodo_v2", "split_files": {"train": {"path": str(aux_train.relative_to(tmp_path))}}},
            ],
        },
    )
    write_json(
        tmp_path / "artifacts/aux/g005_aux_namespace_manifest.json",
        {
            "schema": "g005_aux_namespace_manifest.v1",
            "completion_ready": True,
            "source_namespace": "d2e_aux",
            "d2e_eval_manifests": {
                "splits": {
                    "temporal": {"same_hash": True, "d2e_aux_manifest_sha256": "temporal-hash"},
                    "heldout_recording": {"same_hash": True, "d2e_aux_manifest_sha256": "heldout-recording-hash"},
                    "heldout_game": {"same_hash": True, "d2e_aux_manifest_sha256": "heldout-game-hash"},
                }
            },
        },
    )
    write_json(tmp_path / "artifacts/aux/d2e_eval_manifest_hashes.json", {"schema": "x", "splits": {}})
    write_json(tmp_path / "artifacts/sources/d2e_full_data_universe_manifest.json", {"schema": "x"})
    write_json(tmp_path / "artifacts/sources/d2e_full_split_contract.json", {"schema": "x"})
    target_rows = [
        {"sequence_id": "s1", "recording_id": "r1", "eval_split_tags": ["temporal"], "ground_truth_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"]},
        {"sequence_id": "s2", "recording_id": "r1", "eval_split_tags": ["heldout_recording"], "ground_truth_tokens": ["MOUSE_DX_N1", "MOUSE_DY_Z0"]},
        {"sequence_id": "s3", "recording_id": "r2", "eval_split_tags": ["heldout_game"], "ground_truth_tokens": ["KEY_PRESS_87", "MOUSE_DX_Z0", "MOUSE_DY_Z0"]},
    ]
    pred_rows = [
        {"sequence_id": "s1", "recording_id": "r1", "predicted_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"]},
        {"sequence_id": "s2", "recording_id": "r1", "predicted_tokens": ["MOUSE_LEFT_DOWN", "MOUSE_DX_N1", "MOUSE_DY_Z0"]},
        {"sequence_id": "s3", "recording_id": "r2", "predicted_tokens": ["KEY_PRESS_87", "MOUSE_DX_Z0", "MOUSE_DY_Z0"]},
    ]
    _write_jsonl(tmp_path / "d2e_target.jsonl", target_rows)
    _write_jsonl(tmp_path / "d2e_pred.jsonl", pred_rows)
    _write_jsonl(tmp_path / "d2e_target_part0.jsonl", target_rows[:2])
    _write_jsonl(tmp_path / "d2e_target_part1.jsonl", target_rows[2:])
    _write_jsonl(tmp_path / "d2e_pred_part0.jsonl", pred_rows[:2])
    _write_jsonl(tmp_path / "d2e_pred_part1.jsonl", pred_rows[2:])
    write_json(tmp_path / "artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json", {"metrics": {"baseline": True}})
    split_cfg = {
        "schema": "split_statistical_comparison_builder_config.v1",
        "model_name": "g005_aux_action_prior_d2e_aux_best",
        "predictions_path": "outputs/fdm_aux/d2e_aux_best/predictions.jsonl",
        "ground_truth_path": "d2e_target.jsonl",
        "streaming": True,
        "output_dir": "outputs/fdm_aux/d2e_aux_best",
        "summary_out": "artifacts/eval/g005_split_statistical_comparisons_summary.json",
        "endpoints": "configs/eval/primary_endpoints.yaml",
        "baseline_names": ["noop"],
        "split_tags": ["temporal", "heldout_recording", "heldout_game"],
    }
    write_json(tmp_path / "configs/eval/g005_split_statistics.yaml", split_cfg)
    write_json(
        tmp_path / "configs/eval/primary_endpoints.yaml",
        {
            "schema": "primary_endpoints.v1",
            "reference_baseline": "noop",
            "cluster_key": "recording_id",
            "bootstrap": {"n_resamples": 5, "seed": 1},
            "endpoints": [
                {
                    "name": "mouse_button_f1",
                    "metric_path": ["mouse_button", "f1"],
                    "direction": "higher",
                    "min_effect": 0.0,
                }
            ],
        },
    )

    run_summary = run_g005_aux_prior_candidate(
        {
            "output_dir": "outputs/fdm_aux/d2e_aux_best",
            "aux_examples_summary": "artifacts/aux/g005_aux_examples_summary.json",
            "namespace_manifest": "artifacts/aux/g005_aux_namespace_manifest.json",
            "eval_manifest_hashes": "artifacts/aux/d2e_eval_manifest_hashes.json",
            "data_universe": "artifacts/sources/d2e_full_data_universe_manifest.json",
            "split_contract": "artifacts/sources/d2e_full_split_contract.json",
            "d2e_only_predictions": "d2e_pred.jsonl",
            "d2e_only_prediction_paths": ["d2e_pred_part0.jsonl", "d2e_pred_part1.jsonl"],
            "d2e_target_records": "d2e_target.jsonl",
            "d2e_target_paths": ["d2e_target_part0.jsonl", "d2e_target_part1.jsonl"],
            "d2e_only_summary": "artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json",
            "split_stats_config": "configs/eval/g005_split_statistics.yaml",
            "split_stats_summary": "artifacts/eval/g005_split_statistical_comparisons_summary.json",
            "ablation_summary": "artifacts/aux/d2e_aux_ablation_summary.json",
            "run_summary": "artifacts/aux/g005_d2e_aux_train_run.json",
            "max_button_stride": 2,
            "prediction_workers": 2,
        },
        root=tmp_path,
    )

    assert run_summary["status"] == "pass"
    prediction_summary = json.loads((tmp_path / "outputs/fdm_aux/d2e_aux_best/prediction_build_summary.json").read_text())
    assert prediction_summary["parallel_prediction"] is True
    assert prediction_summary["rows"] == 3
    assert prediction_summary["target_link"]["target"].endswith("d2e_target.jsonl")
    assert (tmp_path / "outputs/fdm_aux/d2e_aux_best/checkpoint.pt").exists()
    metadata = json.loads((tmp_path / "outputs/fdm_aux/d2e_aux_best/checkpoint_metadata.json").read_text())
    assert metadata["source_namespace"] == "d2e_aux"
    ablation = json.loads((tmp_path / "artifacts/aux/d2e_aux_ablation_summary.json").read_text())
    assert ablation["status"] == "pass"
    assert {row["split"] for row in ablation["split_results"]} == {"temporal", "heldout_recording", "heldout_game"}
