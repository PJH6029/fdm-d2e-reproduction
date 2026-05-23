from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.gidm_baseline_contract import build_gidm_baseline_contract, write_gidm_baseline_contract


def _paper_row(game: str, *, model: str = "G-IDM") -> dict:
    return {
        "game": game,
        "model": model,
        "pearson_x": 79.6,
        "pearson_y": 78.3,
        "scale_ratio_x": 1.23,
        "scale_ratio_y": 1.31,
        "keyboard_accuracy": 73.0,
        "mouse_button_accuracy": 95.7,
    }


def _write_fixture_files(root: Path) -> None:
    for rel in [
        "split_contract.json",
        "data_universe.json",
        "temporal.json",
        "heldout_recording.json",
        "heldout_game.json",
    ]:
        write_json(root / rel, {"schema": rel, "status": "pass"})
    write_json(
        root / "external_manifest.json",
        {
            "schema": "external_artifact_manifest.v1",
            "status": "pass",
            "entries": [
                {
                    "path": "target_all_eval.jsonl",
                    "exists": True,
                    "bytes": 123,
                    "sha256": "abc",
                    "storage_uri": "mlxp-pvc://test/target_all_eval.jsonl",
                    "proof": "sha256",
                }
            ],
        },
    )


def _config(root: Path) -> dict:
    _write_fixture_files(root)
    return {
        "output_path": "contract.json",
        "source_evidence": {
            "d2e_repo_commit": "80e98e26e4dc584ec76fec5789b4a97c275dd032",
            "hf_model": {
                "id": "open-world-agents/Generalist-IDM-1B",
                "revision": "eae16486df3169aafe4a1d74fb2375185b5dc641",
                "gated": False,
                "siblings": ["model.safetensors", "config.json"],
            },
        },
        "paper_reported": {
            "expected_aggregate": {
                "tolerance": 0.0001,
                "metrics": {
                    "pearson_x": 0.796,
                    "pearson_y": 0.783,
                    "scale_ratio_x": 1.23,
                    "scale_ratio_y": 1.31,
                    "keyboard_accuracy": 0.73,
                    "mouse_button_accuracy": 0.957,
                },
            },
            "in_distribution_rows": [_paper_row(f"game-{idx}") for idx in range(6)],
        },
        "official_metric_protocol": {
            "bin_ms": 50,
            "empty_bins_as_correct": False,
            "autoregressive": True,
            "teacher_forcing": False,
        },
        "official_inference_defaults": {
            "time_shift_seconds": 0.1,
            "video_filter": "fps=60,scale=448:448",
        },
        "exact_split_contract": {
            "external_artifact_manifest": "external_manifest.json",
            "split_artifacts": {
                "split_contract": "split_contract.json",
                "data_universe": "data_universe.json",
                "temporal": "temporal.json",
                "heldout_recording": "heldout_recording.json",
                "heldout_game": "heldout_game.json",
            },
            "required_external_artifacts": {"target_all_eval": "target_all_eval.jsonl"},
        },
        "fallback_semantics": {
            "released_gidm_unavailable": "block_exact_split_goal",
            "paper_metrics_substitute_for_exact_split": False,
            "requires_error_log": True,
        },
    }


def test_gidm_baseline_contract_passes_for_complete_fixture(tmp_path):
    payload = build_gidm_baseline_contract(_config(tmp_path), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert payload["paper_reported_targets"]["row_count"] == 6
    assert payload["paper_reported_targets"]["aggregate"]["keyboard_accuracy"] == 0.73
    assert payload["paper_reported_targets"]["unreported_metrics"]["mouse_button_f1"]["status"] == "not_paper_reported"


def test_gidm_baseline_contract_fails_for_paper_target_mismatch(tmp_path):
    config = _config(tmp_path)
    config["paper_reported"]["expected_aggregate"]["metrics"]["keyboard_accuracy"] = 0.90
    payload = build_gidm_baseline_contract(config, root=tmp_path)
    assert payload["status"] == "fail"
    assert "paper_target_metric_mismatch" in {finding["code"] for finding in payload["findings"]}


def test_gidm_baseline_contract_fails_for_permissive_fallback(tmp_path):
    config = _config(tmp_path)
    config["fallback_semantics"]["paper_metrics_substitute_for_exact_split"] = True
    payload = build_gidm_baseline_contract(config, root=tmp_path)
    assert payload["status"] == "fail"
    assert "paper_metrics_must_not_substitute_exact_split" in {finding["code"] for finding in payload["findings"]}


def test_gidm_baseline_contract_writes_output(tmp_path):
    payload = write_gidm_baseline_contract(_config(tmp_path), root=tmp_path)
    written = (tmp_path / "contract.json").read_text(encoding="utf-8")
    assert payload["schema"] == "gidm_baseline_contract.v1"
    assert '"status": "pass"' in written
