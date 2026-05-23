from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.failure_root_cause import build_failure_root_cause_audit, write_failure_root_cause_audit


def _metrics(*, predicted: int, fpr: float, f1: float) -> dict:
    return {
        "schema": "metrics.v1",
        "num_examples": 100,
        "keyboard": {"accuracy": 0.1, "num_examples": 20},
        "mouse_button": {
            "accuracy": 0.2,
            "f1": f1,
            "precision": 0.1,
            "recall": 0.2,
            "num_examples": 10,
            "predicted_examples": predicted,
            "false_positive_examples": max(predicted - 2, 0),
            "false_negative_examples": 8,
            "no_button_examples": 90,
            "no_button_false_positive_examples": int(90 * fpr),
            "no_button_false_positive_rate": fpr,
        },
        "mouse_move": {"pearson": 0.3, "scale_ratio": 1.5, "num_values": 50},
        "failure_count": 99,
    }


def _comparison(split: str) -> dict:
    return {
        "split": split,
        "comparisons": [
            {"split": split, "model": "candidate", "endpoint": "keyboard_accuracy", "candidate_value": 0.1, "status": "computed"},
            {"split": split, "model": "candidate", "endpoint": "mouse_button_f1", "candidate_value": 0.2, "status": "computed"},
            {"split": split, "model": "candidate", "endpoint": "no_button_false_positive_rate", "candidate_value": 0.3, "status": "computed"},
        ],
    }


def _fixture(root: Path) -> dict:
    write_json(root / "idm_metrics.json", _metrics(predicted=5, fpr=0.02, f1=0.05))
    write_json(root / "fdm_metrics.json", _metrics(predicted=50, fpr=0.30, f1=0.08))
    write_json(root / "aux_metrics.json", _metrics(predicted=50, fpr=0.30, f1=0.08))
    metadata = {
        "label_source": "idm_pseudolabel",
        "torch_checkpoint_metadata": {
            "calibration": {"mode": "global_threshold_streaming", "category_threshold": 0.35},
            "feature_mode": "summary_causal_compact_grid8_time_prior_action",
            "input_dim": 504,
            "categorical_vocab": ["NOOP", "KEY_PRESS_87", "KEY_RELEASE_87", "MOUSE_LEFT_DOWN", "MOUSE_X1_DOWN", "MOUSE_DX_P1", "MOUSE_DY_Z0"],
            "target_records": 100,
        },
        "oracle_ground_truth_control": False,
    }
    write_json(root / "fdm_metadata.json", metadata)
    write_json(root / "idm_metadata.json", {"feature_mode": "summary", "input_dim": 10, "categorical_vocab": ["NOOP"]})
    write_json(root / "split.json", {"counts": {"target_games": {"GameA": 60, "GameB": 40}}})
    outputs = []
    for split in ["temporal", "heldout_recording", "heldout_game"]:
        rel = f"{split}.json"
        write_json(root / rel, _comparison(split))
        outputs.append({"split": split, "path": rel, "status": "pass"})
    write_json(root / "split_stats.json", {"outputs": outputs})
    write_json(root / "renewed.json", {"status": "pass", "gate_status": "fail", "gate_error_count": 16})
    write_json(
        root / "external.json",
        {
            "status": "pass",
            "entries": [
                {"path": "targets.jsonl", "exists": True, "bytes": 10, "sha256": "a", "storage_uri": "mlxp-pvc://targets", "proof": "sha256"},
                {"path": "predictions.jsonl", "exists": True, "bytes": 10, "sha256": "b", "storage_uri": "mlxp-pvc://preds", "proof": "sha256"},
            ],
        },
    )
    return {
        "fdm_model_name": "candidate",
        "expected_raw_rows": 100,
        "paths": {
            "idm_metrics": "idm_metrics.json",
            "fdm_metrics": "fdm_metrics.json",
            "aux_metrics": "aux_metrics.json",
            "idm_metadata": "idm_metadata.json",
            "fdm_metadata": "fdm_metadata.json",
            "fdm_split_summary": "split.json",
            "fdm_split_stats_summary": "split_stats.json",
            "renewed_gate_audit": "renewed.json",
            "external_artifact_manifest": "external.json",
            "raw_diagnostics": "raw.json",
        },
        "required_artifacts": ["idm_metrics.json", "fdm_metrics.json", "fdm_metadata.json", "external.json"],
        "raw_required_paths": ["targets.jsonl", "predictions.jsonl"],
        "required_axes": [
            "action_distribution",
            "token_vocabulary",
            "label_prediction_alignment",
            "no_op_thresholding",
            "heldout_split_confusion",
            "per_game_confusion",
            "oracle_upper_bound_sanity",
            "d2e_metric_compatibility",
            "feature_sufficiency",
        ],
        "accepted_axis_statuses": ["computed", "inferred", "pvc_required"],
    }


def test_failure_root_cause_audit_passes_with_ranked_evidence_and_external_raw_gap(tmp_path):
    config = _fixture(tmp_path)
    payload = build_failure_root_cause_audit(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["axes"]["per_game_confusion"]["status"] == "pvc_required"
    assert payload["ranked_root_causes"][0]["id"] == "fdm_mouse_button_overfire"
    assert payload["axes"]["token_vocabulary"]["evidence"]["fdm"]["x_button_tokens"] == ["MOUSE_X1_DOWN"]


def test_failure_root_cause_audit_uses_raw_diagnostic_when_present(tmp_path):
    config = _fixture(tmp_path)
    write_json(tmp_path / "raw.json", {"status": "pass", "alignment": {"rows_seen": 100}, "groups": {"game": {"GameA": {}}}})
    payload = build_failure_root_cause_audit(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["axes"]["per_game_confusion"]["status"] == "computed"


def test_failure_root_cause_audit_keeps_sampled_raw_diagnostic_as_pvc_required(tmp_path):
    config = _fixture(tmp_path)
    write_json(tmp_path / "raw.json", {"status": "pass", "alignment": {"rows_seen": 10}, "groups": {"game": {"GameA": {}}}})
    payload = build_failure_root_cause_audit(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["axes"]["per_game_confusion"]["status"] == "pvc_required"


def test_failure_root_cause_audit_fails_without_external_raw_proof(tmp_path):
    config = _fixture(tmp_path)
    write_json(tmp_path / "external.json", {"status": "pass", "entries": []})
    payload = build_failure_root_cause_audit(config, root=tmp_path)
    assert payload["status"] == "fail"
    assert {item["code"] for item in payload["findings"]} == {"raw_artifact_external_proof_missing"}


def test_failure_root_cause_audit_writes_output(tmp_path):
    config = _fixture(tmp_path)
    config["output_path"] = "out.json"
    payload = write_failure_root_cause_audit(config, root=tmp_path)
    assert payload["schema"] == "g002_failure_root_cause_audit.v1"
    assert (tmp_path / "out.json").is_file()
