from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.renewed_gates import validate_renewed_metric_gates, write_renewed_metric_gate_audit


def _metrics(*, good: bool) -> dict:
    if good:
        return {
            "keyboard": {"accuracy": 0.90},
            "mouse_button": {"accuracy": 0.88, "f1": 0.89, "no_button_false_positive_rate": 0.05},
            "mouse_move": {"pearson": 0.91, "scale_ratio": 1.0},
        }
    return {
        "keyboard": {"accuracy": 0.01},
        "mouse_button": {"accuracy": 0.05, "f1": 0.02, "no_button_false_positive_rate": 0.31},
        "mouse_move": {"pearson": 0.35, "scale_ratio": 1.6},
    }


def _comparison(split: str, *, good: bool) -> dict:
    values = {
        "keyboard_accuracy": 0.91 if good else 0.02,
        "mouse_button_f1": 0.92 if good else 0.03,
        "mouse_move_pearson": 0.93 if good else 0.36,
        "no_button_false_positive_rate": 0.04 if good else 0.32,
    }
    return {
        "split": split,
        "comparisons": [
            {
                "split": split,
                "model": "candidate",
                "endpoint": endpoint,
                "candidate_value": value,
                "status": "computed",
            }
            for endpoint, value in values.items()
        ],
    }


def _fixture(root: Path, *, good: bool) -> dict:
    write_json(root / "metrics.json", _metrics(good=good))
    write_json(root / "aux_metrics.json", _metrics(good=good))
    outputs = []
    for split in ["temporal", "heldout_recording", "heldout_game"]:
        rel = f"{split}.json"
        write_json(root / rel, _comparison(split, good=good))
        outputs.append({"split": split, "path": rel, "status": "pass", "comparisons": 4})
    write_json(root / "split_summary.json", {"status": "pass", "outputs": outputs})
    write_json(root / "old_audit.json", {"status": "pass"})
    write_json(root / "evidence.json", {"status": "pass"})
    write_json(
        root / ".omx/ultragoal/archive/fdm-d2e-renewal-test/goals.json",
        {"goals": [{"id": "old-goal", "status": "complete"}]},
    )
    return {
        "expected_gate_status": "pass" if good else "fail",
        "model_name": "candidate",
        "paths": {
            "candidate_metrics": "metrics.json",
            "aux_metrics": "aux_metrics.json",
            "split_stats_summary": "split_summary.json",
        },
        "required_splits": ["temporal", "heldout_recording", "heldout_game"],
        "old_ultragoal_archive_glob": ".omx/ultragoal/archive/fdm-d2e-renewal-*/goals.json",
        "old_goal_ids": ["old-goal"],
        "required_evidence_files": ["evidence.json"],
        "old_completion_audits": [{"id": "old-audit", "path": "old_audit.json", "expected_status": "pass"}],
        "hard_gates": [
            {
                "name": "keyboard_accuracy",
                "endpoint": "keyboard_accuracy",
                "metric_path": ["keyboard", "accuracy"],
                "direction": "higher",
                "target": 0.80,
            },
            {
                "name": "mouse_button_f1",
                "endpoint": "mouse_button_f1",
                "metric_path": ["mouse_button", "f1"],
                "direction": "higher",
                "target": 0.80,
            },
            {
                "name": "mouse_move_pearson",
                "endpoint": "mouse_move_pearson",
                "metric_path": ["mouse_move", "pearson"],
                "direction": "higher",
                "target": 0.80,
            },
            {
                "name": "no_button_false_positive_rate",
                "endpoint": "no_button_false_positive_rate",
                "metric_path": ["mouse_button", "no_button_false_positive_rate"],
                "direction": "lower",
                "target": 0.10,
            },
        ],
    }


def test_renewed_gate_audit_passes_when_expected_failure_is_detected(tmp_path):
    config = _fixture(tmp_path, good=False)
    payload = validate_renewed_metric_gates(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["gate_status"] == "fail"
    assert payload["gate_error_count"] == 16
    assert {row["source"] for row in payload["gate_failures"]} == {
        "aggregate",
        "split:temporal",
        "split:heldout_recording",
        "split:heldout_game",
    }


def test_renewed_gate_audit_passes_for_real_gate_pass_when_expected(tmp_path):
    config = _fixture(tmp_path, good=True)
    payload = validate_renewed_metric_gates(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["gate_status"] == "pass"
    assert payload["gate_error_count"] == 0


def test_renewed_gate_audit_fails_when_expectation_mismatches(tmp_path):
    config = _fixture(tmp_path, good=False)
    config["expected_gate_status"] = "pass"
    payload = validate_renewed_metric_gates(config, root=tmp_path)
    assert payload["status"] == "fail"
    assert payload["gate_status"] == "fail"
    assert {item["code"] for item in payload["findings"]} == {"gate_status_expectation_mismatch"}


def test_renewed_gate_audit_writes_output(tmp_path):
    config = _fixture(tmp_path, good=False)
    config["output_path"] = "artifact.json"
    payload = write_renewed_metric_gate_audit(config, root=tmp_path)
    written = (tmp_path / "artifact.json").read_text(encoding="utf-8")
    assert payload["schema"] == "renewed_metric_gate_audit.v1"
    assert '"gate_status": "fail"' in written


def test_renewed_gate_audit_uses_archive_with_required_old_goal_ids(tmp_path):
    config = _fixture(tmp_path, good=False)
    write_json(
        tmp_path / ".omx/ultragoal/archive/fdm-d2e-renewal-newer/goals.json",
        {"goals": [{"id": "renewed-goal", "status": "in_progress"}]},
    )
    payload = validate_renewed_metric_gates(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["old_evidence"]["old_ultragoal_archive_goals_path"].endswith(
        "fdm-d2e-renewal-test/goals.json"
    )
    assert payload["old_evidence"]["old_goal_statuses"] == {"old-goal": "complete"}
