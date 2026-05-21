from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from plan_g004_launch import plan_launch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/fdm/g004_launch_readiness.json",
        "goals_path": ".omx/ultragoal/goals.json",
        "g003_goal_id": "G003",
        "g003_completion_config": "configs/eval/g003_completion.json",
        "g003_audit": "artifacts/idm/g003_audit.json",
        "skip_refresh_g003_audit": True,
        "allow_precheckpoint": False,
        "fdm_config": "configs/model/fdm.json",
        "idm_predict_config": "configs/model/predict.json",
        "fdm_labels": "outputs/idm/labels.jsonl",
        "g004_run_script": "scripts/run_g004.sh",
        "g004_run_summary": "artifacts/fdm/run.json",
        "nproc_per_node": 4,
        "expected_gpus": 4,
        "check_gpus": False,
        "allow_fail": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _touch(root: Path, rel_path: str, text: str = "{}\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_base(root: Path, *, g003_status: str = "complete", audit_status: str = "pass", labels: bool = True) -> None:
    write_json(root / ".omx/ultragoal/goals.json", {"goals": [{"id": "G003", "status": g003_status}]})
    write_json(root / "artifacts/idm/g003_audit.json", {"schema": "g003_full_idm_completion_audit.v1", "status": audit_status, "error_count": 0 if audit_status == "pass" else 3})
    write_json(
        root / "configs/model/fdm.json",
        {
            "schema": "streaming_fdm_train_config.v1",
            "labels_path": "outputs/idm/labels.jsonl",
            "source_idm_metadata": "outputs/idm/checkpoint_metadata.json",
            "records_path": "outputs/data/train.jsonl",
            "target_records_path": "outputs/data/target.jsonl",
            "data_universe": "artifacts/sources/universe.json",
            "split_contract": "artifacts/sources/splits.json",
        },
    )
    _touch(root, "configs/model/predict.json")
    _touch(root, "scripts/run_g004.sh", "#!/usr/bin/env bash\n")
    _touch(root, "outputs/idm/checkpoint_metadata.json")
    _touch(root, "outputs/data/train.jsonl", "{\"x\": 1}\n")
    _touch(root, "outputs/data/target.jsonl", "{\"x\": 2}\n")
    _touch(root, "artifacts/sources/universe.json")
    _touch(root, "artifacts/sources/splits.json")
    if labels:
        _touch(root, "outputs/idm/labels.jsonl", "{\"label\": 1}\n")


def test_g004_launch_planner_ready_after_g003_checkpoint_and_inputs(tmp_path: Path):
    _write_base(tmp_path)
    payload = plan_launch(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["error_count"] == 0
    assert payload["g003_goal_status"] == "complete"
    assert payload["pseudolabel_mode"] == "reuse_existing_labels"
    assert "bash scripts/run_g004.sh" in payload["recommended_command"]
    assert json.loads((tmp_path / "artifacts/fdm/g004_launch_readiness.json").read_text())["status"] == "ready"


def test_g004_launch_planner_can_plan_pseudolabel_generation_from_idm_checkpoint(tmp_path: Path):
    _write_base(tmp_path, labels=False)
    payload = plan_launch(_args(tmp_path))
    assert payload["status"] == "ready"
    assert payload["pseudolabel_mode"] == "generate_with_trained_g003_idm"
    assert payload["artifacts"]["idm_predict_config"]["exists"] is True


def test_g004_launch_planner_blocks_before_g003_passes(tmp_path: Path):
    _write_base(tmp_path, g003_status="in_progress", audit_status="fail")
    payload = plan_launch(_args(tmp_path))
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "blocked"
    assert "g003_audit_not_pass" in codes
    assert "g003_goal_not_checkpointed_complete" in codes


def test_g004_launch_planner_blocks_missing_required_inputs(tmp_path: Path):
    _write_base(tmp_path)
    (tmp_path / "outputs/data/train.jsonl").unlink()
    payload = plan_launch(_args(tmp_path))
    assert payload["status"] == "blocked"
    assert any(item["code"] == "missing_required_g004_input" and item["input"] == "records_path" for item in payload["findings"])
