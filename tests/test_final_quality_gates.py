from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import write_json
from fdm_d2e.reporting.quality_gates import validate_final_quality_gates, write_final_quality_gate_audit


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _config() -> dict:
    return {
        "goals_path": ".omx/ultragoal/goals.json",
        "package_manifest_path": "artifacts/reproducibility/package_manifest.json",
        "claim_boundary_audit_path": "artifacts/reproducibility/claim_boundary_audit.json",
        "live_suite_evidence_validation_path": "artifacts/harness/live_validation.json",
        "require_all_goals_complete": True,
        "require_live_suite_pass": True,
        "goal_gates": [
            {"id": "G001", "requires_status": "complete", "required_paths": ["artifacts/g001.json"]},
            {
                "id": "G008",
                "requires_status": "complete",
                "required_paths": ["artifacts/harness/live_validation.json"],
                "json_assertions": [{"path": "artifacts/harness/live_validation.json", "json_path": "quality_gate.status", "equals": "pass"}],
            },
        ],
    }


def _complete_fixture(root: Path) -> None:
    write_json(
        root / ".omx/ultragoal/goals.json",
        {"goals": [{"id": "G001", "status": "complete"}, {"id": "G008", "status": "complete"}]},
    )
    _write(root / "artifacts/g001.json", "{}")
    write_json(root / "artifacts/harness/live_validation.json", {"quality_gate": {"status": "pass"}})
    write_json(root / "artifacts/reproducibility/claim_boundary_audit.json", {"status": "pass"})
    entries = [
        {"path": "artifacts/g001.json", "kind": "evidence_artifact", "bytes": 2, "sha256": "stub"},
        {"path": "artifacts/harness/live_validation.json", "kind": "evidence_artifact", "bytes": 2, "sha256": "stub"},
    ]
    write_json(root / "artifacts/reproducibility/package_manifest.json", {"schema": "repro_package_manifest.v1", "entries": entries})


def test_final_quality_gate_passes_when_all_configured_evidence_is_present(tmp_path):
    _complete_fixture(tmp_path)
    payload = validate_final_quality_gates(_config(), root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["error_count"] == 0
    assert {row["goal_id"]: row["status"] for row in payload["goal_reports"]} == {"G001": "pass", "G008": "pass"}


def test_final_quality_gate_fails_on_incomplete_goal_missing_artifact_and_live_gate(tmp_path):
    _complete_fixture(tmp_path)
    goals = json.loads((tmp_path / ".omx/ultragoal/goals.json").read_text())
    goals["goals"][1]["status"] = "pending"
    write_json(tmp_path / ".omx/ultragoal/goals.json", goals)
    (tmp_path / "artifacts/harness/live_validation.json").unlink()
    payload = validate_final_quality_gates(_config(), root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "goal_status_not_complete" in codes
    assert "missing_required_artifact" in codes
    assert "missing_live_suite_evidence_validation" in codes
    assert "not_all_ultragoal_stories_complete" in codes


def test_final_quality_gate_enforces_json_assertion_values(tmp_path):
    _complete_fixture(tmp_path)
    write_json(tmp_path / "artifacts/harness/live_validation.json", {"quality_gate": {"status": "protocol_ready"}})
    payload = validate_final_quality_gates(_config(), root=tmp_path)
    mismatches = [item for item in payload["findings"] if item["code"] == "json_assertion_mismatch"]
    assert payload["status"] == "fail"
    assert mismatches
    assert mismatches[0]["expected"] == "pass"
    assert mismatches[0]["actual"] == "protocol_ready"


def test_final_quality_gate_enforces_allowed_json_assertion_values(tmp_path):
    _complete_fixture(tmp_path)
    config = _config()
    config["goal_gates"][1]["json_assertions"].append(
        {"path": "artifacts/harness/live_validation.json", "json_path": "decode.num_shards", "in": [16, 64]}
    )
    write_json(tmp_path / "artifacts/harness/live_validation.json", {"quality_gate": {"status": "pass"}, "decode": {"num_shards": 32}})
    payload = validate_final_quality_gates(config, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "json_assertion_not_in_allowed_values" in codes


def test_final_quality_gate_writes_output_without_requiring_self_reference(tmp_path):
    _complete_fixture(tmp_path)
    config = {**_config(), "output_path": "artifacts/reproducibility/final_quality_gate_audit.json"}
    payload = write_final_quality_gate_audit(config, root=tmp_path)
    written = json.loads((tmp_path / "artifacts/reproducibility/final_quality_gate_audit.json").read_text())
    assert payload["status"] == "pass"
    assert written["schema"] == "final_quality_gate_audit.v1"


def test_final_quality_gate_accepts_configured_external_artifact_manifest(tmp_path):
    config = _config()
    config["external_artifact_manifest_path"] = "artifacts/reproducibility/external_artifact_manifest.json"
    config["goal_gates"][0]["required_paths"].append("outputs/huge.jsonl")
    config["goal_gates"][0]["external_artifact_paths"] = ["outputs/huge.jsonl"]
    _complete_fixture(tmp_path)
    entries = json.loads((tmp_path / "artifacts/reproducibility/package_manifest.json").read_text())["entries"]
    entries.append(
        {
            "path": "artifacts/reproducibility/external_artifact_manifest.json",
            "kind": "evidence_artifact",
            "bytes": 2,
            "sha256": "stub",
        }
    )
    write_json(tmp_path / "artifacts/reproducibility/package_manifest.json", {"schema": "repro_package_manifest.v1", "entries": entries})
    write_json(
        tmp_path / "artifacts/reproducibility/external_artifact_manifest.json",
        {
            "schema": "external_artifact_manifest.v1",
            "status": "pass",
            "entries": [
                {
                    "path": "outputs/huge.jsonl",
                    "exists": True,
                    "bytes": 123456,
                    "sha256": "abc123",
                    "storage_uri": "mlxp-pvc://pod/repo/outputs/huge.jsonl",
                }
            ],
        },
    )
    payload = validate_final_quality_gates(config, root=tmp_path)
    assert payload["status"] == "pass"
    g001 = next(row for row in payload["goal_reports"] if row["goal_id"] == "G001")
    huge = next(row for row in g001["artifacts"] if row["path"] == "outputs/huge.jsonl")
    assert huge["external_satisfied"] is True
    assert huge["external"]["sha256"] == "abc123"


def test_final_quality_gate_rejects_weak_external_artifact_manifest_entry(tmp_path):
    config = _config()
    config["external_artifact_manifest_path"] = "artifacts/reproducibility/external_artifact_manifest.json"
    config["goal_gates"][0]["required_paths"].append("outputs/huge.jsonl")
    config["goal_gates"][0]["external_artifact_paths"] = ["outputs/huge.jsonl"]
    _complete_fixture(tmp_path)
    write_json(
        tmp_path / "artifacts/reproducibility/external_artifact_manifest.json",
        {"schema": "external_artifact_manifest.v1", "entries": [{"path": "outputs/huge.jsonl", "exists": True, "bytes": 0}]},
    )
    payload = validate_final_quality_gates(config, root=tmp_path)
    codes = {item["code"] for item in payload["findings"]}
    assert payload["status"] == "fail"
    assert "external_artifact_evidence_missing_or_weak" in codes


def test_final_quality_gate_allows_configured_final_story_in_progress(tmp_path):
    write_json(
        tmp_path / ".omx/ultragoal/goals.json",
        {"goals": [{"id": "G001", "status": "complete"}, {"id": "G009", "status": "in_progress"}]},
    )
    _write(tmp_path / "artifacts/g001.json", "{}")
    _write(tmp_path / "artifacts/g009.json", "{}")
    write_json(tmp_path / "artifacts/reproducibility/claim_boundary_audit.json", {"status": "pass"})
    write_json(tmp_path / "artifacts/harness/live_validation.json", {"quality_gate": {"status": "pass"}})
    write_json(
        tmp_path / "artifacts/reproducibility/package_manifest.json",
        {
            "schema": "repro_package_manifest.v1",
            "entries": [
                {"path": "artifacts/g001.json", "kind": "evidence_artifact", "bytes": 2, "sha256": "stub"},
                {"path": "artifacts/g009.json", "kind": "evidence_artifact", "bytes": 2, "sha256": "stub"},
            ],
        },
    )
    config = {
        **_config(),
        "allow_in_progress_goal_ids": ["G009"],
        "goal_gates": [
            {"id": "G001", "requires_status": "complete", "required_paths": ["artifacts/g001.json"]},
            {"id": "G009", "requires_status": "complete", "required_paths": ["artifacts/g009.json"]},
        ],
    }
    payload = validate_final_quality_gates(config, root=tmp_path)
    assert payload["status"] == "pass"
    assert payload["goal_status_counts"]["in_progress"] == 1


def test_repro_manifest_covers_configured_g006_build_summary():
    text = Path("scripts/build_repro_package_manifest.py").read_text()
    assert '"artifacts/eval/g006_final_artifact_build_summary.json"' in text


def test_repro_manifest_covers_external_and_nested_runtime_artifacts():
    text = Path("scripts/build_repro_package_manifest.py").read_text()
    assert '"artifacts/reproducibility/external_artifact_manifest.json"' in text
    assert '"outputs/fdm_streaming_d2e_full_compact/torch_model/*"' in text
    assert '"artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor*"' in text
    assert '"artifacts/harness/g008_repo_live_suite/**/*"' in text


def test_final_quality_config_does_not_require_self_referential_g009_artifacts():
    text = Path("configs/eval/final_quality_gates.yaml").read_text()
    g009_section = text[text.index('"id": "G009-report-repro-package"') :]
    g009_section = g009_section[: g009_section.index("\n    }") + 6]
    assert "artifacts/reproducibility/final_quality_gate_audit.json" not in g009_section
    assert "artifacts/reproducibility/g009_completion_audit.json" not in g009_section


def test_final_quality_config_preserves_full_d2e_source_tier_gates():
    text = Path("configs/eval/final_quality_gates.yaml").read_text()
    assert '"json_path": "data_universe_counts.source_ids"' in text
    assert '"json_path": "data_universe_counts.resolution_tiers"' in text
    assert '"json_path": "decode_counts_by_source"' in text
    assert '"json_path": "decode_counts_by_resolution_tier"' in text
    assert '"json_path": "source_idm_metadata.path"' in text
    assert '"json_path": "source_ids"' in text
    assert '"json_path": "target_source_ids"' in text
    assert '"d2e_480p": 459' in text
    assert '"d2e_original": 459' in text
    assert '"480p": 459' in text
    assert '"original_fhd_qhd": 459' in text


def test_final_quality_config_requires_g003_gpu_monitor_coverage():
    text = Path("configs/eval/final_quality_gates.yaml").read_text()
    g003_section = text[text.index('"id": "G003-d2e-only-idm"') :]
    g003_section = g003_section[: g003_section.index('"id": "G004-d2e-only-fdm-4xh200"')]
    assert '"json_path": "expected_gpus"' in g003_section
    assert '"json_path": "gpu_monitor_status.covers_expected_gpus"' in g003_section
    assert '"artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv"' in g003_section


def test_final_quality_config_uses_external_manifest_for_pvc_resident_large_jsonls():
    text = Path("configs/eval/final_quality_gates.yaml").read_text()
    assert '"external_artifact_manifest_path": "artifacts/reproducibility/external_artifact_manifest.json"' in text
    assert '"allow_in_progress_goal_ids"' in text
    assert '"G009-report-repro-package"' in text
    for path in [
        "outputs/data/d2e_full_corpus/train_core.jsonl",
        "outputs/data/d2e_full_corpus/target_all_eval.jsonl",
        "outputs/idm_streaming_d2e_full_compact/pseudolabels.jsonl",
        "outputs/idm_streaming_d2e_full_compact/predictions.jsonl",
        "outputs/idm_streaming_d2e_full_compact/fdm_train_core_pseudolabels/pseudolabels.jsonl",
        "outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl",
    ]:
        assert path in text
