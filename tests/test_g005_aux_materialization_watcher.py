from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fdm_d2e.io_utils import write_json
from watch_g005_aux_materialization import _pid_running, _tree_status, watch


def _args(root: Path, **overrides) -> Namespace:
    data = {
        "root": str(root),
        "output": "artifacts/aux/watcher.json",
        "allow_fail": False,
        "once": True,
        "poll_seconds": 0.01,
        "max_wait_seconds": -1.0,
        "pid_file": "outputs/cluster/materialize.pid",
        "watcher_pid_file": "outputs/cluster/materialize_watcher.pid",
        "replace_existing_watcher": False,
        "materialization_summary": "artifacts/aux/materialize_summary.json",
        "materialization_log": "artifacts/aux/materialize.log",
        "namespace_root": "outputs/aux",
        "examples_root": "outputs/aux_examples",
        "required_splits": ["train", "val", "test"],
        "max_files": None,
        "max_examples_per_source": None,
        "aux_candidates": "artifacts/sources/aux.json",
        "action_registry": "artifacts/aux/action_registry.json",
        "source_evidence_output": "artifacts/aux/source_evidence.json",
        "integrity_output": "artifacts/aux/integrity.json",
        "aux_examples_output": "artifacts/aux/examples.json",
        "runtime_env_output": "artifacts/aux/runtime_env.json",
        "eval_manifest_hashes": "artifacts/aux/eval_hashes.json",
        "namespace_manifest_output": "artifacts/aux/namespace.json",
        "g005_launch_readiness_output": "artifacts/aux/launch.json",
        "g005_completion_config": "configs/eval/g005.json",
        "g003_audit": "artifacts/idm/g003_audit.json",
        "g004_audit": "artifacts/fdm/g004_audit.json",
        "g005_pid_file": "outputs/cluster/g005_train.pid",
        "allow_overwrite_g005_run_summary": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_watcher_waits_while_materializer_runs(tmp_path: Path):
    pid_path = tmp_path / "outputs/cluster/materialize.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    calls = {"integrity": 0, "source": 0, "aux": 0, "runtime": 0, "namespace": 0, "plan": 0}
    payload = watch(
        _args(tmp_path),
        integrity_func=lambda ns: calls.__setitem__("integrity", calls["integrity"] + 1) or {"status": "pass"},
        source_evidence_func=lambda ns: calls.__setitem__("source", calls["source"] + 1) or {"status": "pass"},
        aux_examples_func=lambda ns: calls.__setitem__("aux", calls["aux"] + 1) or {"status": "pass"},
        runtime_env_func=lambda ns: calls.__setitem__("runtime", calls["runtime"] + 1) or {"status": "pass"},
        namespace_func=lambda ns, root: calls.__setitem__("namespace", calls["namespace"] + 1) or {"completion_ready": True},
        plan_func=lambda ns: calls.__setitem__("plan", calls["plan"] + 1) or {"status": "ready"},
    )
    assert payload["status"] == "waiting_active_materialization"
    assert payload["materialization"]["pid_running"] is True
    assert calls == {"integrity": 0, "source": 0, "aux": 0, "runtime": 0, "namespace": 0, "plan": 0}
    assert not (tmp_path / "outputs/cluster/materialize_watcher.pid").exists()


def test_watcher_builds_evidence_then_reports_g005_not_ready(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass", "execute": True, "executions": []})
    received: dict[str, Namespace] = {}

    def fake_integrity(ns: Namespace) -> dict:
        received["integrity"] = ns
        return {"status": "pass", "error_count": 0}

    def fake_source(ns: Namespace) -> dict:
        received["source"] = ns
        return {"status": "pass", "error_count": 0}

    def fake_aux(ns: Namespace) -> dict:
        received["aux"] = ns
        return {"status": "pass", "error_count": 0}

    def fake_runtime(ns: Namespace) -> dict:
        received["runtime"] = ns
        return {"status": "pass", "error_count": 0}

    def fake_namespace(ns: Namespace, root: Path) -> dict:
        received["namespace"] = ns
        return {"completion_ready": True}

    def fake_plan(ns: Namespace) -> dict:
        received["plan"] = ns
        return {"status": "blocked", "findings": [{"code": "prereq"}]}

    payload = watch(
        _args(tmp_path),
        integrity_func=fake_integrity,
        source_evidence_func=fake_source,
        aux_examples_func=fake_aux,
        runtime_env_func=fake_runtime,
        namespace_func=fake_namespace,
        plan_func=fake_plan,
    )
    assert payload["status"] == "g005_launch_not_ready"
    assert payload["materialization_integrity_status"] == "pass"
    assert payload["source_evidence_status"] == "pass"
    assert payload["aux_examples_status"] == "pass"
    assert payload["runtime_env_status"] == "pass"
    assert payload["namespace_completion_ready"] is True
    assert payload["g005_launch_plan_finding_count"] == 1
    assert received["integrity"].output == "artifacts/aux/integrity.json"
    assert received["source"].namespace_root == "outputs/aux"
    assert received["aux"].examples_root == "outputs/aux_examples"
    assert received["runtime"].output == "artifacts/aux/runtime_env.json"
    assert received["plan"].source_evidence == ["artifacts/aux/source_evidence.json"]
    assert received["plan"].eval_manifest_hashes == "artifacts/aux/eval_hashes.json"
    assert received["plan"].require_namespace_ready is True


def test_watcher_blocks_when_materialization_summary_is_missing_or_failed(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "blocked", "error_count": 1})
    calls = {"source": 0}
    payload = watch(_args(tmp_path), source_evidence_func=lambda ns: calls.__setitem__("source", calls["source"] + 1) or {"status": "pass"})
    assert payload["status"] == "materialization_not_pass"
    assert payload["findings"][0]["code"] == "materialization_not_pass"
    assert calls == {"source": 0}


def test_watcher_blocks_when_materialization_integrity_fails(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass", "error_count": 0})
    calls = {"source": 0}
    payload = watch(
        _args(tmp_path),
        integrity_func=lambda ns: {"status": "blocked", "error_count": 3},
        source_evidence_func=lambda ns: calls.__setitem__("source", calls["source"] + 1) or {"status": "pass"},
    )
    assert payload["status"] == "materialization_integrity_not_pass"
    assert payload["findings"][0]["code"] == "materialization_integrity_not_pass"
    assert payload["materialization_integrity_error_count"] == 3
    assert calls == {"source": 0}


def test_watcher_blocks_when_source_evidence_fails(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass"})
    payload = watch(
        _args(tmp_path),
        integrity_func=lambda ns: {"status": "pass", "error_count": 0},
        source_evidence_func=lambda ns: {"status": "blocked", "error_count": 2},
        namespace_func=lambda ns, root: {"completion_ready": True},
        plan_func=lambda ns: {"status": "ready"},
    )
    assert payload["status"] == "source_evidence_not_pass"
    assert payload["source_evidence_error_count"] == 2


def test_watcher_blocks_when_aux_examples_fail(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass"})
    payload = watch(
        _args(tmp_path),
        integrity_func=lambda ns: {"status": "pass", "error_count": 0},
        source_evidence_func=lambda ns: {"status": "pass", "error_count": 0},
        aux_examples_func=lambda ns: {"status": "blocked", "error_count": 4},
        runtime_env_func=lambda ns: {"status": "pass", "error_count": 0},
        namespace_func=lambda ns, root: {"completion_ready": True},
        plan_func=lambda ns: {"status": "ready"},
    )
    assert payload["status"] == "aux_examples_not_pass"
    assert payload["aux_examples_error_count"] == 4


def test_watcher_blocks_when_runtime_env_fails(tmp_path: Path):
    write_json(tmp_path / "artifacts/aux/materialize_summary.json", {"status": "pass"})
    payload = watch(
        _args(tmp_path),
        integrity_func=lambda ns: {"status": "pass", "error_count": 0},
        source_evidence_func=lambda ns: {"status": "pass", "error_count": 0},
        aux_examples_func=lambda ns: {"status": "pass", "error_count": 0},
        runtime_env_func=lambda ns: {"status": "blocked", "error_count": 1},
        namespace_func=lambda ns, root: {"completion_ready": True},
        plan_func=lambda ns: {"status": "ready"},
    )
    assert payload["status"] == "runtime_env_not_pass"
    assert payload["runtime_env_error_count"] == 1


def test_watcher_refuses_duplicate_running_watcher(tmp_path: Path):
    watcher = tmp_path / "outputs/cluster/materialize_watcher.pid"
    watcher.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        watcher.write_text(f"{proc.pid}\n", encoding="utf-8")
        payload = watch(_args(tmp_path))
        assert payload["status"] == "duplicate_watcher_running"
        assert payload["existing_pid"] == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_watcher_tree_status_tolerates_files_removed_during_download_scan(tmp_path: Path, monkeypatch):
    namespace = tmp_path / "outputs/aux"
    namespace.mkdir(parents=True)
    stable = namespace / "stable.bin"
    vanished = namespace / ".cache/huggingface/download/tmp.incomplete"
    stable.write_bytes(b"x" * 9)

    monkeypatch.setattr("watch_g005_aux_materialization._iter_tree_files", lambda path: [stable, vanished])

    payload = _tree_status(tmp_path, "outputs/aux")
    assert payload["file_count"] == 1
    assert payload["bytes"] == 9
    assert payload["transient_missing_file_count"] == 1


def test_watcher_pid_running_treats_zombie_as_exited(monkeypatch):
    monkeypatch.setattr("watch_g005_aux_materialization.pid_running", lambda pid: False)
    assert _pid_running(12345) is False
