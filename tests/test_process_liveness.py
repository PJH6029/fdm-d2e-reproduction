from __future__ import annotations

from pathlib import Path

from fdm_d2e.process_liveness import pid_is_zombie, pid_running, pid_state


def _write_stat(proc_root: Path, pid: int, state: str) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    # Minimal /proc/<pid>/stat shape: "pid (comm) state ppid ..."
    (proc_dir / "stat").write_text(f"{pid} (uv) {state} 1 1 1\n", encoding="utf-8")


def test_pid_state_reads_linux_proc_state(tmp_path: Path):
    _write_stat(tmp_path, 123, "Z")
    assert pid_state(123, proc_root=tmp_path) == "Z"
    assert pid_is_zombie(123, proc_root=tmp_path) is True
    assert pid_state(999, proc_root=tmp_path) is None


def test_pid_running_treats_proc_zombie_as_exited(tmp_path: Path, monkeypatch):
    _write_stat(tmp_path, 123, "Z")
    monkeypatch.setattr("fdm_d2e.process_liveness.os.kill", lambda pid, sig: None)
    assert pid_running(123, proc_root=tmp_path) is False


def test_pid_running_requires_proc_entry_when_proc_exists(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("fdm_d2e.process_liveness.os.kill", lambda pid, sig: None)
    assert pid_running(123, proc_root=tmp_path) is False


def test_pid_running_accepts_non_zombie_live_process(tmp_path: Path, monkeypatch):
    _write_stat(tmp_path, 123, "S")
    monkeypatch.setattr("fdm_d2e.process_liveness.os.kill", lambda pid, sig: None)
    assert pid_running(123, proc_root=tmp_path) is True
