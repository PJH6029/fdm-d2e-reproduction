from __future__ import annotations

import os
from pathlib import Path


def pid_state(pid: int, *, proc_root: Path = Path("/proc")) -> str | None:
    """Return Linux process state from /proc, or None if unavailable/exited."""

    if pid <= 0 or not proc_root.exists():
        return None
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8", errors="ignore")
        tail = stat.rsplit(")", 1)[1].strip().split()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError):
        return None
    return tail[0] if tail else None


def pid_is_zombie(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    return pid_state(pid, proc_root=proc_root) == "Z"


def pid_running(pid: int | None, *, proc_root: Path = Path("/proc")) -> bool:
    """Best-effort liveness that treats defunct/zombie processes as exited."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if proc_root.exists():
        state = pid_state(pid, proc_root=proc_root)
        if state is None:
            return False
        if state == "Z":
            return False
    return True
