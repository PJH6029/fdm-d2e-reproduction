from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from fdm_d2e.io_utils import read_jsonl, sha256_file, write_json
from fdm_d2e.tokenization.actions import token_to_delta_class


DEFAULT_ALLOWED_KEYS = {
    "32",  # Space
    "65",  # A
    "68",  # D
    "69",  # E
    "70",  # F
    "83",  # S
    "87",  # W
}
DEFAULT_ALLOWED_BUTTONS = {"MOUSE_LEFT", "MOUSE_RIGHT", "MOUSE_MIDDLE"}


@dataclass
class DecodedAction:
    step: int
    timestamp_ns: int
    raw_tokens: list[str]
    key_presses: list[str] = field(default_factory=list)
    key_releases: list[str] = field(default_factory=list)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_buttons: list[str] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)
    valid: bool = True
    blocked_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_ns": self.timestamp_ns,
            "raw_tokens": list(self.raw_tokens),
            "key_presses": list(self.key_presses),
            "key_releases": list(self.key_releases),
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "mouse_buttons": list(self.mouse_buttons),
            "unknown_tokens": list(self.unknown_tokens),
            "valid": self.valid,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class RuntimeSafetyConfig:
    require_focus: bool = True
    allowed_window_title_patterns: list[str] = field(default_factory=lambda: ["open-source", "offline", "fdm-adapter-demo"])
    kill_switch_path: str | None = None
    max_actions_per_second: float = 60.0
    max_mouse_delta_per_frame: float = 50.0
    allowed_keys: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_KEYS))
    allowed_mouse_buttons: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_BUTTONS))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RuntimeSafetyConfig":
        safety = dict(config.get("safety", config))
        allowed_keys = safety.get("allowed_keys")
        allowed_buttons = safety.get("allowed_mouse_buttons")
        return cls(
            require_focus=bool(safety.get("require_focus", True)),
            allowed_window_title_patterns=[str(item) for item in safety.get("allowed_window_title_patterns", ["open-source", "offline", "fdm-adapter-demo"])],
            kill_switch_path=str(safety["kill_switch_path"]) if safety.get("kill_switch_path") else None,
            max_actions_per_second=float(safety.get("max_actions_per_second", 60.0)),
            max_mouse_delta_per_frame=float(safety.get("max_mouse_delta_per_frame", 50.0)),
            allowed_keys=set(str(item) for item in allowed_keys) if allowed_keys is not None else set(DEFAULT_ALLOWED_KEYS),
            allowed_mouse_buttons=set(str(item) for item in allowed_buttons) if allowed_buttons is not None else set(DEFAULT_ALLOWED_BUTTONS),
        )


class InputBackend(Protocol):
    def apply(self, action: DecodedAction) -> dict[str, Any]:
        ...


class DryRunInputBackend:
    """No-OS-injection backend used for deterministic replay and tests."""

    def __init__(self) -> None:
        self.applied: list[dict[str, Any]] = []

    def apply(self, action: DecodedAction) -> dict[str, Any]:
        payload = {"status": "applied", "backend": "dry_run", "action": action.as_dict()}
        self.applied.append(payload)
        return payload


class ActionDecoder:
    def __init__(self, safety: RuntimeSafetyConfig | None = None) -> None:
        self.safety = safety or RuntimeSafetyConfig()

    def decode(self, row: dict[str, Any], *, step: int) -> DecodedAction:
        tokens = [str(token) for token in row.get("predicted_tokens", [])]
        action = DecodedAction(step=step, timestamp_ns=int(row.get("timestamp_ns", step)), raw_tokens=tokens)
        for token in tokens:
            if token.startswith("KEY_PRESS_"):
                key = token.removeprefix("KEY_PRESS_")
                if key in self.safety.allowed_keys:
                    action.key_presses.append(key)
                else:
                    action.unknown_tokens.append(token)
            elif token.startswith("KEY_RELEASE_"):
                key = token.removeprefix("KEY_RELEASE_")
                if key in self.safety.allowed_keys:
                    action.key_releases.append(key)
                else:
                    action.unknown_tokens.append(token)
            elif token.startswith("MOUSE_DX_"):
                value = token_to_delta_class(token)
                if value is None:
                    action.unknown_tokens.append(token)
                else:
                    action.mouse_dx += float(value)
            elif token.startswith("MOUSE_DY_"):
                value = token_to_delta_class(token)
                if value is None:
                    action.unknown_tokens.append(token)
                else:
                    action.mouse_dy += float(value)
            elif token.endswith("_DOWN") and token.removesuffix("_DOWN") in self.safety.allowed_mouse_buttons:
                action.mouse_buttons.append(token.removesuffix("_DOWN"))
            elif token in {"NOOP", ""}:
                continue
            else:
                action.unknown_tokens.append(token)
        limit = float(self.safety.max_mouse_delta_per_frame)
        action.mouse_dx = max(-limit, min(limit, action.mouse_dx))
        action.mouse_dy = max(-limit, min(limit, action.mouse_dy))
        if action.unknown_tokens:
            action.valid = False
            action.blocked_reason = "unknown_or_disallowed_tokens"
        return action


class LatencyLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, *, step: int, decode_ms: float, safety_ms: float, backend_ms: float, blocked: bool) -> None:
        total_ms = decode_ms + safety_ms + backend_ms
        self.rows.append(
            {
                "step": step,
                "decode_ms": decode_ms,
                "safety_ms": safety_ms,
                "backend_ms": backend_ms,
                "total_ms": total_ms,
                "blocked": blocked,
            }
        )

    @staticmethod
    def _percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return ordered[idx]

    def summary(self) -> dict[str, Any]:
        totals = [float(row["total_ms"]) for row in self.rows]
        return {
            "schema": "runtime_latency_summary.v1",
            "num_actions": len(self.rows),
            "p50_ms": statistics.median(totals) if totals else None,
            "p95_ms": self._percentile(totals, 0.95),
            "max_ms": max(totals) if totals else None,
            "blocked_actions": sum(1 for row in self.rows if row["blocked"]),
            "rows": self.rows,
        }


class SafeActionAdapter:
    """Safety wrapper for future open-source/offline game input backends."""

    def __init__(
        self,
        *,
        backend: InputBackend | None = None,
        safety: RuntimeSafetyConfig | None = None,
        focus_title_provider: Callable[[], str] | None = None,
    ) -> None:
        self.backend = backend or DryRunInputBackend()
        self.safety = safety or RuntimeSafetyConfig()
        self.focus_title_provider = focus_title_provider or (lambda: "fdm-adapter-demo open-source offline")
        self._last_action_ts_ns: int | None = None

    def _focus_ok(self) -> bool:
        if not self.safety.require_focus:
            return True
        title = self.focus_title_provider()
        return any(re.search(pattern, title, flags=re.IGNORECASE) for pattern in self.safety.allowed_window_title_patterns)

    def _rate_ok(self, timestamp_ns: int) -> bool:
        if self._last_action_ts_ns is None:
            return True
        min_interval_ns = int(1_000_000_000 / max(1e-9, float(self.safety.max_actions_per_second)))
        return (timestamp_ns - self._last_action_ts_ns) >= min_interval_ns

    def _kill_switch_active(self) -> bool:
        return bool(self.safety.kill_switch_path and Path(self.safety.kill_switch_path).exists())

    def apply(self, action: DecodedAction) -> dict[str, Any]:
        if not action.valid:
            return {"status": "blocked", "reason": action.blocked_reason or "invalid_action", "action": action.as_dict()}
        if self._kill_switch_active():
            action.valid = False
            action.blocked_reason = "kill_switch_active"
            return {"status": "blocked", "reason": "kill_switch_active", "action": action.as_dict()}
        if not self._focus_ok():
            action.valid = False
            action.blocked_reason = "focus_guard_failed"
            return {"status": "blocked", "reason": "focus_guard_failed", "action": action.as_dict()}
        if not self._rate_ok(action.timestamp_ns):
            action.valid = False
            action.blocked_reason = "rate_limited"
            return {"status": "blocked", "reason": "rate_limited", "action": action.as_dict()}
        result = self.backend.apply(action)
        self._last_action_ts_ns = action.timestamp_ns
        return result


def run_replay_adapter(
    config: dict[str, Any],
    *,
    focus_title_provider: Callable[[], str] | None = None,
    backend: InputBackend | None = None,
) -> dict[str, Any]:
    predictions_path = Path(config["predictions_path"])
    predictions = read_jsonl(predictions_path)
    limit = int(config.get("action_limit", 256))
    safety = RuntimeSafetyConfig.from_config(config)
    decoder = ActionDecoder(safety)
    adapter = SafeActionAdapter(backend=backend, safety=safety, focus_title_provider=focus_title_provider)
    latency = LatencyLogger()
    results: list[dict[str, Any]] = []
    for step, row in enumerate(predictions[:limit]):
        t0 = time.perf_counter()
        action = decoder.decode(row, step=step)
        t1 = time.perf_counter()
        result = adapter.apply(action)
        t2 = time.perf_counter()
        results.append(result)
        latency.record(
            step=step,
            decode_ms=(t1 - t0) * 1000.0,
            safety_ms=0.0,
            backend_ms=(t2 - t1) * 1000.0,
            blocked=result.get("status") == "blocked",
        )
    blocked = [row for row in results if row.get("status") == "blocked"]
    output = {
        "schema": "runtime_replay_adapter.v1",
        "mode": "deterministic_replay_dry_run" if isinstance(adapter.backend, DryRunInputBackend) else "backend_adapter",
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "num_input_predictions": len(predictions),
        "num_actions": len(results),
        "blocked_actions": len(blocked),
        "applied_actions": len(results) - len(blocked),
        "safety": {
            "require_focus": safety.require_focus,
            "allowed_window_title_patterns": safety.allowed_window_title_patterns,
            "kill_switch_path": safety.kill_switch_path,
            "max_actions_per_second": safety.max_actions_per_second,
            "max_mouse_delta_per_frame": safety.max_mouse_delta_per_frame,
        },
        "latency": latency.summary(),
        "action_results": results,
        "adapter_targets": list(config.get("adapter_targets", [])),
        "notes": "Safe replay SDK evidence only; no OS-level input injection or commercial-game control claim.",
    }
    if config.get("output_path"):
        write_json(config["output_path"], output)
    return output
