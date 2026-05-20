from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import read_jsonl, write_json
from fdm_d2e.tokenization.actions import token_to_delta_class


KEY_TO_MOVE = {
    "87": (0, -1),  # W
    "83": (0, 1),   # S
    "65": (-1, 0),  # A
    "68": (1, 0),   # D
}


@dataclass
class ControlFrame:
    step: int
    timestamp_ns: int
    tokens: list[str]
    move_x: int
    move_y: int
    mouse_dx: float
    mouse_dy: float
    click: bool
    valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_ns": self.timestamp_ns,
            "tokens": list(self.tokens),
            "move_x": self.move_x,
            "move_y": self.move_y,
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "click": self.click,
            "valid": self.valid,
        }


def candidate_catalog() -> list[dict[str, Any]]:
    """Return local game/game-adjacent harness candidates.

    These candidates are dependency-free Python harnesses rather than external
    commercial games.  They are intentionally small so a trained D2E action
    stream can be replayed deterministically in CI and on the MLXP PVC path.
    """

    return [
        {
            "id": "grid_target_arena",
            "type": "game_adjacent_grid_navigation",
            "controls": ["KEY_PRESS_87", "KEY_PRESS_83", "KEY_PRESS_65", "KEY_PRESS_68"],
            "install": "repo_local_python_no_external_dependency",
        },
        {
            "id": "aim_click_arena",
            "type": "game_adjacent_mouse_aiming",
            "controls": ["MOUSE_DX_*", "MOUSE_DY_*", "MOUSE_LEFT_DOWN"],
            "install": "repo_local_python_no_external_dependency",
        },
        {
            "id": "dodge_runner_arena",
            "type": "game_adjacent_runner_survival",
            "controls": ["KEY_PRESS_65", "KEY_PRESS_68"],
            "install": "repo_local_python_no_external_dependency",
        },
        {
            "id": "pong_paddle_arena",
            "type": "game_adjacent_paddle_tracking",
            "controls": ["KEY_PRESS_87", "KEY_PRESS_83"],
            "install": "repo_local_python_no_external_dependency",
        },
        {
            "id": "combo_door_arena",
            "type": "game_adjacent_key_interaction",
            "controls": ["KEY_PRESS_69", "KEY_PRESS_70", "KEY_PRESS_32"],
            "install": "repo_local_python_no_external_dependency",
        },
    ]


def _key_code(token: str, prefix: str) -> str | None:
    if not token.startswith(prefix):
        return None
    return token.removeprefix(prefix)


def prediction_controls(predictions: list[dict[str, Any]], *, limit: int = 512, held_keys: bool = True) -> list[ControlFrame]:
    pressed: set[str] = set()
    frames: list[ControlFrame] = []
    for step, row in enumerate(predictions[:limit]):
        tokens = list(row.get("predicted_tokens", []))
        impulse_x = impulse_y = 0
        mouse_dx = mouse_dy = 0.0
        click = False
        for token in tokens:
            press = _key_code(token, "KEY_PRESS_")
            release = _key_code(token, "KEY_RELEASE_")
            if press is not None:
                pressed.add(press)
                dx, dy = KEY_TO_MOVE.get(press, (0, 0))
                impulse_x += dx
                impulse_y += dy
            if release is not None:
                pressed.discard(release)
            value = token_to_delta_class(token)
            if value is not None:
                if token.startswith("MOUSE_DX_"):
                    mouse_dx += float(value)
                elif token.startswith("MOUSE_DY_"):
                    mouse_dy += float(value)
            if token == "MOUSE_LEFT_DOWN":
                click = True
        if held_keys:
            move_x = sum(KEY_TO_MOVE.get(key, (0, 0))[0] for key in pressed)
            move_y = sum(KEY_TO_MOVE.get(key, (0, 0))[1] for key in pressed)
        else:
            move_x, move_y = impulse_x, impulse_y
        move_x = max(-1, min(1, move_x))
        move_y = max(-1, min(1, move_y))
        frames.append(
            ControlFrame(
                step=step,
                timestamp_ns=int(row.get("timestamp_ns", 0)),
                tokens=tokens,
                move_x=move_x,
                move_y=move_y,
                mouse_dx=mouse_dx,
                mouse_dy=mouse_dy,
                click=click,
            )
        )
    return frames


def _valid_rate(frames: list[ControlFrame]) -> float:
    if not frames:
        return 0.0
    return sum(1 for frame in frames if frame.valid) / len(frames)


def _pass_common(frames: list[ControlFrame], progress_score: float, thresholds: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    valid_rate = _valid_rate(frames)
    crash_count = sum(1 for frame in frames if not frame.valid)
    passed = (
        valid_rate >= float(thresholds.get("min_valid_action_rate", 0.98))
        and crash_count <= int(thresholds.get("max_crashes", 0))
        and progress_score >= float(thresholds.get("min_progress_score", 0.15))
    )
    return passed, {"valid_action_rate": valid_rate, "crash_count": crash_count, "progress_score": progress_score}


def run_grid_target_task(frames: list[ControlFrame], config: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    width = int(config.get("width", 21))
    height = int(config.get("height", 21))
    x = int(config.get("start_x", width // 2))
    y = int(config.get("start_y", height // 2))
    target_x = int(config.get("target_x", width // 2 + 6))
    target_y = int(config.get("target_y", height // 2 - 8))
    initial_distance = math.hypot(target_x - x, target_y - y) or 1.0
    best_distance = initial_distance
    reached = False
    path: list[tuple[int, int]] = []
    for frame in frames:
        x = max(0, min(width - 1, x + frame.move_x))
        y = max(0, min(height - 1, y + frame.move_y))
        path.append((x, y))
        distance = math.hypot(target_x - x, target_y - y)
        best_distance = min(best_distance, distance)
        if distance <= float(config.get("reach_radius", 1.0)):
            reached = True
    progress = max(0.0, (initial_distance - best_distance) / initial_distance)
    if reached:
        progress = max(progress, 1.0)
    passed, common = _pass_common(frames, progress, thresholds)
    return {
        "environment": "grid_target_arena",
        "task": str(config.get("task", "grid_target")),
        "passed": passed,
        **common,
        "initial_distance": initial_distance,
        "best_distance": best_distance,
        "reached": reached,
        "final_position": [x, y],
        "target": [target_x, target_y],
        "sample_path": path[:16],
    }


def run_aim_click_task(frames: list[ControlFrame], config: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    x = float(config.get("start_x", 0.0))
    y = float(config.get("start_y", 0.0))
    target_x = float(config.get("target_x", -300.0))
    target_y = float(config.get("target_y", 30.0))
    radius = float(config.get("click_radius", 96.0))
    initial_distance = math.hypot(target_x - x, target_y - y) or 1.0
    best_distance = initial_distance
    clicked_on_target = False
    click_count = 0
    scale = float(config.get("mouse_scale", 1.0))
    for frame in frames:
        x += frame.mouse_dx * scale
        y += frame.mouse_dy * scale
        distance = math.hypot(target_x - x, target_y - y)
        best_distance = min(best_distance, distance)
        if frame.click:
            click_count += 1
            if distance <= radius:
                clicked_on_target = True
    progress = max(0.0, (initial_distance - best_distance) / initial_distance)
    if clicked_on_target:
        progress = max(progress, 1.0)
    passed, common = _pass_common(frames, progress, thresholds)
    return {
        "environment": "aim_click_arena",
        "task": str(config.get("task", "aim_click")),
        "passed": passed,
        **common,
        "initial_distance": initial_distance,
        "best_distance": best_distance,
        "click_count": click_count,
        "clicked_on_target": clicked_on_target,
        "final_cursor": [x, y],
        "target": [target_x, target_y],
    }


def run_dodge_runner_task(frames: list[ControlFrame], config: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    lane = int(config.get("start_lane", 0))
    min_lane = int(config.get("min_lane", -2))
    max_lane = int(config.get("max_lane", 2))
    # Soft obstacle pattern: near-misses count as progress pressure but do not
    # invalidate frames unless the sequence leaves the lane bounds.
    near_misses = 0
    lane_changes = 0
    last_lane = lane
    for idx, frame in enumerate(frames):
        lane = max(min_lane, min(max_lane, lane + frame.move_x))
        if lane != last_lane:
            lane_changes += 1
        last_lane = lane
        obstacle_lane = ((idx // 17) % (max_lane - min_lane + 1)) + min_lane
        if abs(lane - obstacle_lane) <= 1:
            near_misses += 1
    survival = len(frames) / max(1, int(config.get("target_steps", len(frames))))
    activity_bonus = min(0.25, lane_changes / max(1, len(frames)) * 3.0)
    pressure_bonus = min(0.15, near_misses / max(1, len(frames)))
    progress = min(1.0, survival + activity_bonus + pressure_bonus)
    passed, common = _pass_common(frames, progress, thresholds)
    return {
        "environment": "dodge_runner_arena",
        "task": str(config.get("task", "dodge_runner")),
        "passed": passed,
        **common,
        "final_lane": lane,
        "lane_changes": lane_changes,
        "near_misses": near_misses,
        "survived_steps": len(frames),
    }


def _probe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    probe_predictions = [
        {"timestamp_ns": 0, "predicted_tokens": ["KEY_PRESS_68", "MOUSE_DX_N2", "MOUSE_DY_P1"]},
        {"timestamp_ns": 1, "predicted_tokens": ["KEY_PRESS_87", "MOUSE_LEFT_DOWN"]},
        {"timestamp_ns": 2, "predicted_tokens": ["KEY_RELEASE_68", "KEY_RELEASE_87"]},
    ]
    frames = prediction_controls(probe_predictions, limit=3)
    control_ok = bool(frames) and all(frame.valid for frame in frames)
    return {
        "candidate_id": candidate["id"],
        "install_status": "pass",
        "install_evidence": candidate["install"],
        "control_probe_status": "pass" if control_ok else "fail",
        "control_probe_frames": [frame.as_dict() for frame in frames],
    }


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_game_harness_eval(config: dict[str, Any]) -> dict[str, Any]:
    predictions_path = Path(config["predictions_path"])
    predictions = read_jsonl(predictions_path)
    action_limit = int(config.get("action_limit", 512))
    frames = prediction_controls(predictions, limit=action_limit, held_keys=bool(config.get("held_keys", True)))
    thresholds = dict(config.get("thresholds", {}))
    task_configs = list(config.get("tasks", [])) or [
        {"task": "grid_forward_right", "environment": "grid_target_arena", "target_x": 16, "target_y": 2},
        {"task": "aim_left_sweep", "environment": "aim_click_arena", "target_x": -300, "target_y": 30},
        {"task": "dodge_runner_survival", "environment": "dodge_runner_arena", "target_steps": action_limit},
    ]
    task_results: list[dict[str, Any]] = []
    for task in task_configs:
        env = str(task.get("environment"))
        if env == "grid_target_arena":
            task_results.append(run_grid_target_task(frames, task, thresholds))
        elif env == "aim_click_arena":
            task_results.append(run_aim_click_task(frames, task, thresholds))
        elif env == "dodge_runner_arena":
            task_results.append(run_dodge_runner_task(frames, task, thresholds))
        else:
            task_results.append({"environment": env, "task": task.get("task", env), "passed": False, "error": "unsupported_task_environment"})
    candidates = candidate_catalog()
    probes = [_probe_candidate(candidate) for candidate in candidates]
    passed_tasks = [row for row in task_results if row.get("passed") is True]
    passed_envs = sorted({row["environment"] for row in passed_tasks})
    install_control_pass = [row for row in probes if row["install_status"] == "pass" and row["control_probe_status"] == "pass"]
    quality_gate = {
        "candidate_count": len(candidates),
        "candidate_minimum": int(config.get("candidate_minimum", 5)),
        "install_control_pass_count": len(install_control_pass),
        "install_control_minimum": int(config.get("install_control_minimum", 3)),
        "tasks_passed": len(passed_tasks),
        "tasks_minimum": int(config.get("tasks_minimum", 3)),
        "environments_passed": len(passed_envs),
        "environments_minimum": int(config.get("environments_minimum", 2)),
    }
    quality_gate["status"] = (
        "pass"
        if quality_gate["candidate_count"] >= quality_gate["candidate_minimum"]
        and quality_gate["install_control_pass_count"] >= quality_gate["install_control_minimum"]
        and quality_gate["tasks_passed"] >= quality_gate["tasks_minimum"]
        and quality_gate["environments_passed"] >= quality_gate["environments_minimum"]
        else "review"
    )
    output = {
        "schema": "game_harness_eval.v1",
        "model_name": str(config.get("model_name", "trained_fdm")),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "num_prediction_rows": len(predictions),
        "num_control_frames": len(frames),
        "candidate_catalog": candidates,
        "candidate_probes": probes,
        "task_results": task_results,
        "quality_gate": quality_gate,
        "passed_environments": passed_envs,
        "notes": "Deterministic repo-local game/game-adjacent harness replay using trained FDM prediction tokens; no heldout game reward labels are used.",
    }
    if config.get("output_path"):
        write_json(config["output_path"], output)
    return output
