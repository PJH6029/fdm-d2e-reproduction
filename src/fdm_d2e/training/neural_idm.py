from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import bin_delta, token_to_delta_class


_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>\.ppm)$")


def _summary_features(row: dict[str, Any]) -> list[float]:
    frame = row.get("frame", {})
    features = [float(v) for v in frame.get("features", [])]
    while len(features) < 5:
        features.append(0.0)
    next_features = [float(v) for v in row.get("next_frame_features", [])]
    while len(next_features) < 5:
        next_features.append(0.0)
    delta_features = [float(v) for v in row.get("frame_delta_features", [])]
    while len(delta_features) < 5:
        delta_features.append(0.0)
    return features[:5] + next_features[:5] + delta_features[:5] + [float(row.get("bin_index", 0)) / 100.0]


def _temporal_basis_features(row: dict[str, Any]) -> list[float]:
    bin_index = float(row.get("bin_index", 0))
    values: list[float] = []
    for period in (2.0, 3.0, 4.0, 5.0, 8.0, 16.0):
        phase = 2.0 * math.pi * bin_index / period
        values.extend([math.sin(phase), math.cos(phase)])
    return values


def _read_ppm_tokens(payload: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    idx = 0
    while len(tokens) < 4:
        while idx < len(payload) and payload[idx] in b" \t\r\n":
            idx += 1
        if idx < len(payload) and payload[idx] == ord("#"):
            while idx < len(payload) and payload[idx] not in b"\r\n":
                idx += 1
            continue
        start = idx
        while idx < len(payload) and payload[idx] not in b" \t\r\n":
            idx += 1
        if start == idx:
            raise ValueError("invalid PPM header")
        tokens.append(payload[start:idx])
    while idx < len(payload) and payload[idx] in b" \t\r\n":
        idx += 1
    return tokens, idx


@lru_cache(maxsize=8192)
def _ppm_grid_and_luma(path: str, grid_size: int = 4, luma_size: int = 16) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return compact frame features from a raw 8-bit P6 PPM frame.

    D2E sample decode writes 64x64 PPM frames.  Loading those pixels directly in
    the MLP would add a dependency and a large input surface; this helper keeps a
    deterministic, dependency-free inverse-dynamics signal by caching only small
    RGB grid averages plus a luma grid used for coarse frame-shift features.
    """

    payload = Path(path).read_bytes()
    tokens, offset = _read_ppm_tokens(payload)
    if tokens[0] != b"P6":
        raise ValueError(f"expected P6 PPM frame: {path}")
    width, height, max_value = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if width <= 0 or height <= 0 or max_value <= 0 or max_value > 255:
        raise ValueError(f"unsupported PPM geometry/header: {path}")
    expected = width * height * 3
    pixels = payload[offset : offset + expected]
    if len(pixels) != expected:
        raise ValueError(f"truncated PPM payload: {path}")

    grid_sums = [[0.0, 0.0, 0.0, 0.0] for _ in range(grid_size * grid_size)]
    luma_sums = [[0.0, 0.0] for _ in range(luma_size * luma_size)]
    for y in range(height):
        gy = min(grid_size - 1, y * grid_size // height)
        ly = min(luma_size - 1, y * luma_size // height)
        for x in range(width):
            gx = min(grid_size - 1, x * grid_size // width)
            lx = min(luma_size - 1, x * luma_size // width)
            base = (y * width + x) * 3
            r = pixels[base] / max_value
            g = pixels[base + 1] / max_value
            b = pixels[base + 2] / max_value
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            grid_bucket = grid_sums[gy * grid_size + gx]
            grid_bucket[0] += r
            grid_bucket[1] += g
            grid_bucket[2] += b
            grid_bucket[3] += 1.0
            luma_bucket = luma_sums[ly * luma_size + lx]
            luma_bucket[0] += luma
            luma_bucket[1] += 1.0

    grid: list[float] = []
    for r, g, b, count in grid_sums:
        denom = count or 1.0
        grid.extend([r / denom, g / denom, b / denom])
    luma = tuple(total / (count or 1.0) for total, count in luma_sums)
    return tuple(grid), luma


def _frame_path_with_offset(row: dict[str, Any], offset: int) -> str | None:
    path = str(row.get("frame", {}).get("path", ""))
    if not path:
        return None
    frame_path = Path(path)
    match = _FRAME_RE.match(frame_path.name)
    if not match:
        return None
    frame_number = int(match.group("number")) + int(offset)
    if frame_number < 0:
        return None
    width = len(match.group("number"))
    shifted_name = f"{match.group('prefix')}{frame_number:0{width}d}{match.group('suffix')}"
    shifted_path = frame_path.with_name(shifted_name)
    return str(shifted_path) if shifted_path.exists() else None


def _next_frame_path(row: dict[str, Any]) -> str | None:
    return _frame_path_with_offset(row, 1)


def _frame_pair_features(
    row: dict[str, Any],
    *,
    grid_size: int = 4,
    luma_size: int = 16,
    shift_surface: bool = False,
) -> list[float]:
    current_path = str(row.get("frame", {}).get("path", ""))
    grid_len = grid_size * grid_size * 3
    shift_len = 16 if shift_surface else 4
    luma_len = luma_size * luma_size
    if not current_path or not Path(current_path).exists():
        return [0.0] * (grid_len * 3 + shift_len)
    try:
        cur_grid, cur_luma = _ppm_grid_and_luma(current_path, grid_size, luma_size)
    except (OSError, ValueError):
        return [0.0] * (grid_len * 3 + shift_len)
    next_path = _next_frame_path(row)
    if next_path:
        try:
            next_grid, next_luma = _ppm_grid_and_luma(next_path, grid_size, luma_size)
        except (OSError, ValueError):
            next_grid = tuple(0.0 for _ in range(grid_len))
            next_luma = tuple(0.0 for _ in range(luma_len))
    else:
        next_grid = tuple(0.0 for _ in range(grid_len))
        next_luma = tuple(0.0 for _ in range(luma_len))
    delta_grid = [float(n - c) for c, n in zip(cur_grid, next_grid)]
    shift = (
        _coarse_shift_surface_features(cur_luma, next_luma, luma_size=luma_size)
        if shift_surface
        else _coarse_shift_features(cur_luma, next_luma, luma_size=luma_size)
    )
    return list(cur_grid) + list(next_grid) + delta_grid + shift


def _compact_grid_luma(row: dict[str, Any], *, grid_size: int = 8, luma_size: int = 16) -> tuple[list[float], list[float], list[float], list[float]] | None:
    frame = row.get("frame", {})
    grid_key = f"grid{grid_size}"
    luma_key = f"luma{luma_size}"
    cur_grid = [float(value) for value in frame.get(grid_key, [])]
    next_grid = [float(value) for value in row.get(f"next_frame_{grid_key}", [])]
    cur_luma = [float(value) for value in frame.get(luma_key, [])]
    next_luma = [float(value) for value in row.get(f"next_frame_{luma_key}", [])]
    expected_grid = grid_size * grid_size * 3
    expected_luma = luma_size * luma_size
    if (
        len(cur_grid) == expected_grid
        and len(next_grid) == expected_grid
        and len(cur_luma) == expected_luma
        and len(next_luma) == expected_luma
    ):
        return cur_grid, next_grid, cur_luma, next_luma
    return None


def _compact_frame_pair_features(
    row: dict[str, Any],
    *,
    grid_size: int = 8,
    luma_size: int = 16,
    shift_surface: bool = True,
) -> list[float]:
    """Frame-pair features from compact JSONL fields, with PPM fallback.

    Full-corpus D2E extraction stores compact grid/luma arrays instead of
    long-lived per-frame PPM files.  The fallback keeps existing sample/shooter
    configs compatible.
    """

    compact = _compact_grid_luma(row, grid_size=grid_size, luma_size=luma_size)
    if compact is None:
        return _frame_pair_features(row, grid_size=grid_size, luma_size=luma_size, shift_surface=shift_surface)
    cur_grid, next_grid, cur_luma, next_luma = compact
    delta_grid = [float(n - c) for c, n in zip(cur_grid, next_grid)]
    shift = (
        _coarse_shift_surface_features(tuple(cur_luma), tuple(next_luma), luma_size=luma_size)
        if shift_surface
        else _coarse_shift_features(tuple(cur_luma), tuple(next_luma), luma_size=luma_size)
    )
    return cur_grid + next_grid + delta_grid + shift


def _coarse_shift_features(cur_luma: tuple[float, ...], next_luma: tuple[float, ...], *, luma_size: int = 16, max_shift: int = 4) -> list[float]:
    return _coarse_shift_surface_features(cur_luma, next_luma, luma_size=luma_size, max_shift=max_shift)[:4]


def _coarse_shift_surface_features(
    cur_luma: tuple[float, ...],
    next_luma: tuple[float, ...],
    *,
    luma_size: int = 16,
    max_shift: int = 4,
) -> list[float]:
    if not cur_luma or not next_luma or len(cur_luma) != len(next_luma):
        return [0.0] * 16
    best_shift = (0, 0)
    best_mse: float | None = None
    zero_mse: float | None = None
    costs: list[tuple[int, int, float]] = []
    for sy in range(-max_shift, max_shift + 1):
        for sx in range(-max_shift, max_shift + 1):
            total = 0.0
            count = 0
            for y in range(luma_size):
                ny = y + sy
                if ny < 0 or ny >= luma_size:
                    continue
                for x in range(luma_size):
                    nx = x + sx
                    if nx < 0 or nx >= luma_size:
                        continue
                    diff = next_luma[ny * luma_size + nx] - cur_luma[y * luma_size + x]
                    total += diff * diff
                    count += 1
            if not count:
                continue
            mse = total / count
            costs.append((sx, sy, mse))
            if sx == 0 and sy == 0:
                zero_mse = mse
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_shift = (sx, sy)
    best = best_mse if best_mse is not None else 0.0
    zero = zero_mse if zero_mse is not None else best
    improvement = max(0.0, zero - best)
    if not costs:
        return [0.0] * 16
    mean_mse = sum(cost for _, _, cost in costs) / len(costs)
    std_mse = math.sqrt(sum((cost - mean_mse) ** 2 for _, _, cost in costs) / max(1, len(costs) - 1))
    temperature = max(std_mse, mean_mse * 0.05, 1e-6)
    weights = [math.exp(-(cost - best) / temperature) for _, _, cost in costs]
    weight_total = sum(weights) or 1.0
    soft_x = sum(sx * weight for (sx, _, _), weight in zip(costs, weights)) / weight_total
    soft_y = sum(sy * weight for (_, sy, _), weight in zip(costs, weights)) / weight_total

    def axis_profile(axis: str) -> tuple[float, float]:
        buckets: dict[int, list[float]] = {}
        for sx, sy, cost in costs:
            key = sx if axis == "x" else sy
            buckets.setdefault(key, []).append(cost)
        averaged = sorted((sum(values) / len(values), key) for key, values in buckets.items())
        if not averaged:
            return 0.0, 0.0
        best_axis_cost, best_axis = averaged[0]
        second_axis_cost = averaged[1][0] if len(averaged) > 1 else best_axis_cost
        return best_axis / max_shift, max(0.0, second_axis_cost - best_axis_cost)

    x_axis, x_margin = axis_profile("x")
    y_axis, y_margin = axis_profile("y")
    near_best = sum(1 for _, _, cost in costs if cost <= best + max(1e-6, 0.05 * max(best, mean_mse)))
    return [
        best_shift[0] / max_shift,
        best_shift[1] / max_shift,
        1.0 / (1.0 + best),
        improvement,
        soft_x / max_shift,
        soft_y / max_shift,
        x_axis,
        y_axis,
        x_margin,
        y_margin,
        zero,
        best,
        mean_mse,
        std_mse,
        max(0.0, mean_mse - best),
        near_best / len(costs),
    ]


def _luma_stack_features(
    row: dict[str, Any],
    *,
    offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
    luma_size: int = 16,
    include_deltas: bool = True,
) -> list[float]:
    plane_len = luma_size * luma_size
    planes: list[tuple[float, ...]] = []
    present: list[bool] = []
    for offset in offsets:
        path = _frame_path_with_offset(row, offset)
        if path is None:
            planes.append(tuple(0.0 for _ in range(plane_len)))
            present.append(False)
            continue
        try:
            _, luma = _ppm_grid_and_luma(path, grid_size=1, luma_size=luma_size)
        except (OSError, ValueError):
            planes.append(tuple(0.0 for _ in range(plane_len)))
            present.append(False)
        else:
            planes.append(luma)
            present.append(True)
    values: list[float] = [float(value) for plane in planes for value in plane]
    if include_deltas:
        for idx in range(len(planes) - 1):
            if present[idx] and present[idx + 1]:
                values.extend(float(next_value - cur_value) for cur_value, next_value in zip(planes[idx], planes[idx + 1]))
            else:
                values.extend(0.0 for _ in range(plane_len))
    return values


def record_features(row: dict[str, Any], *, feature_mode: str = "summary") -> list[float]:
    base = _summary_features(row)
    if feature_mode == "summary":
        return base
    if feature_mode == "summary_grid4_shift":
        return base + _frame_pair_features(row, grid_size=4, luma_size=16)
    if feature_mode == "summary_grid8_shift":
        return base + _frame_pair_features(row, grid_size=8, luma_size=16)
    if feature_mode == "summary_grid8_shift_time":
        return base + _frame_pair_features(row, grid_size=8, luma_size=16) + _temporal_basis_features(row)
    if feature_mode == "summary_grid8_shift_surface_time":
        return base + _frame_pair_features(row, grid_size=8, luma_size=16, shift_surface=True) + _temporal_basis_features(row)
    if feature_mode == "summary_compact_grid8_shift_surface_time":
        return base + _compact_frame_pair_features(row, grid_size=8, luma_size=16, shift_surface=True) + _temporal_basis_features(row)
    if feature_mode == "summary_luma16_stack5_time":
        return base + _luma_stack_features(row, offsets=(-2, -1, 0, 1, 2), luma_size=16) + _temporal_basis_features(row)
    raise ValueError(f"unsupported IDM feature_mode: {feature_mode}")


def target_mouse_delta(row: dict[str, Any]) -> tuple[float, float]:
    dxs: list[float] = []
    dys: list[float] = []
    for token in row.get("ground_truth_tokens", []):
        value = token_to_delta_class(token)
        if value is None:
            continue
        if token.startswith("MOUSE_DX_"):
            dxs.append(float(value))
        elif token.startswith("MOUSE_DY_"):
            dys.append(float(value))
    return (sum(dxs) / len(dxs) if dxs else 0.0, sum(dys) / len(dys) if dys else 0.0)


def tokens_from_delta(dx: float, dy: float) -> list[str]:
    return [f"MOUSE_DX_{bin_delta(dx)}", f"MOUSE_DY_{bin_delta(dy)}"]


@dataclass
class Standardizer:
    mean: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, xs: list[list[float]]) -> "Standardizer":
        if not xs:
            return cls([], [])
        dims = len(xs[0])
        mean = [sum(row[i] for row in xs) / len(xs) for i in range(dims)]
        scale = []
        for i in range(dims):
            var = sum((row[i] - mean[i]) ** 2 for row in xs) / max(1, len(xs) - 1)
            scale.append(math.sqrt(var) or 1.0)
        return cls(mean, scale)

    def transform(self, x: list[float]) -> list[float]:
        return [(x[i] - self.mean[i]) / self.scale[i] for i in range(len(self.mean))]


class TinyMouseIDM:
    def __init__(self, input_dim: int, hidden_dim: int = 8, seed: int = 0) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = random.Random(seed)
        if hidden_dim > 0:
            self.w1 = [[rng.uniform(-0.2, 0.2) for _ in range(input_dim)] for _ in range(hidden_dim)]
            self.b1 = [0.0 for _ in range(hidden_dim)]
            self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden_dim)] for _ in range(2)]
        else:
            self.w1 = []
            self.b1 = []
            self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(input_dim)] for _ in range(2)]
        self.b2 = [0.0, 0.0]
        self.standardizer = Standardizer([], [])
        self.train_rmse = None

    def _forward_std(self, x: list[float]) -> tuple[list[float], list[float]]:
        if self.hidden_dim > 0:
            hidden = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.w1, self.b1)]
            out = [sum(w * hi for w, hi in zip(row, hidden)) + b for row, b in zip(self.w2, self.b2)]
            return hidden, out
        out = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.w2, self.b2)]
        return x, out

    def fit(self, records: list[dict[str, Any]], *, epochs: int = 300, lr: float = 0.02) -> "TinyMouseIDM":
        raw_x = [record_features(row) for row in records]
        self.standardizer = Standardizer.fit(raw_x)
        xs = [self.standardizer.transform(x) for x in raw_x]
        ys = [target_mouse_delta(row) for row in records]
        for _ in range(int(epochs)):
            for x, (tx, ty) in zip(xs, ys):
                hidden, out = self._forward_std(x)
                errors = [out[0] - tx, out[1] - ty]
                if self.hidden_dim > 0:
                    old_w2 = [row[:] for row in self.w2]
                    for k in range(2):
                        for j in range(self.hidden_dim):
                            self.w2[k][j] -= lr * errors[k] * hidden[j]
                        self.b2[k] -= lr * errors[k]
                    hidden_grad = []
                    for j in range(self.hidden_dim):
                        grad = sum(errors[k] * old_w2[k][j] for k in range(2)) * (1 - hidden[j] ** 2)
                        hidden_grad.append(grad)
                    for j in range(self.hidden_dim):
                        for i in range(self.input_dim):
                            self.w1[j][i] -= lr * hidden_grad[j] * x[i]
                        self.b1[j] -= lr * hidden_grad[j]
                else:
                    for k in range(2):
                        for i in range(self.input_dim):
                            self.w2[k][i] -= lr * errors[k] * x[i]
                        self.b2[k] -= lr * errors[k]
        residuals = []
        for x, (tx, ty) in zip(xs, ys):
            _, out = self._forward_std(x)
            residuals.append((out[0] - tx) ** 2 + (out[1] - ty) ** 2)
        self.train_rmse = math.sqrt(sum(residuals) / len(residuals)) if residuals else None
        return self

    def predict_delta(self, row: dict[str, Any]) -> tuple[float, float]:
        _, out = self._forward_std(self.standardizer.transform(record_features(row)))
        return out[0], out[1]

    def confidence(self, dx: float, dy: float) -> float:
        rmse = self.train_rmse if self.train_rmse is not None else 1.0
        magnitude = math.sqrt(dx * dx + dy * dy)
        return max(0.05, min(0.99, 1.0 / (1.0 + rmse / (1.0 + magnitude))))

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "standardizer": {"mean": self.standardizer.mean, "scale": self.standardizer.scale},
            "train_rmse": self.train_rmse,
        }


def train_idm_variant(
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    model_name: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    confidence_threshold: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    model = TinyMouseIDM(input_dim=len(record_features(train_records[0])), hidden_dim=hidden_dim, seed=seed).fit(train_records, epochs=epochs, lr=lr)
    train_hash = stable_hash_json([{"id": row["sequence_id"], "tokens": row.get("ground_truth_tokens", []), "features": record_features(row)} for row in train_records])
    rows = []
    predictions = []
    for row in target_records:
        dx, dy = model.predict_delta(row)
        tokens = tokens_from_delta(dx, dy)
        confidence = model.confidence(dx, dy)
        pseudo = {
            "schema": "idm_pseudolabel.v1",
            "sequence_id": row["sequence_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "predicted_tokens": tokens,
            "label_source": "idm_generated",
            "confidence": confidence,
            "model": model_name,
            "training_split_hash": train_hash,
            "input_window": {"frame_ref": row.get("frame", {}).get("path", ""), "frame_index": int(row.get("frame", {}).get("index", 0))},
        }
        validate_named(pseudo, "idm_pseudolabel.schema.json")
        rows.append(pseudo)
        predictions.append({"sequence_id": row["sequence_id"], "timestamp_ns": row["timestamp_ns"], "recording_id": row.get("recording_id"), "game": row.get("game"), "predicted_tokens": tokens})
    out = Path(output_dir) / model_name
    pseudo_path = out / "pseudolabels.jsonl"
    filtered_path = out / "pseudolabels.filtered.jsonl"
    checkpoint_path = out / "checkpoint.json"
    filtered = [row for row in rows if float(row["confidence"]) >= confidence_threshold]
    write_jsonl(pseudo_path, rows)
    write_jsonl(filtered_path, filtered)
    write_json(checkpoint_path, model.state_dict())
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": model_name,
        "dataset_fingerprint": train_hash,
        "train_records": len(train_records),
        "target_records": len(target_records),
        "pseudo_label_path": str(pseudo_path),
        "filtered_pseudo_label_path": str(filtered_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(out / "metrics.json"),
        "calibration": {"confidence_threshold": confidence_threshold, "kept": len(filtered), "total": len(rows), "train_rmse": model.train_rmse},
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(out / "checkpoint_metadata.json", metadata)
    return {"metadata": metadata, "pseudolabels": rows, "predictions": predictions, "model_state": model.state_dict()}
