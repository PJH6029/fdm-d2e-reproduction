from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import bin_delta, token_to_delta_class


def record_features(row: dict[str, Any]) -> list[float]:
    frame = row.get("frame", {})
    features = [float(v) for v in frame.get("features", [])]
    while len(features) < 5:
        features.append(0.0)
    return features[:5] + [float(row.get("bin_index", 0)) / 100.0]


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
