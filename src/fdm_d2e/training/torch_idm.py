from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_jsonl, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.training.neural_idm import record_features, target_mouse_delta, tokens_from_delta


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("Torch IDM training requires `uv sync --extra train` or the cluster training image") from exc
    return torch


class TorchUnavailableError(RuntimeError):
    pass


def _build_model(torch, input_dim: int, hidden_dim: int, depth: int, dropout: float):
    layers = []
    dim = input_dim
    for _ in range(max(1, depth)):
        layers.extend([torch.nn.Linear(dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(torch.nn.Linear(dim, 2))
    return torch.nn.Sequential(*layers)


def _tensorize(torch, records: list[dict[str, Any]], device: str):
    xs = torch.tensor([record_features(row) for row in records], dtype=torch.float32, device=device)
    ys = torch.tensor([target_mouse_delta(row) for row in records], dtype=torch.float32, device=device)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (xs - mean) / std, ys, mean.squeeze(0).detach().cpu().tolist(), std.squeeze(0).detach().cpu().tolist()


def train_torch_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not bool(config.get("force_cpu", False)):
        device = "cuda"
    else:
        device = "cpu"
    train_records = read_jsonl(config["train_records"])
    target_records = read_jsonl(config["target_records"])
    train_x, train_y, mean, std = _tensorize(torch, train_records, device)
    input_dim = int(train_x.shape[1])
    model = _build_model(
        torch,
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 128)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.05)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(train_x.shape[0], device=device)
        losses = []
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_x[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses)})
    out_dir = Path(config.get("output_dir", "outputs/idm_torch"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "mean": mean, "std": std, "config": config, "history": history}, checkpoint_path)
    # Predict heldout pseudo-labels.
    raw_target_x = torch.tensor([record_features(row) for row in target_records], dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6)
    target_x = (raw_target_x - mean_t) / std_t
    model.eval()
    with torch.no_grad():
        deltas = model(target_x).detach().cpu().tolist()
    train_hash = stable_hash_json([{"id": row["sequence_id"], "tokens": row.get("ground_truth_tokens", []), "features": record_features(row)} for row in train_records])
    pseudo_rows = []
    predictions = []
    for row, (dx, dy) in zip(target_records, deltas):
        tokens = tokens_from_delta(float(dx), float(dy))
        confidence = max(0.05, min(0.99, 1.0 / (1.0 + abs(float(dx)) + abs(float(dy)))))
        pseudo = {
            "schema": "idm_pseudolabel.v1",
            "sequence_id": row["sequence_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "predicted_tokens": tokens,
            "label_source": "idm_generated",
            "confidence": confidence,
            "model": str(config.get("model_name", "torch_mlp_idm")),
            "training_split_hash": train_hash,
            "input_window": {"frame_ref": row.get("frame", {}).get("path", ""), "frame_index": int(row.get("frame", {}).get("index", 0))},
        }
        validate_named(pseudo, "idm_pseudolabel.schema.json")
        pseudo_rows.append(pseudo)
        predictions.append({"sequence_id": row["sequence_id"], "recording_id": row.get("recording_id"), "game": row.get("game"), "timestamp_ns": row["timestamp_ns"], "predicted_tokens": tokens})
    pseudo_path = out_dir / "pseudolabels.jsonl"
    filtered_path = out_dir / "pseudolabels.filtered.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    threshold = float(config.get("confidence_threshold", 0.15))
    write_jsonl(pseudo_path, pseudo_rows)
    write_jsonl(filtered_path, [row for row in pseudo_rows if row["confidence"] >= threshold])
    write_jsonl(predictions_path, predictions)
    metrics = compute_metrics(predictions, target_records)
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(config.get("model_name", "torch_mlp_idm")),
        "dataset_fingerprint": train_hash,
        "train_records": len(train_records),
        "target_records": len(target_records),
        "pseudo_label_path": str(pseudo_path),
        "filtered_pseudo_label_path": str(filtered_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "calibration": {"confidence_threshold": threshold, "kept": sum(1 for row in pseudo_rows if row["confidence"] >= threshold), "total": len(pseudo_rows), "last_train_loss": history[-1]["loss"] if history else None},
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    endpoints_path = config.get("endpoints")
    stat_comparison = None
    if endpoints_path:
        predictions_by_name = build_baseline_predictions(train_records, target_records)
        predictions_by_name[str(config.get("model_name", "torch_mlp_idm"))] = predictions
        stat_comparison = compare_systems(predictions_by_name, target_records, load_config(endpoints_path))
        write_json(out_dir / "statistical_comparison.json", stat_comparison)
    summary = {
        "schema": "torch_idm_train_summary.v1",
        "metadata": metadata,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "statistical_comparison": stat_comparison,
        "history_tail": history[-5:],
        "device": device,
    }
    write_json(config.get("summary_out", out_dir / "summary.json"), summary)
    return summary
