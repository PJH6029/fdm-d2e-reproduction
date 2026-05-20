from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.io_utils import read_jsonl, sha256_file, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.training.neural_idm import record_features, target_mouse_delta, tokens_from_delta
from fdm_d2e.training.torch_idm import require_torch
from fdm_d2e.training.train_fdm import _records_with_pseudolabel_tokens


def _categorical_tokens(tokens: list[str], *, button: bool) -> list[str]:
    out: list[str] = []
    for token in tokens:
        if token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            continue
        is_button = token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        if button == is_button and (is_button or token.startswith("KEY_")):
            out.append(token)
    return sorted(set(out))


def _vote_tokens(neighbors: list[tuple[float, list[str]]], *, threshold: float, button: bool) -> list[str]:
    scores: dict[str, float] = {}
    total = sum(weight for weight, _ in neighbors) or 1.0
    for weight, tokens in neighbors:
        for token in _categorical_tokens(tokens, button=button):
            scores[token] = scores.get(token, 0.0) + weight
    return sorted(token for token, score in scores.items() if (score / total) >= threshold)


def _vote_mouse(neighbors: list[tuple[float, list[str]]]) -> list[str]:
    total = sum(weight for weight, _ in neighbors) or 1.0
    dx = dy = 0.0
    for weight, tokens in neighbors:
        row = {"ground_truth_tokens": tokens}
        ndx, ndy = target_mouse_delta(row)
        dx += weight * float(ndx)
        dy += weight * float(ndy)
    return tokens_from_delta(dx / total, dy / total)


def _feature_matrix(torch, records: list[dict[str, Any]], *, feature_mode: str, device: str):
    rows = [record_features(row, feature_mode=feature_mode) for row in records]
    tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (tensor - mean) / std, mean, std


def train_knn_fdm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    device = "cuda" if torch.cuda.is_available() and not bool(config.get("force_cpu", False)) else "cpu"
    labels_path = Path(config["labels_path"])
    records_path = Path(config["records_path"])
    target_records_source_path = Path(config["target_records_path"])
    labels = read_jsonl(labels_path)
    for row in labels:
        validate_named(row, "idm_pseudolabel.schema.json")
        if row.get("label_source") != "idm_generated":
            raise ValueError(f"KNN FDM requires IDM-generated labels; got {row.get('label_source')}")
    source_records = read_jsonl(records_path)
    records_by_id = {str(row["sequence_id"]): row for row in source_records}
    train_records = _records_with_pseudolabel_tokens(records_by_id, labels)
    target_records = read_jsonl(target_records_source_path)
    feature_mode = str(config.get("feature_mode", "summary_grid8_shift_surface_time"))
    k = max(1, int(config.get("k", 3)))
    batch_size = max(1, int(config.get("batch_size", 256)))
    temperature = max(float(config.get("distance_temperature", 0.1)), 1e-6)
    keyboard_threshold = float(config.get("keyboard_vote_threshold", 0.5))
    button_threshold = float(config.get("button_vote_threshold", 0.5))
    output_dir = Path(config.get("output_dir", "outputs/fdm_knn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records_path = output_dir / "fdm_train_pseudolabeled_records.jsonl"
    target_records_path = output_dir / "fdm_target_ground_truth_records.jsonl"
    write_jsonl(train_records_path, train_records)
    write_jsonl(target_records_path, target_records)

    train_x, mean, std = _feature_matrix(torch, train_records, feature_mode=feature_mode, device=device)
    target_rows = [record_features(row, feature_mode=feature_mode) for row in target_records]
    train_x = torch.nn.functional.normalize(train_x, dim=1)
    train_tokens = [list(row.get("ground_truth_tokens", [])) for row in train_records]
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(target_records), batch_size):
        rows = target_records[start:start + batch_size]
        raw = torch.tensor(target_rows[start:start + batch_size], dtype=torch.float32, device=device)
        x = torch.nn.functional.normalize((raw - mean) / std, dim=1)
        sims = x @ train_x.T
        top_scores, top_indices = torch.topk(sims, k=min(k, train_x.shape[0]), dim=1)
        for row, scores, indices in zip(rows, top_scores.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            raw_weights = [float(torch.exp(torch.tensor(score / temperature)).item()) for score in scores]
            max_weight = max(raw_weights) if raw_weights else 1.0
            neighbors = [(weight / max_weight, train_tokens[int(idx)]) for weight, idx in zip(raw_weights, indices)]
            tokens = []
            tokens.extend(_vote_mouse(neighbors))
            tokens.extend(_vote_tokens(neighbors, threshold=keyboard_threshold, button=False))
            tokens.extend(_vote_tokens(neighbors, threshold=button_threshold, button=True))
            predictions.append({
                "sequence_id": row["sequence_id"],
                "recording_id": row.get("recording_id"),
                "game": row.get("game"),
                "timestamp_ns": row["timestamp_ns"],
                "predicted_tokens": tokens or ["NOOP"],
            })
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    metrics = compute_metrics(predictions, target_records)
    write_json(output_dir / "metrics.json", metrics)
    predictions_by_name = build_baseline_predictions(train_records, target_records)
    model_name = str(config.get("model_name", "knn_fdm"))
    predictions_by_name[model_name] = predictions
    stat = compare_systems(predictions_by_name, target_records, load_config(str(config.get("endpoints", "configs/eval/primary_endpoints.yaml"))))
    write_json(output_dir / "statistical_comparison.json", stat)
    label_hash = sha256_file(labels_path)
    checkpoint = {
        "schema": "fdm_checkpoint_metadata.v1",
        "model": model_name,
        "label_source": "idm_pseudolabel",
        "source_label_artifact": str(labels_path),
        "source_label_sha256": label_hash,
        "predictions_path": str(predictions_path),
        "num_training_examples": len(train_records),
        "oracle_ground_truth_control": False,
        "records_path": str(records_path),
        "target_records_source_path": str(target_records_source_path),
        "train_records_path": str(train_records_path),
        "target_records_path": str(target_records_path),
        "target_examples": len(target_records),
        "dataset_fingerprint": stable_hash_json({
            "labels_sha256": label_hash,
            "records_path": str(records_path),
            "target_records_path": str(target_records_source_path),
            "config": {k: v for k, v in config.items() if k not in {"output_dir"}},
        }),
        "statistical_comparison_path": str(output_dir / "statistical_comparison.json"),
        "summary_path": str(output_dir / "summary.json"),
    }
    validate_named(checkpoint, "fdm_checkpoint_metadata.schema.json")
    write_json(output_dir / "checkpoint_metadata.json", checkpoint)
    summary = {
        "schema": "fdm_knn_train_summary.v1",
        "checkpoint": checkpoint,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
        "statistical_comparison": stat,
        "device": device,
        "config": {
            "feature_mode": feature_mode,
            "k": k,
            "distance_temperature": temperature,
            "keyboard_vote_threshold": keyboard_threshold,
            "button_vote_threshold": button_threshold,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
