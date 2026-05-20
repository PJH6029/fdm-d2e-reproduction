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


def categorical_token_vocab(records: list[dict[str, Any]], *, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for row in records:
        for token in row.get("ground_truth_tokens", []):
            if token.startswith("KEY_") or (
                token.startswith("MOUSE_")
                and not token.startswith("MOUSE_DX_")
                and not token.startswith("MOUSE_DY_")
            ):
                counts[token] = counts.get(token, 0) + 1
    return sorted(token for token, count in counts.items() if count >= min_count)


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


def _build_model(torch, input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float):
    if depth <= 0:
        return torch.nn.Linear(input_dim, output_dim)
    layers = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([torch.nn.Linear(dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout)])
        dim = hidden_dim
    layers.append(torch.nn.Linear(dim, output_dim))
    return torch.nn.Sequential(*layers)


def _split_calibration_records(
    records: list[dict[str, Any]],
    *,
    fraction: float,
    min_calibration_per_recording: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically hold out the tail of each training recording for calibration.

    Calibration records come only from the training split.  The tail split keeps
    threshold tuning away from the test/heldout split while preserving temporal
    order within each recording.
    """

    if fraction <= 0 or len(records) < 2:
        return list(records), []
    fraction = min(0.95, float(fraction))
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_recording.setdefault(str(row.get("recording_id", "")), []).append(row)
    fit: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for rows in by_recording.values():
        ordered = sorted(rows, key=lambda item: int(item.get("timestamp_ns", 0)))
        if len(ordered) < 2:
            fit.extend(ordered)
            continue
        n_cal = max(int(round(len(ordered) * fraction)), int(min_calibration_per_recording))
        n_cal = min(max(1, n_cal), len(ordered) - 1)
        fit.extend(ordered[:-n_cal])
        calibration.extend(ordered[-n_cal:])
    return fit, calibration


def _category_label_matrix(records: list[dict[str, Any]], vocab: list[str]) -> list[list[int]]:
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    rows = [[0 for _ in vocab] for _ in records]
    for row_idx, row in enumerate(records):
        for token in set(row.get("ground_truth_tokens", [])):
            if token in vocab_index:
                rows[row_idx][vocab_index[token]] = 1
    return rows


def _select_threshold(
    scores: list[float],
    labels: list[int],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[float, dict[str, Any]]:
    positives = sum(1 for label in labels if label)
    if not scores or positives == 0:
        return default_threshold, {"positive_count": positives, "score": None, "precision": None, "recall": None}
    beta2 = beta * beta
    best_threshold = default_threshold
    best_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -abs(default_threshold - 0.5))
    best_stats: dict[str, Any] = {}
    for threshold in grid:
        tp = fp = fn = 0
        for score, label in zip(scores, labels):
            pred = score >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif (not pred) and label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision == 0.0 and recall == 0.0:
            f_score = 0.0
        else:
            f_score = (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
        # Prefer F-score, then precision to avoid noisy rare-token flood, then
        # recall, then a conservative/higher threshold for deterministic ties.
        key = (f_score, precision, recall, threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_stats = {
                "positive_count": positives,
                "score": f_score,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best_threshold, best_stats


def _calibrated_category_thresholds_from_scores(
    score_rows: list[list[float]],
    label_rows: list[list[int]],
    vocab: list[str],
    *,
    default_threshold: float,
    grid: list[float],
    beta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    per_token: dict[str, Any] = {}
    for idx, token in enumerate(vocab):
        scores = [float(row[idx]) for row in score_rows]
        labels = [int(row[idx]) for row in label_rows]
        threshold, stats = _select_threshold(
            scores,
            labels,
            default_threshold=default_threshold,
            grid=grid,
            beta=beta,
        )
        thresholds[token] = float(threshold)
        per_token[token] = {"threshold": float(threshold), **stats}
    diagnostics = {
        "default_threshold": default_threshold,
        "beta": beta,
        "grid": grid,
        "per_token": per_token,
    }
    return thresholds, diagnostics


def _mouse_baseline_deltas(
    records: list[dict[str, Any]],
    *,
    mode: str,
    train_records: list[dict[str, Any]] | None = None,
) -> list[tuple[float, float]]:
    if mode == "none":
        return [(0.0, 0.0) for _ in records]
    if mode not in {"causal_last_seen", "target_last_seen_train"}:
        raise ValueError(f"unsupported mouse_baseline mode: {mode}")

    fallback = (0.0, 0.0)
    last_by_recording: dict[str, tuple[float, float]] = {}
    last_by_game: dict[str, tuple[float, float]] = {}
    if mode == "target_last_seen_train":
        if train_records is None:
            raise ValueError("target_last_seen_train requires train_records")
        for row in sorted(train_records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0)))):
            delta = target_mouse_delta(row)
            last_by_recording[str(row.get("recording_id", ""))] = delta
            last_by_game[str(row.get("game", "unknown"))] = delta
        if train_records:
            fallback = target_mouse_delta(train_records[-1])

    baselines_by_id: dict[str, tuple[float, float]] = {}
    ordered = sorted(records, key=lambda item: (str(item.get("recording_id", "")), int(item.get("timestamp_ns", 0))))
    for row in ordered:
        baseline = (
            last_by_recording.get(str(row.get("recording_id", "")))
            or last_by_game.get(str(row.get("game", "unknown")))
            or fallback
        )
        baselines_by_id[str(row["sequence_id"])] = baseline
        if mode == "causal_last_seen":
            delta = target_mouse_delta(row)
            last_by_recording[str(row.get("recording_id", ""))] = delta
            last_by_game[str(row.get("game", "unknown"))] = delta
            fallback = delta
    return [baselines_by_id[str(row["sequence_id"])] for row in records]


def _tensorize(
    torch,
    records: list[dict[str, Any]],
    device: str,
    vocab: list[str],
    *,
    feature_mode: str,
    mouse_baselines: list[tuple[float, float]] | None = None,
    residual_mouse: bool = False,
):
    mouse_baselines = mouse_baselines or [(0.0, 0.0) for _ in records]
    def features_for(row: dict[str, Any], baseline: tuple[float, float]) -> list[float]:
        values = record_features(row, feature_mode=feature_mode)
        if residual_mouse:
            values = values + [float(baseline[0]), float(baseline[1])]
        return values

    xs = torch.tensor(
        [features_for(row, baseline) for row, baseline in zip(records, mouse_baselines)],
        dtype=torch.float32,
        device=device,
    )
    target_deltas = [target_mouse_delta(row) for row in records]
    if residual_mouse:
        mouse_targets = [(tx - bx, ty - by) for (tx, ty), (bx, by) in zip(target_deltas, mouse_baselines)]
    else:
        mouse_targets = target_deltas
    mouse_y = torch.tensor(mouse_targets, dtype=torch.float32, device=device)
    cat_y = torch.zeros((len(records), len(vocab)), dtype=torch.float32, device=device)
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    for row_idx, row in enumerate(records):
        for token in set(row.get("ground_truth_tokens", [])):
            if token in vocab_index:
                cat_y[row_idx, vocab_index[token]] = 1.0
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (xs - mean) / std, mouse_y, cat_y, mean.squeeze(0).detach().cpu().tolist(), std.squeeze(0).detach().cpu().tolist()


def _categorical_loss(torch, logits, targets, pos_weight, config: dict[str, Any]):
    mode = str(config.get("categorical_loss", "bce"))
    if mode == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
    if mode != "focal":
        raise ValueError(f"unsupported categorical_loss: {mode}")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    p_t = (probs * targets) + ((1.0 - probs) * (1.0 - targets))
    gamma = float(config.get("focal_gamma", 2.0))
    loss = ((1.0 - p_t).clamp_min(1e-6) ** gamma) * bce
    if "focal_alpha" in config:
        alpha = float(config["focal_alpha"])
        alpha_factor = (alpha * targets) + ((1.0 - alpha) * (1.0 - targets))
        loss = alpha_factor * loss
    return loss.mean()


def _fit_torch_model(
    torch,
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    device: str,
    vocab: list[str],
    feature_mode: str,
    residual_mouse: bool,
):
    train_mouse_baselines = _mouse_baseline_deltas(
        records,
        mode="causal_last_seen" if residual_mouse else "none",
    )
    train_x, mouse_y, cat_y, mean, std = _tensorize(
        torch,
        records,
        device,
        vocab,
        feature_mode=feature_mode,
        mouse_baselines=train_mouse_baselines,
        residual_mouse=residual_mouse,
    )
    cat_pos_weight = None
    if vocab:
        positives = cat_y.sum(dim=0)
        negatives = max(1, cat_y.shape[0]) - positives
        cat_pos_weight = (negatives / positives.clamp_min(1.0)).clamp(
            max=float(config.get("categorical_pos_weight_cap", 20.0))
        )
    input_dim = int(train_x.shape[1])
    model = _build_model(
        torch,
        input_dim=input_dim,
        output_dim=2 + len(vocab),
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
            mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y[idx])
            if vocab:
                cat_loss = _categorical_loss(torch, pred[:, 2:], cat_y[idx], cat_pos_weight, config)
            else:
                cat_loss = torch.tensor(0.0, device=device)
            loss = mouse_loss + float(config.get("categorical_loss_weight", 0.5)) * cat_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses)})
    return model, mean, std, history, input_dim, train_mouse_baselines


def _predict_raw_outputs(
    torch,
    model,
    records: list[dict[str, Any]],
    *,
    device: str,
    mean: list[float],
    std: list[float],
    feature_mode: str,
    mouse_baselines: list[tuple[float, float]],
    residual_mouse: bool,
) -> list[list[float]]:
    target_features = []
    for row, baseline in zip(records, mouse_baselines):
        values = record_features(row, feature_mode=feature_mode)
        if residual_mouse:
            values = values + [float(baseline[0]), float(baseline[1])]
        target_features.append(values)
    raw_target_x = torch.tensor(target_features, dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6)
    target_x = (raw_target_x - mean_t) / std_t
    model.eval()
    with torch.no_grad():
        return model(target_x).detach().cpu().tolist()


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
    feature_mode = str(config.get("feature_mode", "summary"))
    mouse_target_mode = str(config.get("mouse_target_mode", "absolute"))
    if mouse_target_mode not in {"absolute", "residual_last_seen"}:
        raise ValueError(f"unsupported mouse_target_mode: {mouse_target_mode}")
    residual_mouse = mouse_target_mode == "residual_last_seen"
    vocab = categorical_token_vocab(train_records, min_count=int(config.get("categorical_min_count", 1)))
    category_threshold = float(config.get("category_threshold", 0.35))
    category_thresholds = {token: category_threshold for token in vocab}
    calibration_info: dict[str, Any] = {
        "category_threshold_mode": str(config.get("category_threshold_mode", "global")),
        "category_threshold": category_threshold,
        "category_thresholds": category_thresholds,
    }
    threshold_mode = calibration_info["category_threshold_mode"]
    if threshold_mode not in {"global", "per_token_calibrated"}:
        raise ValueError(f"unsupported category_threshold_mode: {threshold_mode}")

    calibrated_model = None
    calibrated_mean = calibrated_std = calibrated_history = None
    if threshold_mode == "per_token_calibrated" and vocab:
        fraction = float(config.get("category_calibration_fraction", 0.0))
        fit_records, calibration_records = _split_calibration_records(
            train_records,
            fraction=fraction,
            min_calibration_per_recording=int(config.get("category_calibration_min_per_recording", 1)),
        )
        if not calibration_records:
            fit_records, calibration_records = train_records, train_records
        torch.manual_seed(seed)
        calibration_model, calibration_mean, calibration_std, calibration_history, _, _ = _fit_torch_model(
            torch,
            fit_records,
            config=config,
            device=device,
            vocab=vocab,
            feature_mode=feature_mode,
            residual_mouse=residual_mouse,
        )
        calibration_baselines = _mouse_baseline_deltas(
            calibration_records,
            mode="target_last_seen_train" if residual_mouse else "none",
            train_records=fit_records,
        )
        calibration_outputs = _predict_raw_outputs(
            torch,
            calibration_model,
            calibration_records,
            device=device,
            mean=calibration_mean,
            std=calibration_std,
            feature_mode=feature_mode,
            mouse_baselines=calibration_baselines,
            residual_mouse=residual_mouse,
        )
        import math

        score_rows = [[1.0 / (1.0 + math.exp(-float(logit))) for logit in output[2:]] for output in calibration_outputs]
        label_rows = _category_label_matrix(calibration_records, vocab)
        raw_grid = config.get("category_calibration_grid")
        grid = (
            [float(value) for value in raw_grid]
            if isinstance(raw_grid, list)
            else [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        category_thresholds, threshold_diagnostics = _calibrated_category_thresholds_from_scores(
            score_rows,
            label_rows,
            vocab,
            default_threshold=category_threshold,
            grid=grid,
            beta=float(config.get("category_calibration_beta", 1.0)),
        )
        calibration_info.update(
            {
                "category_thresholds": category_thresholds,
                "category_calibration_records": len(calibration_records),
                "category_calibration_fit_records": len(fit_records),
                "category_calibration_refit_full_train": bool(config.get("category_calibration_refit_full_train", True)),
                "category_calibration_history_tail": calibration_history[-3:],
                "category_threshold_diagnostics": threshold_diagnostics,
            }
        )
        if not bool(config.get("category_calibration_refit_full_train", True)):
            calibrated_model = calibration_model
            calibrated_mean = calibration_mean
            calibrated_std = calibration_std
            calibrated_history = calibration_history

    if calibrated_model is not None:
        model = calibrated_model
        mean = calibrated_mean
        std = calibrated_std
        history = calibrated_history
        input_dim = len(mean)
    else:
        torch.manual_seed(seed)
        model, mean, std, history, input_dim, _ = _fit_torch_model(
            torch,
            train_records,
            config=config,
            device=device,
            vocab=vocab,
            feature_mode=feature_mode,
            residual_mouse=residual_mouse,
        )
    out_dir = Path(config.get("output_dir", "outputs/idm_torch"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "mean": mean, "std": std, "config": config, "history": history, "category_thresholds": category_thresholds}, checkpoint_path)
    # Predict heldout pseudo-labels.
    target_mouse_baselines = _mouse_baseline_deltas(
        target_records,
        mode="target_last_seen_train" if residual_mouse else "none",
        train_records=train_records,
    )
    deltas = _predict_raw_outputs(
        torch,
        model,
        target_records,
        device=device,
        mean=mean,
        std=std,
        feature_mode=feature_mode,
        mouse_baselines=target_mouse_baselines,
        residual_mouse=residual_mouse,
    )
    fingerprint_mouse_baselines = _mouse_baseline_deltas(
        train_records,
        mode="causal_last_seen" if residual_mouse else "none",
    )
    train_hash = stable_hash_json(
        [
            {
                "id": row["sequence_id"],
                "tokens": row.get("ground_truth_tokens", []),
                "features": record_features(row, feature_mode=feature_mode),
                "mouse_baseline": baseline,
                "feature_mode": feature_mode,
                "mouse_target_mode": mouse_target_mode,
            }
            for row, baseline in zip(train_records, fingerprint_mouse_baselines)
        ]
    )
    pseudo_rows = []
    predictions = []
    for row, output, (base_dx, base_dy) in zip(target_records, deltas, target_mouse_baselines):
        dx, dy = float(output[0]), float(output[1])
        if residual_mouse:
            dx += float(base_dx)
            dy += float(base_dy)
        tokens = tokens_from_delta(float(dx), float(dy))
        if vocab:
            import math

            for token, logit in zip(vocab, output[2:]):
                prob = 1.0 / (1.0 + math.exp(-float(logit)))
                if prob >= float(category_thresholds.get(token, category_threshold)):
                    tokens.append(token)
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
        "calibration": {
            "confidence_threshold": threshold,
            "kept": sum(1 for row in pseudo_rows if row["confidence"] >= threshold),
            "total": len(pseudo_rows),
            "last_train_loss": history[-1]["loss"] if history else None,
            **calibration_info,
        },
        "categorical_vocab": vocab,
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "mouse_target_mode": mouse_target_mode,
        "category_threshold": category_threshold,
        "category_threshold_mode": threshold_mode,
        "category_thresholds": category_thresholds,
        "categorical_loss": str(config.get("categorical_loss", "bce")),
        "categorical_pos_weight_cap": float(config.get("categorical_pos_weight_cap", 20.0)),
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
