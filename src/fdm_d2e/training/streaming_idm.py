from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.config import load_config
from fdm_d2e.eval.statistics import cluster_bootstrap_delta, endpoint_value, holm_bonferroni
from fdm_d2e.io_utils import ensure_dir, read_json, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.training.neural_idm import record_features, target_mouse_delta
from fdm_d2e.training.torch_idm import (
    MOUSE_AXIS_CLASSES,
    _axis_class_indices,
    _build_model,
    _categorical_loss,
    _prediction_from_output,
    require_torch,
)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _category_vocab_from_counts(counts: dict[str, int], min_count: int) -> list[str]:
    return sorted(token for token, count in counts.items() if count >= min_count)


def _is_category_token(token: str) -> bool:
    return token.startswith("KEY_") or (
        token.startswith("MOUSE_")
        and not token.startswith("MOUSE_DX_")
        and not token.startswith("MOUSE_DY_")
    )


def _tokens(row: dict[str, Any]) -> list[str]:
    return list(row.get("ground_truth_tokens") or ["NOOP"])


def scan_streaming_idm_stats(train_records: str | Path, *, feature_mode: str, categorical_min_count: int = 1) -> dict[str, Any]:
    count = 0
    mean: list[float] = []
    m2: list[float] = []
    category_counts: dict[str, int] = {}
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    last_tokens_by_recording: dict[str, list[str]] = {}
    last_tokens_by_game: dict[str, list[str]] = {}
    fingerprint = hashlib.sha256()
    for row in iter_jsonl(train_records):
        features = [float(value) for value in record_features(row, feature_mode=feature_mode)]
        if not mean:
            mean = [0.0 for _ in features]
            m2 = [0.0 for _ in features]
        if len(features) != len(mean):
            raise ValueError(f"inconsistent feature dimension in {train_records}: {len(features)} != {len(mean)}")
        count += 1
        for idx, value in enumerate(features):
            delta = value - mean[idx]
            mean[idx] += delta / count
            m2[idx] += delta * (value - mean[idx])
        for token in row.get("ground_truth_tokens", []):
            token = str(token)
            if _is_category_token(token):
                category_counts[token] = category_counts.get(token, 0) + 1
        tokens = _tokens(row)
        sequence_counts[tuple(tokens)] += 1
        last_tokens_by_recording[str(row.get("recording_id", ""))] = tokens
        last_tokens_by_game[str(row.get("game", "unknown"))] = tokens
        fingerprint.update(
            json.dumps(
                {
                    "sequence_id": row.get("sequence_id"),
                    "tokens": row.get("ground_truth_tokens", []),
                    "features": features,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        fingerprint.update(b"\n")
    if count == 0:
        raise ValueError(f"no training rows found in {train_records}")
    std = [(m2[idx] / max(1, count - 1)) ** 0.5 or 1.0 for idx in range(len(mean))]
    return {
        "schema": "streaming_idm_stats.v1",
        "train_records": str(train_records),
        "num_examples": count,
        "feature_mode": feature_mode,
        "input_dim": len(mean),
        "mean": mean,
        "std": std,
        "category_vocab": _category_vocab_from_counts(category_counts, categorical_min_count),
        "category_counts": category_counts,
        "global_majority_tokens": list(sequence_counts.most_common(1)[0][0]) if sequence_counts else ["NOOP"],
        "last_tokens_by_recording": last_tokens_by_recording,
        "last_tokens_by_game": last_tokens_by_game,
        "dataset_fingerprint": fingerprint.hexdigest(),
    }


def _batch_features(torch, rows: list[dict[str, Any]], *, feature_mode: str, mean: list[float], std: list[float], device: str):
    xs = [[float(value) for value in record_features(row, feature_mode=feature_mode)] for row in rows]
    x = torch.tensor(xs, dtype=torch.float32, device=device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).clamp_min(1e-6)
    return (x - mean_t) / std_t


def _category_targets(torch, rows: list[dict[str, Any]], vocab: list[str], *, device: str):
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    y = torch.zeros((len(rows), len(vocab)), dtype=torch.float32, device=device)
    for row_idx, row in enumerate(rows):
        for token in set(row.get("ground_truth_tokens", [])):
            idx = vocab_index.get(str(token))
            if idx is not None:
                y[row_idx, idx] = 1.0
    return y


def _mouse_targets(torch, rows: list[dict[str, Any]], *, device: str):
    return torch.tensor([target_mouse_delta(row) for row in rows], dtype=torch.float32, device=device)


def _axis_targets(torch, rows: list[dict[str, Any]], axis_classes: list[str], *, device: str):
    dx, dy = _axis_class_indices(rows, axis_classes)
    return (
        torch.tensor(dx, dtype=torch.long, device=device),
        torch.tensor(dy, dtype=torch.long, device=device),
    )


def _iter_batches(path: str | Path, batch_size: int, max_examples: int | None = None) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    seen = 0
    for row in iter_jsonl(path):
        if max_examples is not None and seen >= max_examples:
            break
        batch.append(row)
        seen += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _soft_pos_weight(torch, category_counts: dict[str, int], vocab: list[str], total: int, *, cap: float, device: str):
    if not vocab:
        return None
    values = []
    for token in vocab:
        pos = max(1, int(category_counts.get(token, 0)))
        neg = max(1, total - pos)
        values.append(min(float(cap), neg / pos))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _train_one_epoch(
    torch,
    model,
    opt,
    *,
    train_records: str | Path,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    cat_pos_weight,
    mouse_axis_classes: list[str],
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    batch_size = int(config.get("batch_size", 2048))
    feature_mode = str(stats["feature_mode"])
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    losses: list[float] = []
    batches = 0
    examples = 0
    loss_sum = 0.0
    for batch_idx, rows in enumerate(_iter_batches(train_records, batch_size, config.get("max_train_examples"))):
        if world_size > 1 and (batch_idx % world_size) != rank:
            continue
        x = _batch_features(torch, rows, feature_mode=feature_mode, mean=stats["mean"], std=stats["std"], device=device)
        mouse_y = _mouse_targets(torch, rows, device=device)
        cat_y = _category_targets(torch, rows, category_vocab, device=device)
        pred = model(x)
        mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y)
        category_end = 2 + len(category_vocab)
        if category_vocab:
            cat_loss = _categorical_loss(torch, pred[:, 2:category_end], cat_y, cat_pos_weight, config)
        else:
            cat_loss = torch.tensor(0.0, device=device)
        if mouse_head_mode == "axis_softmax":
            dx_y, dy_y = _axis_targets(torch, rows, mouse_axis_classes, device=device)
            axis_count = len(mouse_axis_classes)
            dx_logits = pred[:, category_end : category_end + axis_count]
            dy_logits = pred[:, category_end + axis_count : category_end + (2 * axis_count)]
            axis_loss = 0.5 * (
                torch.nn.functional.cross_entropy(dx_logits, dx_y)
                + torch.nn.functional.cross_entropy(dy_logits, dy_y)
            )
        else:
            axis_loss = torch.tensor(0.0, device=device)
        loss = (
            float(config.get("mouse_regression_loss_weight", 1.0)) * mouse_loss
            + float(config.get("categorical_loss_weight", 0.5)) * cat_loss
            + float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)) * axis_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
        opt.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        loss_sum += loss_value * len(rows)
        batches += 1
        examples += len(rows)
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "loss_sum": loss_sum,
        "batches": batches,
        "examples": examples,
    }


class StreamingActionMetrics:
    def __init__(self) -> None:
        self.matched = 0
        self.keyboard_total = 0
        self.keyboard_correct = 0
        self.button_total = 0
        self.button_correct = 0
        self.button_predicted_total = 0
        self.button_exact_tp = 0
        self.button_fp = 0
        self.button_fn = 0
        self.button_no_gt = 0
        self.button_no_gt_fp = 0
        self.mouse_n = 0
        self.sum_pred = 0.0
        self.sum_gt = 0.0
        self.sum_pred_sq = 0.0
        self.sum_gt_sq = 0.0
        self.sum_cross = 0.0
        self.sum_abs_pred = 0.0
        self.sum_abs_gt = 0.0
        self.failure_count = 0

    @staticmethod
    def _category(tokens: list[str], prefixes: tuple[str, ...]) -> list[str]:
        return sorted(token for token in tokens if token.startswith(prefixes))

    @staticmethod
    def _axis_mean(tokens: list[str], prefix: str) -> float | None:
        from fdm_d2e.tokenization.actions import token_to_delta_class

        values = [token_to_delta_class(token) for token in tokens if token.startswith(prefix)]
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    def update(self, predicted_tokens: list[str], row: dict[str, Any]) -> None:
        self.matched += 1
        gtokens = list(row.get("ground_truth_tokens", []))
        pk = self._category(predicted_tokens, ("KEY_",))
        gk = self._category(gtokens, ("KEY_",))
        if gk:
            self.keyboard_total += 1
            self.keyboard_correct += int(pk == gk)
        pb = self._category(predicted_tokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        gb = self._category(gtokens, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
        if pb:
            self.button_predicted_total += 1
        if gb:
            self.button_total += 1
            self.button_correct += int(pb == gb)
            if pb == gb:
                self.button_exact_tp += 1
            else:
                self.button_fn += 1
                if pb:
                    self.button_fp += 1
        else:
            self.button_no_gt += 1
            if pb:
                self.button_fp += 1
                self.button_no_gt_fp += 1
        for axis_prefix in ("MOUSE_DX_", "MOUSE_DY_"):
            pred_axis = self._axis_mean(predicted_tokens, axis_prefix)
            gt_axis = self._axis_mean(gtokens, axis_prefix)
            if pred_axis is not None and gt_axis is not None:
                self.mouse_n += 1
                self.sum_pred += pred_axis
                self.sum_gt += gt_axis
                self.sum_pred_sq += pred_axis * pred_axis
                self.sum_gt_sq += gt_axis * gt_axis
                self.sum_cross += pred_axis * gt_axis
                self.sum_abs_pred += abs(pred_axis)
                self.sum_abs_gt += abs(gt_axis)
        if predicted_tokens != gtokens:
            self.failure_count += 1

    def payload(self) -> dict[str, Any]:
        if self.mouse_n >= 2:
            n = float(self.mouse_n)
            cov = self.sum_cross - (self.sum_pred * self.sum_gt / n)
            var_pred = self.sum_pred_sq - (self.sum_pred * self.sum_pred / n)
            var_gt = self.sum_gt_sq - (self.sum_gt * self.sum_gt / n)
            pearson = cov / ((var_pred * var_gt) ** 0.5) if var_pred > 0 and var_gt > 0 else None
            mean_abs_pred = self.sum_abs_pred / n
            mean_abs_gt = self.sum_abs_gt / n
            scale_ratio = max(mean_abs_pred, mean_abs_gt) / min(mean_abs_pred, mean_abs_gt) if mean_abs_pred > 0 and mean_abs_gt > 0 else None
        else:
            pearson = None
            scale_ratio = None
        metrics = {
            "schema": "metrics.v1",
            "stage": "fdm_eval",
            "num_examples": self.matched,
            "keyboard": {
                "status": "computed" if self.keyboard_total else "absent",
                "accuracy": self.keyboard_correct / self.keyboard_total if self.keyboard_total else None,
                "num_examples": self.keyboard_total,
            },
            "mouse_button": {
                "status": "computed" if self.button_total else "absent",
                "accuracy": self.button_correct / self.button_total if self.button_total else None,
                "num_examples": self.button_total,
                "predicted_examples": self.button_predicted_total,
                "exact_true_positive_examples": self.button_exact_tp,
                "false_positive_examples": self.button_fp,
                "false_negative_examples": self.button_fn,
                "precision": self.button_exact_tp / (self.button_exact_tp + self.button_fp) if (self.button_exact_tp + self.button_fp) else None,
                "recall": self.button_exact_tp / (self.button_exact_tp + self.button_fn) if (self.button_exact_tp + self.button_fn) else None,
                "f1": (
                    (2 * self.button_exact_tp) / ((2 * self.button_exact_tp) + self.button_fp + self.button_fn)
                    if ((2 * self.button_exact_tp) + self.button_fp + self.button_fn)
                    else None
                ),
                "no_button_examples": self.button_no_gt,
                "no_button_false_positive_examples": self.button_no_gt_fp,
                "no_button_false_positive_rate": self.button_no_gt_fp / self.button_no_gt if self.button_no_gt else None,
            },
            "mouse_move": {
                "status": "computed" if self.mouse_n else "absent",
                "pearson": pearson,
                "scale_ratio": scale_ratio,
                "num_values": self.mouse_n,
            },
            "failure_count": self.failure_count,
        }
        validate_named(metrics, "metrics.schema.json")
        return metrics


def _group_keys(row: dict[str, Any]) -> list[str]:
    keys = ["all"]
    for tag in row.get("eval_split_tags", []) or []:
        keys.append(f"eval_tag:{tag}")
    for field in ("game", "resolution_tier", "source_id"):
        value = row.get(field)
        if value is not None:
            keys.append(f"{field}:{value}")
    for field in ("split_temporal", "split_heldout_recording", "split_heldout_game"):
        value = row.get(field)
        if value is not None:
            keys.append(f"{field}:{value}")
    return keys


def _ensure_metric(metrics: dict[str, StreamingActionMetrics], key: str) -> StreamingActionMetrics:
    if key not in metrics:
        metrics[key] = StreamingActionMetrics()
    return metrics[key]


def _baseline_tokens(name: str, row: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    if name == "noop":
        return ["NOOP"]
    majority = list(stats.get("global_majority_tokens") or ["NOOP"])
    if name == "global_majority":
        return majority
    if name == "last_seen_train":
        by_recording = stats.get("last_tokens_by_recording", {})
        by_game = stats.get("last_tokens_by_game", {})
        return list(
            by_recording.get(str(row.get("recording_id", "")))
            or by_game.get(str(row.get("game", "unknown")))
            or majority
        )
    raise ValueError(f"unsupported streaming baseline: {name}")


def _metric_payloads(metrics: dict[str, StreamingActionMetrics]) -> dict[str, Any]:
    return {name: metric.payload() for name, metric in sorted(metrics.items())}


def _streaming_statistical_comparison(
    metrics_by_model_cluster: dict[str, dict[str, StreamingActionMetrics]],
    endpoints_config: dict[str, Any],
) -> dict[str, Any]:
    default_reference_name = str(endpoints_config.get("reference_baseline", "noop"))
    bootstrap_cfg = dict(endpoints_config.get("bootstrap", {}))
    comparisons: list[dict[str, Any]] = []
    payload_by_model_cluster = {
        model: {cluster: metric.payload() for cluster, metric in cluster_metrics.items()}
        for model, cluster_metrics in metrics_by_model_cluster.items()
    }
    for endpoint in endpoints_config.get("endpoints", []):
        reference_name = str(endpoint.get("reference_baseline", default_reference_name))
        if reference_name not in payload_by_model_cluster:
            continue
        reference_values = {
            cluster: value
            for cluster, metrics in payload_by_model_cluster[reference_name].items()
            if (value := endpoint_value(metrics, endpoint)) is not None
        }
        for name, cluster_payloads in payload_by_model_cluster.items():
            if name == reference_name:
                continue
            candidate_values = {
                cluster: value
                for cluster, metrics in cluster_payloads.items()
                if (value := endpoint_value(metrics, endpoint)) is not None
            }
            stats = cluster_bootstrap_delta(
                candidate_values,
                reference_values,
                direction=str(endpoint.get("direction", "higher")),
                n_resamples=int(bootstrap_cfg.get("n_resamples", 2000)),
                confidence=float(bootstrap_cfg.get("confidence", 0.95)),
                seed=int(bootstrap_cfg.get("seed", 0)) + len(comparisons),
            )
            comparisons.append(
                {
                    "model": name,
                    "reference": reference_name,
                    "endpoint": endpoint["name"],
                    "direction": endpoint.get("direction", "higher"),
                    "min_effect": endpoint.get("min_effect"),
                    **stats,
                }
            )
    payload = {
        "schema": "stat_comparison.v1",
        "reference_baseline": default_reference_name,
        "correction": str(endpoints_config.get("correction", "holm_bonferroni")),
        "cluster_key": str(endpoints_config.get("cluster_key", "recording_id")),
        "comparisons": holm_bonferroni(comparisons),
    }
    validate_named(payload, "stat_comparison.schema.json")
    return payload


def _distributed_runtime(torch, config: dict[str, Any]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    force_cpu = bool(config.get("force_cpu", False))
    enabled = world_size > 1
    backend = None
    if enabled:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is required for WORLD_SIZE>1 streaming training")
        backend = str(config.get("distributed_backend") or ("nccl" if torch.cuda.is_available() and not force_cpu else "gloo"))
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("distributed_backend=nccl requires CUDA")
        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available() and not force_cpu:
        device = f"cuda:{local_rank}" if enabled else "cuda"
    else:
        device = "cpu"
    return {
        "enabled": enabled,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_rank0": rank == 0,
        "backend": backend,
        "device": device,
    }


def _barrier(torch, dist: dict[str, Any]) -> None:
    if dist["enabled"] and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _aggregate_epoch_stats(torch, stats: dict[str, Any], *, device: str, dist: dict[str, Any]) -> dict[str, Any]:
    if not dist["enabled"]:
        return stats
    tensor = torch.tensor(
        [
            float(stats.get("loss_sum") or 0.0),
            float(stats.get("examples") or 0),
            float(stats.get("batches") or 0),
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    loss_sum = float(tensor[0].item())
    examples = int(tensor[1].item())
    batches = int(tensor[2].item())
    return {
        "loss": loss_sum / examples if examples else None,
        "loss_sum": loss_sum,
        "batches": batches,
        "examples": examples,
    }


def _predict_stream(
    torch,
    model,
    *,
    target_records: str | Path,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    category_vocab: list[str],
    mouse_axis_classes: list[str],
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model_name = str(config.get("model_name", "streaming_compact_idm"))
    baseline_names = [str(name) for name in config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])]
    all_model_names = [model_name, *baseline_names]
    metrics_by_model = {name: StreamingActionMetrics() for name in all_model_names}
    group_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    cluster_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    category_threshold = float(config.get("category_threshold", 0.35))
    category_thresholds = {token: category_threshold for token in category_vocab}
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 2048)))
    max_target_examples = config.get("max_target_examples")
    model.eval()
    target_count = 0
    sequence_fingerprint = hashlib.sha256()
    pseudo_path = output_dir / "pseudolabels.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    with pseudo_path.open("w") as pseudo_f, predictions_path.open("w") as pred_f, torch.no_grad():
        for rows in _iter_batches(target_records, batch_size, max_target_examples):
            x = _batch_features(torch, rows, feature_mode=str(stats["feature_mode"]), mean=stats["mean"], std=stats["std"], device=device)
            outputs = model(x).detach().cpu().tolist()
            for row, output in zip(rows, outputs):
                _, _, tokens = _prediction_from_output(
                    output,
                    base_dx=0.0,
                    base_dy=0.0,
                    residual_mouse=False,
                    category_vocab=category_vocab,
                    category_thresholds=category_thresholds,
                    category_threshold=category_threshold,
                    mouse_head_mode=mouse_head_mode,
                    mouse_axis_classes=mouse_axis_classes,
                    mouse_axis_decode_mode=str(config.get("mouse_axis_decode_mode", "expected")),
                    mouse_axis_temperature=float(config.get("mouse_axis_temperature", 1.0)),
                    mouse_output_gain=float(config.get("mouse_output_gain", 1.0)),
                )
                confidence = max(0.05, min(0.99, 1.0 / (1.0 + len(tokens))))
                pseudo = {
                    "schema": "idm_pseudolabel.v1",
                    "sequence_id": row["sequence_id"],
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "predicted_tokens": tokens,
                    "label_source": "idm_generated",
                    "confidence": confidence,
                    "model": model_name,
                    "training_split_hash": str(stats["dataset_fingerprint"]),
                    "input_window": {"frame_ref": row.get("frame", {}).get("path", ""), "frame_index": int(row.get("frame", {}).get("index", 0))},
                }
                validate_named(pseudo, "idm_pseudolabel.schema.json")
                pred = {
                    "sequence_id": row["sequence_id"],
                    "recording_id": row.get("recording_id"),
                    "cross_resolution_key": row.get("cross_resolution_key"),
                    "game": row.get("game"),
                    "timestamp_ns": row["timestamp_ns"],
                    "predicted_tokens": tokens,
                }
                pseudo_f.write(json.dumps(pseudo, ensure_ascii=False, sort_keys=True) + "\n")
                pred_f.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
                model_tokens = {model_name: tokens}
                for baseline_name in baseline_names:
                    model_tokens[baseline_name] = _baseline_tokens(baseline_name, row, stats)
                cluster = str(row.get("recording_id") or row.get("cross_resolution_key") or row.get("sequence_id"))
                for name, pred_tokens in model_tokens.items():
                    metrics_by_model[name].update(pred_tokens, row)
                    _ensure_metric(cluster_metrics_by_model[name], cluster).update(pred_tokens, row)
                    for group_key in _group_keys(row):
                        _ensure_metric(group_metrics_by_model[name], group_key).update(pred_tokens, row)
                sequence_fingerprint.update(json.dumps({"id": row["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                sequence_fingerprint.update(b"\n")
                target_count += 1
    metrics_path = output_dir / "metrics.json"
    metrics_payload = metrics_by_model[model_name].payload()
    write_json(metrics_path, metrics_payload)
    label_quality_report = {
        "schema": "idm_label_quality_report.v1",
        "model": model_name,
        "target_records": target_count,
        "model_metrics": metrics_payload,
        "baseline_metrics": {name: metrics_by_model[name].payload() for name in baseline_names},
        "groups_by_model": {
            name: _metric_payloads(group_metrics)
            for name, group_metrics in group_metrics_by_model.items()
        },
        "cluster_count": len(cluster_metrics_by_model[model_name]),
    }
    label_quality_report_path = output_dir / "label_quality_report.json"
    write_json(label_quality_report_path, label_quality_report)
    statistical_comparison = None
    statistical_comparison_path = None
    if config.get("endpoints"):
        statistical_comparison = _streaming_statistical_comparison(
            cluster_metrics_by_model,
            load_config(config["endpoints"]),
        )
        statistical_comparison_path = output_dir / "statistical_comparison.json"
        write_json(statistical_comparison_path, statistical_comparison)
    return {
        "pseudo_label_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
        "label_quality_report_path": str(label_quality_report_path),
        "label_quality_report": label_quality_report,
        "statistical_comparison_path": str(statistical_comparison_path) if statistical_comparison_path else None,
        "statistical_comparison": statistical_comparison,
        "target_records": target_count,
        "prediction_fingerprint": sequence_fingerprint.hexdigest(),
        "checkpoint_path": str(checkpoint_path),
    }


def train_streaming_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    dist = _distributed_runtime(torch, config)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = str(dist["device"])
    train_records = Path(config["train_records"])
    target_records = Path(config["target_records"])
    feature_mode = str(config.get("feature_mode", "summary_compact_grid8_shift_surface_time"))
    out_dir = ensure_dir(config.get("output_dir", "outputs/idm_streaming_full"))
    stats_path = out_dir / "streaming_stats.json"
    if dist["enabled"] and not dist["is_rank0"]:
        _barrier(torch, dist)
        stats = read_json(stats_path)
    else:
        if stats_path.exists() and not bool(config.get("rescan_stats", False)):
            stats = read_json(stats_path)
        else:
            stats = scan_streaming_idm_stats(
                train_records,
                feature_mode=feature_mode,
                categorical_min_count=int(config.get("categorical_min_count", 1)),
            )
            write_json(stats_path, stats)
        if dist["enabled"]:
            _barrier(torch, dist)
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    mouse_axis_classes = [str(value) for value in config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)]
    mouse_head_mode = str(config.get("mouse_head_mode", "axis_softmax"))
    if mouse_head_mode not in {"regression", "axis_softmax"}:
        raise ValueError(f"unsupported mouse_head_mode: {mouse_head_mode}")
    output_dim = 2 + len(category_vocab) + (2 * len(mouse_axis_classes) if mouse_head_mode == "axis_softmax" else 0)
    model = _build_model(
        torch,
        input_dim=int(stats["input_dim"]),
        output_dim=output_dim,
        hidden_dim=int(config.get("hidden_dim", 512)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.05)),
        config=config,
        feature_mode=feature_mode,
    ).to(device)
    train_model = model
    if dist["enabled"]:
        ddp_kwargs = {"device_ids": [int(dist["local_rank"])]} if str(device).startswith("cuda") else {}
        train_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    opt = torch.optim.AdamW(train_model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    cat_pos_weight = _soft_pos_weight(
        torch,
        {str(k): int(v) for k, v in stats.get("category_counts", {}).items()},
        category_vocab,
        int(stats["num_examples"]),
        cap=float(config.get("categorical_pos_weight_cap", 20.0)),
        device=device,
    )
    history = []
    for epoch in range(int(config.get("epochs", 3))):
        join_context = train_model.join() if dist["enabled"] else nullcontext()
        with join_context:
            epoch_stats = _train_one_epoch(
                torch,
                train_model,
                opt,
                train_records=train_records,
                stats=stats,
                config=config,
                device=device,
                category_vocab=category_vocab,
                cat_pos_weight=cat_pos_weight,
                mouse_axis_classes=mouse_axis_classes,
                rank=int(dist["rank"]),
                world_size=int(dist["world_size"]),
            )
        epoch_stats = _aggregate_epoch_stats(torch, epoch_stats, device=device, dist=dist)
        if dist["is_rank0"]:
            history.append({"epoch": epoch + 1, **epoch_stats})
            write_json(out_dir / "train_history.json", {"schema": "streaming_idm_train_history.v1", "history": history})
    if not dist["is_rank0"]:
        _barrier(torch, dist)
        return {
            "schema": "streaming_idm_worker_summary.v1",
            "rank": int(dist["rank"]),
            "world_size": int(dist["world_size"]),
            "status": "worker_complete",
        }
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stats": stats,
            "category_vocab": category_vocab,
            "mouse_head_mode": mouse_head_mode,
            "mouse_axis_classes": mouse_axis_classes,
            "history": history,
        },
        checkpoint_path,
    )
    prediction = _predict_stream(
        torch,
        model,
        target_records=target_records,
        stats=stats,
        config=config,
        device=device,
        category_vocab=category_vocab,
        mouse_axis_classes=mouse_axis_classes,
        checkpoint_path=checkpoint_path,
        output_dir=out_dir,
    )
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(config.get("model_name", "streaming_compact_idm")),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "train_records": int(stats["num_examples"]),
        "target_records": int(prediction["target_records"]),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "filtered_pseudo_label_path": prediction["pseudo_label_path"],
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": prediction["metrics_path"],
        "label_quality_report_path": prediction["label_quality_report_path"],
        "statistical_comparison_path": prediction["statistical_comparison_path"],
        "calibration": {
            "mode": "global_threshold_streaming",
            "category_threshold": float(config.get("category_threshold", 0.35)),
            "last_train_loss": history[-1]["loss"] if history else None,
            "prediction_fingerprint": prediction["prediction_fingerprint"],
        },
        "feature_mode": feature_mode,
        "input_dim": int(stats["input_dim"]),
        "categorical_vocab": category_vocab,
        "mouse_head_mode": mouse_head_mode,
        "mouse_axis_classes": mouse_axis_classes if mouse_head_mode == "axis_softmax" else [],
        "distributed": {
            "enabled": bool(dist["enabled"]),
            "world_size": int(dist["world_size"]),
            "backend": dist["backend"],
            "rank0_device": device,
        },
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    summary = {
        "schema": "streaming_idm_train_summary.v1",
        "metadata": metadata,
        "metrics": prediction["metrics"],
        "label_quality_report": prediction["label_quality_report"],
        "statistical_comparison": prediction["statistical_comparison"],
        "history_tail": history[-5:],
        "device": device,
        "stats_path": str(stats_path),
        "predictions_path": prediction["predictions_path"],
    }
    write_json(config.get("summary_out", out_dir / "summary.json"), summary)
    _barrier(torch, dist)
    return summary
