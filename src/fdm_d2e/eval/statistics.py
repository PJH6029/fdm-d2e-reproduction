from __future__ import annotations

import math
import random
from typing import Any

from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.schema import validate_named


def cluster_id(row: dict[str, Any], key: str = "recording_id") -> str:
    if key in row and row[key] is not None:
        return str(row[key])
    return str(row["sequence_id"]).split("#", 1)[0]


def _get_path(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _transform(value: Any, transform: str | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if transform == "abs_log_distance":
        if numeric <= 0:
            return None
        return abs(math.log(numeric))
    return numeric


def endpoint_value(metrics: dict[str, Any], endpoint: dict[str, Any]) -> float | None:
    return _transform(_get_path(metrics, list(endpoint["metric_path"])), endpoint.get("transform"))


def values_by_cluster(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    endpoint: dict[str, Any],
    *,
    cluster_key: str = "recording_id",
) -> dict[str, float]:
    gt_by_id = {row["sequence_id"]: row for row in ground_truth}
    clusters = sorted({cluster_id(row, cluster_key) for row in ground_truth})
    values: dict[str, float] = {}
    for cid in clusters:
        gt_cluster = [row for row in ground_truth if cluster_id(row, cluster_key) == cid]
        ids = {row["sequence_id"] for row in gt_cluster}
        pred_cluster = [row for row in predictions if row.get("sequence_id") in ids and row.get("sequence_id") in gt_by_id]
        value = endpoint_value(compute_metrics(pred_cluster, gt_cluster), endpoint)
        if value is not None:
            values[cid] = value
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _delta(candidate: list[float], reference: list[float], direction: str) -> float | None:
    cand_mean = _mean(candidate)
    ref_mean = _mean(reference)
    if cand_mean is None or ref_mean is None:
        return None
    if direction == "lower":
        return ref_mean - cand_mean
    return cand_mean - ref_mean


def cluster_bootstrap_delta(
    candidate_values: dict[str, float],
    reference_values: dict[str, float],
    *,
    direction: str,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    clusters = sorted(set(candidate_values) & set(reference_values))
    if not clusters:
        return {"status": "no_shared_clusters", "delta": None, "ci": [None, None], "p_value": None, "num_clusters": 0}
    observed = _delta([candidate_values[c] for c in clusters], [reference_values[c] for c in clusters], direction)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        value = _delta([candidate_values[c] for c in sampled], [reference_values[c] for c in sampled], direction)
        if value is not None:
            samples.append(value)
    samples.sort()
    if not samples or observed is None:
        return {"status": "insufficient", "delta": observed, "ci": [None, None], "p_value": None, "num_clusters": len(clusters)}
    alpha = 1.0 - confidence
    lo = samples[max(0, int(math.floor(alpha / 2 * (len(samples) - 1))))]
    hi = samples[min(len(samples) - 1, int(math.ceil((1 - alpha / 2) * (len(samples) - 1))))]
    if observed >= 0:
        tail = sum(1 for value in samples if value <= 0) / len(samples)
    else:
        tail = sum(1 for value in samples if value >= 0) / len(samples)
    p_value = min(1.0, 2.0 * tail)
    return {"status": "computed", "delta": observed, "ci": [lo, hi], "p_value": p_value, "num_clusters": len(clusters)}


def holm_bonferroni(rows: list[dict[str, Any]], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    indexed = [(idx, row) for idx, row in enumerate(rows) if row.get("p_value") is not None]
    ordered = sorted(indexed, key=lambda item: float(item[1]["p_value"]))
    adjusted_by_idx: dict[int, tuple[float, bool]] = {}
    prev_adjusted = 0.0
    stopped = False
    m = len(ordered)
    for rank, (idx, row) in enumerate(ordered, start=1):
        p = float(row["p_value"])
        adjusted = min(1.0, max(prev_adjusted, (m - rank + 1) * p))
        prev_adjusted = adjusted
        reject = (not stopped) and p <= alpha / (m - rank + 1)
        if not reject:
            stopped = True
        adjusted_by_idx[idx] = (adjusted, reject)
    output = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        adjusted, reject = adjusted_by_idx.get(idx, (None, False))
        enriched["p_adjusted_holm"] = adjusted
        enriched["reject_holm_0_05"] = reject
        output.append(enriched)
    return output


def compare_systems(
    predictions_by_name: dict[str, list[dict[str, Any]]],
    ground_truth: list[dict[str, Any]],
    endpoints_config: dict[str, Any],
) -> dict[str, Any]:
    default_reference_name = str(endpoints_config.get("reference_baseline", "noop"))
    cluster_key_name = str(endpoints_config.get("cluster_key", "recording_id"))
    bootstrap_cfg = dict(endpoints_config.get("bootstrap", {}))
    comparisons: list[dict[str, Any]] = []
    for endpoint in endpoints_config.get("endpoints", []):
        reference_name = str(endpoint.get("reference_baseline", default_reference_name))
        reference_predictions = predictions_by_name[reference_name]
        reference_values = values_by_cluster(reference_predictions, ground_truth, endpoint, cluster_key=cluster_key_name)
        for name, predictions in predictions_by_name.items():
            if name == reference_name:
                continue
            candidate_values = values_by_cluster(predictions, ground_truth, endpoint, cluster_key=cluster_key_name)
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
    comparisons = holm_bonferroni(comparisons)
    payload = {
        "schema": "stat_comparison.v1",
        "reference_baseline": default_reference_name,
        "correction": str(endpoints_config.get("correction", "holm_bonferroni")),
        "cluster_key": cluster_key_name,
        "comparisons": comparisons,
    }
    validate_named(payload, "stat_comparison.schema.json")
    return payload
