from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.eval.statistics import cluster_bootstrap_delta, endpoint_value, holm_bonferroni, values_by_cluster
from fdm_d2e.io_utils import read_jsonl, stable_hash_json, write_json
from fdm_d2e.schema import validate_named


def _tokens_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("ground_truth_tokens", []) or ["NOOP"])


def _train_baseline_stats(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[str, ...]] = Counter(_tokens_key(row) for row in train_records)
    majority = list(counts.most_common(1)[0][0]) if counts else ["NOOP"]
    last_by_recording: dict[str, list[str]] = {}
    last_by_game: dict[str, list[str]] = {}
    for row in train_records:
        tokens = list(_tokens_key(row))
        last_by_recording[str(row.get("recording_id", ""))] = tokens
        last_by_game[str(row.get("game", "unknown"))] = tokens
    return {"global_majority_tokens": majority, "last_tokens_by_recording": last_by_recording, "last_tokens_by_game": last_by_game}


def _baseline_tokens(name: str, row: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    if name == "noop":
        return ["NOOP"]
    majority = list(stats.get("global_majority_tokens") or ["NOOP"])
    if name == "global_majority":
        return majority
    if name == "last_seen_train":
        return list(
            (stats.get("last_tokens_by_recording") or {}).get(str(row.get("recording_id", "")))
            or (stats.get("last_tokens_by_game") or {}).get(str(row.get("game", "unknown")))
            or majority
        )
    raise ValueError(f"unsupported baseline: {name}")


def _prediction_rows_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["sequence_id"]): row for row in read_jsonl(path)}


def _model_predictions_for_ground_truth(predictions_by_id: dict[str, dict[str, Any]], ground_truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for gt in ground_truth:
        pred = predictions_by_id.get(str(gt["sequence_id"]))
        if pred is None:
            continue
        rows.append({"sequence_id": gt["sequence_id"], "predicted_tokens": list(pred.get("predicted_tokens", []))})
    return rows


def _baseline_predictions(name: str, ground_truth: list[dict[str, Any]], train_stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"sequence_id": row["sequence_id"], "predicted_tokens": _baseline_tokens(name, row, train_stats)} for row in ground_truth]


def _mean(values: dict[str, float]) -> float | None:
    return sum(values.values()) / len(values) if values else None


def _split_rows(rows: list[dict[str, Any]], split_tag: str) -> list[dict[str, Any]]:
    return [row for row in rows if split_tag in [str(tag) for tag in row.get("eval_split_tags", []) or []]]


def compare_split_predictions(
    *,
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    endpoints_config: dict[str, Any],
    split_tag: str,
    model_name: str,
    baseline_names: list[str],
    train_records_path: str | Path | None = None,
) -> dict[str, Any]:
    ground_truth_all = read_jsonl(ground_truth_path)
    ground_truth = _split_rows(ground_truth_all, split_tag)
    predictions_by_id = _prediction_rows_by_id(predictions_path)
    train_records = read_jsonl(train_records_path) if train_records_path else []
    train_stats = _train_baseline_stats(train_records)
    predictions_by_name: dict[str, list[dict[str, Any]]] = {
        model_name: _model_predictions_for_ground_truth(predictions_by_id, ground_truth),
    }
    for baseline in baseline_names:
        predictions_by_name[baseline] = _baseline_predictions(baseline, ground_truth, train_stats)

    default_reference = str(endpoints_config.get("reference_baseline", "noop"))
    cluster_key = str(endpoints_config.get("cluster_key", "recording_id"))
    bootstrap_cfg = dict(endpoints_config.get("bootstrap", {}))
    comparisons: list[dict[str, Any]] = []
    for endpoint in endpoints_config.get("endpoints", []):
        reference = str(endpoint.get("reference_baseline", default_reference))
        if reference not in predictions_by_name:
            raise ValueError(f"reference baseline {reference!r} not available for endpoint {endpoint.get('name')!r}")
        reference_values = values_by_cluster(predictions_by_name[reference], ground_truth, endpoint, cluster_key=cluster_key)
        for name, predictions in predictions_by_name.items():
            if name == reference:
                continue
            candidate_values = values_by_cluster(predictions, ground_truth, endpoint, cluster_key=cluster_key)
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
                    "split": split_tag,
                    "model": name,
                    "reference": reference,
                    "endpoint": endpoint["name"],
                    "direction": endpoint.get("direction", "higher"),
                    "min_effect": endpoint.get("min_effect"),
                    "candidate_value": _mean(candidate_values),
                    "baseline_value": _mean(reference_values),
                    "candidate_clusters": len(candidate_values),
                    "reference_clusters": len(reference_values),
                    **stats,
                }
            )
    comparisons = holm_bonferroni(comparisons)
    payload = {
        "schema": "stat_comparison.v1",
        "reference_baseline": default_reference,
        "correction": str(endpoints_config.get("correction", "holm_bonferroni")),
        "cluster_key": cluster_key,
        "split": split_tag,
        "model": model_name,
        "ground_truth_path": str(ground_truth_path),
        "predictions_path": str(predictions_path),
        "train_records_path": str(train_records_path) if train_records_path else None,
        "ground_truth_records": len(ground_truth),
        "model_prediction_records": len(predictions_by_name[model_name]),
        "baseline_names": baseline_names,
        "comparisons": comparisons,
        "dataset_fingerprint": stable_hash_json(
            {
                "split": split_tag,
                "model": model_name,
                "ground_truth_ids": [row.get("sequence_id") for row in ground_truth[:10000]],
                "prediction_count": len(predictions_by_name[model_name]),
                "comparisons": comparisons,
            }
        ),
        "claim_boundary": "Split-specific statistical comparison for G006; it is valid only for the named split and source predictions.",
    }
    validate_named(payload, "stat_comparison.schema.json")
    return payload


def write_split_statistical_comparisons(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    endpoints = load_config(root_path / str(config.get("endpoints", "configs/eval/primary_endpoints.yaml")))
    predictions_path = root_path / str(config["predictions_path"])
    ground_truth_path = root_path / str(config["ground_truth_path"])
    train_records_path = config.get("train_records_path")
    train_path = root_path / str(train_records_path) if train_records_path else None
    output_dir = root_path / str(config.get("output_dir", Path(config["predictions_path"]).parent))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_tags = [str(tag) for tag in config.get("split_tags", ["temporal", "heldout_recording", "heldout_game"])]
    model_name = str(config.get("model_name", "model"))
    baseline_names = [str(name) for name in config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])]
    outputs = []
    for split_tag in split_tags:
        payload = compare_split_predictions(
            predictions_path=predictions_path,
            ground_truth_path=ground_truth_path,
            endpoints_config=endpoints,
            split_tag=split_tag,
            model_name=model_name,
            baseline_names=baseline_names,
            train_records_path=train_path,
        )
        out_path = output_dir / f"split_{split_tag}_statistical_comparison.json"
        write_json(out_path, payload)
        outputs.append({"split": split_tag, "path": str(out_path), "status": "pass" if payload["comparisons"] else "empty", "comparisons": len(payload["comparisons"])})
    summary = {
        "schema": "split_statistical_comparison_build.v1",
        "status": "pass" if all(row["status"] == "pass" for row in outputs) else "fail",
        "model_name": model_name,
        "outputs": outputs,
        "claim_boundary": "Builder creates split-specific comparison artifacts; G006 still requires prerequisite goals and final artifact synthesis.",
    }
    summary_out = config.get("summary_out")
    if summary_out:
        write_json(root_path / str(summary_out), summary)
    return summary
