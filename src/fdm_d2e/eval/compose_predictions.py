from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.eval.action_metrics import compute_metrics
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.io_utils import read_jsonl, stable_hash_json, write_json, write_jsonl

TOKEN_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "keyboard": ("KEY_",),
    "mouse_move": ("MOUSE_DX_", "MOUSE_DY_"),
    "mouse_button": ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"),
}


def _tokens_for_group(tokens: list[str], group: str) -> list[str]:
    prefixes = TOKEN_GROUP_PREFIXES[group]
    return [token for token in tokens if token.startswith(prefixes)]


def _prediction_map(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["sequence_id"]): row for row in read_jsonl(path)}


def compose_prediction_tokens(
    sequence_id: str,
    *,
    sources_by_group: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    tokens: list[str] = []
    for group in ("mouse_move", "keyboard", "mouse_button"):
        source = sources_by_group[group].get(sequence_id)
        if not source:
            continue
        tokens.extend(_tokens_for_group(list(source.get("predicted_tokens", [])), group))
    return tokens


def compose_idm_predictions(config: dict[str, Any]) -> dict[str, Any]:
    required_groups = ("keyboard", "mouse_move", "mouse_button")
    source_paths = {group: config.get(f"{group}_predictions") for group in required_groups}
    missing = [group for group, path in source_paths.items() if not path]
    if missing:
        raise ValueError(f"missing prediction source(s): {', '.join(missing)}")

    train_records = read_jsonl(config["train_records"])
    target_records = read_jsonl(config["target_records"])
    sources_by_group = {group: _prediction_map(str(path)) for group, path in source_paths.items()}
    predictions: list[dict[str, Any]] = []
    for row in target_records:
        sequence_id = str(row["sequence_id"])
        predictions.append(
            {
                "sequence_id": row["sequence_id"],
                "recording_id": row.get("recording_id"),
                "game": row.get("game"),
                "timestamp_ns": row["timestamp_ns"],
                "predicted_tokens": compose_prediction_tokens(sequence_id, sources_by_group=sources_by_group),
            }
        )

    output_dir = Path(config.get("output_dir", "outputs/idm_portfolio"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(config.get("predictions_out", output_dir / "predictions.jsonl"))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, predictions)

    metrics = compute_metrics(predictions, target_records)
    metrics_path = Path(config.get("metrics_out", output_dir / "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)

    source_fingerprint = stable_hash_json(
        {
            "source_paths": source_paths,
            "source_hashes": {
                group: stable_hash_json(read_jsonl(str(path))) for group, path in source_paths.items()
            },
            "target_ids": [row["sequence_id"] for row in target_records],
        }
    )
    model_name = str(config.get("model_name", "idm_prediction_portfolio"))
    statistical_comparison = None
    if config.get("endpoints"):
        predictions_by_name = build_baseline_predictions(train_records, target_records)
        predictions_by_name[model_name] = predictions
        statistical_comparison = compare_systems(predictions_by_name, target_records, load_config(config["endpoints"]))
        write_json(output_dir / "statistical_comparison.json", statistical_comparison)

    summary = {
        "schema": "idm_prediction_portfolio_summary.v1",
        "model_name": model_name,
        "source_paths": source_paths,
        "source_fingerprint": source_fingerprint,
        "target_records": len(target_records),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "statistical_comparison": statistical_comparison,
    }
    summary_out = Path(config.get("summary_out", output_dir / "summary.json"))
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_out, summary)
    return summary
