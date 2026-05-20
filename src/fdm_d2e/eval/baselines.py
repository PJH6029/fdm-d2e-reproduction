from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from fdm_d2e.io_utils import write_jsonl

TokenSeq = tuple[str, ...]


def _tokens(row: dict[str, Any]) -> list[str]:
    return list(row.get("ground_truth_tokens") or ["NOOP"])


def _majority_sequence(records: list[dict[str, Any]]) -> list[str]:
    counts: Counter[TokenSeq] = Counter(tuple(_tokens(row)) for row in records)
    return list(counts.most_common(1)[0][0]) if counts else ["NOOP"]


def _prediction_row(model: str, row: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    return {
        "schema": "baseline_prediction.v1",
        "model": model,
        "sequence_id": row["sequence_id"],
        "recording_id": row.get("recording_id", row["sequence_id"].split("#", 1)[0]),
        "game": row.get("game", "unknown"),
        "timestamp_ns": int(row.get("timestamp_ns", 0)),
        "predicted_tokens": list(tokens),
    }


def predict_noop(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_prediction_row("noop", row, ["NOOP"]) for row in records]


def predict_global_majority(train_records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    majority = _majority_sequence(train_records)
    return [_prediction_row("global_majority", row, majority) for row in target_records]


def predict_game_majority(train_records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_records:
        by_game[str(row.get("game", "unknown"))].append(row)
    global_majority = _majority_sequence(train_records)
    majority_by_game = {game: _majority_sequence(rows) for game, rows in by_game.items()}
    return [_prediction_row("game_majority", row, majority_by_game.get(str(row.get("game", "unknown")), global_majority)) for row in target_records]


def predict_last_seen_train(train_records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-oracle baseline: replay latest train action for the same recording/game when available."""

    ordered = sorted(train_records, key=lambda row: (str(row.get("recording_id", "")), int(row.get("timestamp_ns", 0))))
    last_by_recording: dict[str, list[str]] = {}
    last_by_game: dict[str, list[str]] = {}
    for row in ordered:
        tokens = _tokens(row)
        last_by_recording[str(row.get("recording_id", ""))] = tokens
        last_by_game[str(row.get("game", "unknown"))] = tokens
    global_majority = _majority_sequence(train_records)
    predictions = []
    for row in target_records:
        tokens = last_by_recording.get(str(row.get("recording_id", ""))) or last_by_game.get(str(row.get("game", "unknown"))) or global_majority
        predictions.append(_prediction_row("last_seen_train", row, tokens))
    return predictions


BASELINE_BUILDERS: dict[str, Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "noop": lambda train, target: predict_noop(target),
    "global_majority": predict_global_majority,
    "game_majority": predict_game_majority,
    "last_seen_train": predict_last_seen_train,
}


def build_baseline_predictions(
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    baseline_names: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    names = baseline_names or list(BASELINE_BUILDERS)
    return {name: BASELINE_BUILDERS[name](train_records, target_records) for name in names}


def write_baseline_predictions(
    predictions_by_name: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, rows in predictions_by_name.items():
        path = Path(output_dir) / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = str(path)
    return paths
