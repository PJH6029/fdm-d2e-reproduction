from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import read_jsonl, sha256_file, stable_hash_json, write_json, write_jsonl
from fdm_d2e.eval.baselines import build_baseline_predictions
from fdm_d2e.eval.statistics import compare_systems
from fdm_d2e.models.fdm import FrequencyFDM
from fdm_d2e.schema import validate_named
from fdm_d2e.training.torch_idm import train_torch_idm


def run_fdm_smoke(config: dict[str, Any], labels_path: str | Path) -> dict[str, Any]:
    labels_path = Path(labels_path)
    labels = read_jsonl(labels_path)
    required_source = config.get('label_source_required', 'idm_generated')
    for row in labels:
        validate_named(row, 'idm_pseudolabel.schema.json')
        if row.get('label_source') != required_source:
            raise ValueError(f"FDM canonical smoke requires label_source={required_source}; got {row.get('label_source')}")
    model = FrequencyFDM().fit(labels)
    label_hash = sha256_file(labels_path)
    predictions = []
    for row in labels:
        pred = {
            'schema': 'fdm_prediction.v1',
            'sequence_id': row['sequence_id'],
            'timestamp_ns': int(row['timestamp_ns']),
            'predicted_tokens': model.predict(row),
            'model': 'frequency_fdm_smoke',
            'source_label_artifact': str(labels_path),
            'source_label_sha256': label_hash,
        }
        predictions.append(pred)
    pred_path = Path(config.get('predictions_path', 'outputs/fdm/predictions.jsonl'))
    write_jsonl(pred_path, predictions)
    checkpoint = {
        'schema': 'fdm_checkpoint_metadata.v1',
        'model': 'frequency_fdm_smoke',
        'label_source': 'idm_pseudolabel',
        'source_label_artifact': str(labels_path),
        'source_label_sha256': label_hash,
        'predictions_path': str(pred_path),
        'num_training_examples': len(labels),
        'oracle_ground_truth_control': False,
    }
    validate_named(checkpoint, 'fdm_checkpoint_metadata.schema.json')
    checkpoint_path = Path(config.get('checkpoint_metadata_path', 'outputs/fdm/checkpoint_metadata.json'))
    write_json(checkpoint_path, checkpoint)
    train_log = {
        'schema': 'fdm_train_log.v1',
        'consumed_idm_pseudolabels': True,
        'source_label_artifact': str(labels_path),
        'source_label_sha256': label_hash,
        'checkpoint_metadata_path': str(checkpoint_path),
    }
    write_json(config.get('train_log_path', 'outputs/fdm/train_log.json'), train_log)
    return checkpoint


def _split_labels_by_recording_tail(
    labels: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_target_per_recording: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        recording_id = str(row.get("recording_id") or str(row.get("sequence_id", "")).split("#", 1)[0])
        by_recording.setdefault(recording_id, []).append(row)
    train: list[dict[str, Any]] = []
    target: list[dict[str, Any]] = []
    for rows in by_recording.values():
        ordered = sorted(rows, key=lambda item: int(item.get("timestamp_ns", 0)))
        if len(ordered) < 2:
            train.extend(ordered)
            continue
        n_train = max(1, int(round(len(ordered) * float(train_fraction))))
        n_train = min(n_train, max(1, len(ordered) - int(min_target_per_recording)))
        train.extend(ordered[:n_train])
        target.extend(ordered[n_train:])
    return train, target


def _records_with_pseudolabel_tokens(
    records_by_id: dict[str, dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        sequence_id = str(label["sequence_id"])
        if sequence_id not in records_by_id:
            raise KeyError(f"pseudo-label sequence_id not found in records: {sequence_id}")
        row = dict(records_by_id[sequence_id])
        row["ground_truth_tokens"] = list(label.get("predicted_tokens", []))
        row["label_source"] = "idm_pseudolabel_for_fdm"
        rows.append(row)
    return rows


def train_fdm_real(config: dict[str, Any]) -> dict[str, Any]:
    """Train a non-smoke FDM action model from IDM pseudo-labels.

    The trainer uses a causal per-recording tail split over the IDM pseudo-label
    artifact: earlier pseudo-labeled windows train the FDM; later windows are
    held out and evaluated against real D2E ground-truth tokens.  Baselines are
    built from the same pseudo-labeled training rows, avoiding oracle
    ground-truth control for the FDM training signal.
    """

    labels_path = Path(config["labels_path"])
    records_path = Path(config["records_path"])
    target_records_path = Path(config["target_records_path"]) if config.get("target_records_path") else None
    labels = read_jsonl(labels_path)
    for row in labels:
        validate_named(row, "idm_pseudolabel.schema.json")
        if row.get("label_source") != "idm_generated":
            raise ValueError(f"FDM real training requires IDM-generated labels; got {row.get('label_source')}")
    source_records = read_jsonl(records_path)
    records_by_id = {str(row["sequence_id"]): row for row in source_records}
    if target_records_path is None:
        train_labels, target_labels = _split_labels_by_recording_tail(
            labels,
            train_fraction=float(config.get("fdm_train_fraction", 0.75)),
            min_target_per_recording=int(config.get("min_target_per_recording", 1)),
        )
        target_records = [records_by_id[str(row["sequence_id"])] for row in target_labels if str(row["sequence_id"]) in records_by_id]
    else:
        train_labels = labels
        target_records = read_jsonl(target_records_path)
    train_records = _records_with_pseudolabel_tokens(records_by_id, train_labels)
    if not train_records or not target_records:
        raise ValueError("FDM real training needs non-empty pseudo-label train and ground-truth target splits")

    output_dir = Path(config.get("output_dir", "outputs/fdm_real"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records_path = output_dir / "fdm_train_pseudolabeled_records.jsonl"
    target_records_path = output_dir / "fdm_target_ground_truth_records.jsonl"
    write_jsonl(train_records_path, train_records)
    write_jsonl(target_records_path, target_records)

    torch_cfg = dict(config.get("torch_idm_config", {}))
    torch_cfg.update(
        {
            "model_name": str(config.get("model_name", "torch_fdm_real")),
            "train_records": str(train_records_path),
            "target_records": str(target_records_path),
            "output_dir": str(output_dir / "torch_model"),
            "summary_out": str(output_dir / "torch_train_summary.json"),
            "endpoints": str(config.get("endpoints", "configs/eval/primary_endpoints.yaml")),
        }
    )
    summary = train_torch_idm(torch_cfg)
    model_name = str(torch_cfg["model_name"])
    predictions = read_jsonl(summary["predictions_path"])
    endpoints_path = str(config.get("endpoints", "configs/eval/primary_endpoints.yaml"))
    predictions_by_name = build_baseline_predictions(train_records, target_records)
    predictions_by_name[model_name] = predictions
    stat_comparison = compare_systems(predictions_by_name, target_records, load_config(endpoints_path))
    write_json(output_dir / "statistical_comparison.json", stat_comparison)

    label_hash = sha256_file(labels_path)
    checkpoint = {
        "schema": "fdm_checkpoint_metadata.v1",
        "model": model_name,
        "label_source": "idm_pseudolabel",
        "source_label_artifact": str(labels_path),
        "source_label_sha256": label_hash,
        "predictions_path": str(summary["predictions_path"]),
        "num_training_examples": len(train_records),
        "oracle_ground_truth_control": False,
        "records_path": str(records_path),
        "target_records_source_path": str(target_records_path) if target_records_path is not None else str(records_path),
        "train_records_path": str(train_records_path),
        "target_records_path": str(target_records_path),
        "target_examples": len(target_records),
        "fdm_train_fraction": float(config.get("fdm_train_fraction", 0.75)),
        "dataset_fingerprint": stable_hash_json(
            {
                "labels_sha256": label_hash,
                "records_path": str(records_path),
                "train_sequence_ids": [row["sequence_id"] for row in train_records],
                "target_sequence_ids": [row["sequence_id"] for row in target_records],
                "torch_config": torch_cfg,
            }
        ),
        "torch_checkpoint_metadata": summary["metadata"],
        "statistical_comparison_path": str(output_dir / "statistical_comparison.json"),
        "summary_path": str(output_dir / "summary.json"),
    }
    validate_named(checkpoint, "fdm_checkpoint_metadata.schema.json")
    checkpoint_path = output_dir / "checkpoint_metadata.json"
    write_json(checkpoint_path, checkpoint)
    fdm_summary = {
        "schema": "fdm_real_train_summary.v1",
        "checkpoint": checkpoint,
        "metrics": summary["metrics"],
        "predictions_path": summary["predictions_path"],
        "statistical_comparison": stat_comparison,
        "torch_summary_path": str(output_dir / "torch_train_summary.json"),
    }
    write_json(output_dir / "summary.json", fdm_summary)
    return fdm_summary
