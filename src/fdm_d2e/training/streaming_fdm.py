from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, read_json, sha256_file, stable_hash_json, write_json
from fdm_d2e.schema import validate_named
from fdm_d2e.training.streaming_idm import iter_jsonl, train_streaming_idm
from fdm_d2e.training.torch_idm import require_torch


def _recording_id(row: dict[str, Any]) -> str:
    return str(row.get("recording_id") or str(row.get("sequence_id", "")).split("#", 1)[0])


def _label_tokens(label: dict[str, Any]) -> list[str]:
    return [str(token) for token in (label.get("predicted_tokens") or ["NOOP"])]


def _validate_label(label: dict[str, Any], *, labels_path: Path, line_no: int) -> None:
    validate_named(label, "idm_pseudolabel.schema.json")
    if label.get("label_source") != "idm_generated":
        raise ValueError(f"{labels_path}:{line_no} expected label_source=idm_generated, got {label.get('label_source')!r}")


def iter_ordered_record_label_pairs(
    records_path: str | Path,
    labels_path: str | Path,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Stream-join D2E records with IDM pseudo-labels in sequence order.

    The G003 IDM predictor emits one pseudo-label per target record while
    iterating the target JSONL.  Requiring order equality keeps this join O(1)
    memory for full-corpus G004 runs and fails loudly if an artifact was mixed
    with the wrong record stream.
    """

    records_path = Path(records_path)
    labels_path = Path(labels_path)
    with records_path.open() as rf, labels_path.open() as lf:
        line_no = 0
        while True:
            rline = rf.readline()
            lline = lf.readline()
            if not rline and not lline:
                break
            line_no += 1
            if not rline or not lline:
                raise ValueError(
                    f"record/label count mismatch at line {line_no}: "
                    f"records_exhausted={not bool(rline)} labels_exhausted={not bool(lline)}"
                )
            record = json.loads(rline)
            label = json.loads(lline)
            if not isinstance(record, dict) or not isinstance(label, dict):
                raise ValueError(f"record and label rows must be JSON objects at line {line_no}")
            _validate_label(label, labels_path=labels_path, line_no=line_no)
            if str(record.get("sequence_id")) != str(label.get("sequence_id")):
                raise ValueError(
                    f"record/label sequence_id mismatch at line {line_no}: "
                    f"{record.get('sequence_id')!r} != {label.get('sequence_id')!r}"
                )
            yield record, label


def _pseudo_record(record: dict[str, Any], label: dict[str, Any], *, labels_path: Path, label_sha256: str) -> dict[str, Any]:
    out = dict(record)
    out["ground_truth_tokens"] = _label_tokens(label)
    out["label_source"] = "idm_pseudolabel_for_fdm"
    out["source_label_artifact"] = str(labels_path)
    out["source_label_sha256"] = label_sha256
    out["source_idm_model"] = label.get("model")
    out["source_idm_confidence"] = label.get("confidence")
    return out


def _write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _tail_split_count(num_rows: int, train_fraction: float, min_target_per_recording: int) -> int:
    if num_rows <= 1:
        return num_rows
    train_count = max(1, int(round(num_rows * float(train_fraction))))
    return min(train_count, max(1, num_rows - int(min_target_per_recording)))


def _split_group(
    group: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    train_fraction: float,
    min_target_per_recording: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    ordered = sorted(group, key=lambda item: (int(item[0].get("timestamp_ns", 0)), str(item[0].get("sequence_id", ""))))
    train_count = _tail_split_count(len(ordered), train_fraction, min_target_per_recording)
    train_pairs = ordered[:train_count]
    target_records = [record for record, _label in ordered[train_count:]]
    return train_pairs, target_records


def materialize_fdm_streaming_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Create full-corpus FDM train/eval JSONLs without loading them all.

    The FDM training target is the IDM-generated pseudo-label, not D2E
    ground-truth.  Evaluation records retain their original ground-truth tokens.
    This preserves the no-oracle-control boundary while allowing the downstream
    action-model trainer to reuse the same token/metric stack as IDM.
    """

    labels_path = Path(config["labels_path"])
    records_path = Path(config["records_path"])
    output_dir = ensure_dir(config.get("output_dir", "outputs/fdm_streaming_d2e_full_compact"))
    train_records_path = output_dir / "fdm_train_pseudolabeled_records.jsonl"
    target_records_path = output_dir / "fdm_target_ground_truth_records.jsonl"
    label_sha256 = sha256_file(labels_path)
    train_fraction = float(config.get("fdm_train_fraction", 0.75))
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("fdm_train_fraction must be in (0, 1]")
    min_target_per_recording = int(config.get("min_target_per_recording", 1))
    explicit_target_records_path = Path(config["target_records_path"]) if config.get("target_records_path") else None

    counts: dict[str, Any] = {
        "pairs": 0,
        "train": 0,
        "target": 0,
        "recordings": 0,
        "games": {},
        "eval_split_tags": {},
        "mode": "explicit_target" if explicit_target_records_path is not None else "recording_tail",
    }

    def observe_record(record: dict[str, Any]) -> None:
        game = str(record.get("game", "unknown"))
        counts["games"][game] = int(counts["games"].get(game, 0)) + 1
        for tag in record.get("eval_split_tags", []) or []:
            tag = str(tag)
            counts["eval_split_tags"][tag] = int(counts["eval_split_tags"].get(tag, 0)) + 1

    with train_records_path.open("w") as train_f, target_records_path.open("w") as target_f:
        if explicit_target_records_path is not None:
            seen_recordings: set[str] = set()
            for line_no, (record, label) in enumerate(iter_ordered_record_label_pairs(records_path, labels_path), 1):
                _write_jsonl_row(train_f, _pseudo_record(record, label, labels_path=labels_path, label_sha256=label_sha256))
                counts["pairs"] = line_no
                counts["train"] += 1
                seen_recordings.add(_recording_id(record))
                observe_record(record)
            for line_no, record in enumerate(iter_jsonl(explicit_target_records_path), 1):
                _write_jsonl_row(target_f, record)
                counts["target"] = line_no
            counts["recordings"] = len(seen_recordings)
        else:
            current_recording: str | None = None
            group: list[tuple[dict[str, Any], dict[str, Any]]] = []

            def flush_group() -> None:
                if not group:
                    return
                train_pairs, target_records = _split_group(
                    group,
                    train_fraction=train_fraction,
                    min_target_per_recording=min_target_per_recording,
                )
                for record, label in train_pairs:
                    _write_jsonl_row(train_f, _pseudo_record(record, label, labels_path=labels_path, label_sha256=label_sha256))
                    counts["train"] += 1
                for record in target_records:
                    _write_jsonl_row(target_f, record)
                    counts["target"] += 1
                counts["recordings"] += 1

            for record, label in iter_ordered_record_label_pairs(records_path, labels_path):
                rid = _recording_id(record)
                if current_recording is None:
                    current_recording = rid
                if rid != current_recording:
                    flush_group()
                    group = []
                    current_recording = rid
                group.append((record, label))
                counts["pairs"] += 1
                observe_record(record)
            flush_group()

    if counts["train"] == 0 or counts["target"] == 0:
        raise ValueError(f"streaming FDM split is empty: train={counts['train']} target={counts['target']}")
    payload = {
        "schema": "streaming_fdm_split_summary.v1",
        "labels_path": str(labels_path),
        "labels_sha256": label_sha256,
        "records_path": str(records_path),
        "target_records_source_path": str(explicit_target_records_path) if explicit_target_records_path is not None else str(records_path),
        "train_records_path": str(train_records_path),
        "target_records_path": str(target_records_path),
        "fdm_train_fraction": train_fraction,
        "min_target_per_recording": min_target_per_recording,
        "counts": counts,
        "dataset_fingerprint": stable_hash_json(
            {
                "labels_path": str(labels_path),
                "labels_sha256": label_sha256,
                "records_path": str(records_path),
                "target_records_source_path": str(explicit_target_records_path) if explicit_target_records_path is not None else str(records_path),
                "train_fraction": train_fraction,
                "min_target_per_recording": min_target_per_recording,
                "counts": counts,
            }
        ),
    }
    write_json(output_dir / "fdm_streaming_split_summary.json", payload)
    return payload


def train_streaming_fdm(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config.get("output_dir", "outputs/fdm_streaming_d2e_full_compact"))
    split_summary_path = output_dir / "fdm_streaming_split_summary.json"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch = None
    if world_size > 1:
        torch = require_torch()
        torch_cfg_for_dist = dict(config.get("torch_idm_config", {}))
        force_cpu = bool(torch_cfg_for_dist.get("force_cpu", config.get("force_cpu", False)))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        backend = str(
            torch_cfg_for_dist.get("distributed_backend")
            or config.get("distributed_backend")
            or ("nccl" if torch.cuda.is_available() and not force_cpu else "gloo")
        )
        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
    if rank == 0:
        split_summary = materialize_fdm_streaming_splits(config)
    if world_size > 1:
        assert torch is not None
        torch.distributed.barrier()
    if rank != 0:
        split_summary = read_json(split_summary_path)
    model_name = str(config.get("model_name", "streaming_compact_fdm"))
    torch_cfg = dict(config.get("torch_idm_config", {}))
    torch_cfg.update(
        {
            "model_name": model_name,
            "train_records": split_summary["train_records_path"],
            "target_records": split_summary["target_records_path"],
            "output_dir": str(output_dir / "torch_model"),
            "summary_out": str(output_dir / "torch_train_summary.json"),
            "endpoints": str(config.get("endpoints", "configs/eval/primary_endpoints.yaml")),
            "baseline_names": list(config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])),
        }
    )
    torch_summary = train_streaming_idm(torch_cfg)
    if torch_summary.get("schema") == "streaming_idm_worker_summary.v1":
        return {
            "schema": "streaming_fdm_worker_summary.v1",
            "rank": torch_summary.get("rank"),
            "world_size": torch_summary.get("world_size"),
            "status": "worker_complete",
        }
    label_hash = str(split_summary["labels_sha256"])
    checkpoint = {
        "schema": "fdm_checkpoint_metadata.v1",
        "model": model_name,
        "label_source": "idm_pseudolabel",
        "source_label_artifact": str(config["labels_path"]),
        "source_label_sha256": label_hash,
        "predictions_path": str(torch_summary["predictions_path"]),
        "num_training_examples": int(split_summary["counts"]["train"]),
        "oracle_ground_truth_control": False,
        "records_path": str(config["records_path"]),
        "train_records_path": str(split_summary["train_records_path"]),
        "target_records_path": str(split_summary["target_records_path"]),
        "target_examples": int(split_summary["counts"]["target"]),
        "split_summary_path": str(output_dir / "fdm_streaming_split_summary.json"),
        "torch_checkpoint_metadata": torch_summary["metadata"],
        "statistical_comparison_path": torch_summary["metadata"].get("statistical_comparison_path"),
        "metrics_path": torch_summary["metadata"].get("metrics_path"),
        "convergence_report_path": torch_summary["metadata"].get("convergence_report_path"),
        "convergence_plateau_met": bool(torch_summary["metadata"].get("convergence_plateau_met", False)),
        "dataset_fingerprint": split_summary["dataset_fingerprint"],
    }
    validate_named(checkpoint, "fdm_checkpoint_metadata.schema.json")
    write_json(output_dir / "checkpoint_metadata.json", checkpoint)
    summary = {
        "schema": "streaming_fdm_train_summary.v1",
        "checkpoint": checkpoint,
        "metrics": torch_summary["metrics"],
        "statistical_comparison": torch_summary["statistical_comparison"],
        "convergence_report": torch_summary.get("convergence_report"),
        "split_summary": split_summary,
        "torch_summary_path": str(output_dir / "torch_train_summary.json"),
        "predictions_path": str(torch_summary["predictions_path"]),
    }
    write_json(config.get("summary_out", output_dir / "summary.json"), summary)
    if config.get("artifact_summary_out"):
        write_json(config["artifact_summary_out"], summary)
    return summary


def train_streaming_fdm_from_config(path: str | Path) -> dict[str, Any]:
    return train_streaming_fdm(load_config(path))
