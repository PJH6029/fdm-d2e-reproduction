from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, read_json, sha256_file, stable_hash_json, write_json
from fdm_d2e.schema import validate_named
from fdm_d2e.training.streaming_idm import (
    _file_artifact_metadata,
    _json_fingerprint,
    iter_jsonl,
    recover_streaming_idm_outputs_from_checkpoint,
    train_streaming_idm,
)
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


def _open_shard_writers(output_dir: Path, *, prefix: str, num_shards: int) -> tuple[list[Path], list[Any]]:
    if num_shards <= 1:
        return [], []
    shard_dir = ensure_dir(output_dir / f"{prefix}_shards")
    paths = [shard_dir / f"shard_{idx:05d}.jsonl" for idx in range(num_shards)]
    handles = []
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        handles.append(path.open("w"))
    return paths, handles


def _close_all(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.close()


def _write_jsonl_row_with_shards(
    handle,
    shard_handles: list[Any],
    row: dict[str, Any],
    *,
    row_index: int,
) -> None:
    _write_jsonl_row(handle, row)
    if shard_handles:
        _write_jsonl_row(shard_handles[row_index % len(shard_handles)], row)


def _with_prior_action_context(record: dict[str, Any], prior_tokens: list[str] | None, *, source: str) -> dict[str, Any]:
    out = dict(record)
    tokens = list(prior_tokens or ["NOOP"])
    out["prior_action_tokens"] = tokens
    out["prior_action_source"] = source
    out["prior_action_is_reset"] = tokens == ["NOOP"]
    return out


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
    num_output_shards = int(config.get("num_output_shards", config.get("output_shards", 1)))
    if num_output_shards < 1:
        raise ValueError("num_output_shards must be >= 1")
    train_shard_paths, train_shard_handles = _open_shard_writers(output_dir, prefix="fdm_train", num_shards=num_output_shards)
    target_shard_paths, target_shard_handles = _open_shard_writers(output_dir, prefix="fdm_target", num_shards=num_output_shards)
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
        "target_games": {},
        "source_ids": {},
        "resolution_tiers": {},
        "split_names": {},
        "eval_split_tags": {},
        "target_source_ids": {},
        "target_resolution_tiers": {},
        "target_split_names": {},
        "target_eval_split_tags": {},
        "mode": "explicit_target" if explicit_target_records_path is not None else "recording_tail",
    }

    def bump(mapping_name: str, value: Any) -> None:
        if value is None:
            return
        key = str(value)
        if not key:
            return
        mapping = counts[mapping_name]
        mapping[key] = int(mapping.get(key, 0)) + 1

    def observe_record(record: dict[str, Any], *, target: bool = False) -> None:
        game = str(record.get("game", "unknown"))
        game_mapping = counts["target_games" if target else "games"]
        game_mapping[game] = int(game_mapping.get(game, 0)) + 1
        bump("target_source_ids" if target else "source_ids", record.get("source_id"))
        bump("target_resolution_tiers" if target else "resolution_tiers", record.get("resolution_tier"))
        bump("target_split_names" if target else "split_names", record.get("split"))
        for tag in record.get("eval_split_tags", []) or []:
            bump("target_eval_split_tags" if target else "eval_split_tags", tag)

    try:
        with train_records_path.open("w") as train_f, target_records_path.open("w") as target_f:
            if explicit_target_records_path is not None:
                seen_recordings: set[str] = set()
                last_train_label_tokens_by_recording: dict[str, list[str]] = {}
                for line_no, (record, label) in enumerate(iter_ordered_record_label_pairs(records_path, labels_path), 1):
                    rid = _recording_id(record)
                    prior_tokens = last_train_label_tokens_by_recording.get(rid, ["NOOP"])
                    contextual_record = _with_prior_action_context(record, prior_tokens, source="idm_pseudolabel_previous_teacher_forced")
                    train_row = _pseudo_record(contextual_record, label, labels_path=labels_path, label_sha256=label_sha256)
                    _write_jsonl_row_with_shards(train_f, train_shard_handles, train_row, row_index=int(counts["train"]))
                    last_train_label_tokens_by_recording[rid] = _label_tokens(label)
                    counts["pairs"] = line_no
                    counts["train"] += 1
                    seen_recordings.add(rid)
                    observe_record(record)
                last_target_tokens_by_recording: dict[str, list[str]] = {}
                for line_no, record in enumerate(iter_jsonl(explicit_target_records_path), 1):
                    rid = _recording_id(record)
                    prior_tokens = last_target_tokens_by_recording.get(rid, ["NOOP"])
                    target_row = _with_prior_action_context(record, prior_tokens, source="d2e_ground_truth_previous_teacher_forced")
                    _write_jsonl_row_with_shards(target_f, target_shard_handles, target_row, row_index=int(counts["target"]))
                    last_target_tokens_by_recording[rid] = [str(token) for token in record.get("ground_truth_tokens", []) or ["NOOP"]]
                    counts["target"] = line_no
                    observe_record(record, target=True)
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
                    previous_label_tokens = ["NOOP"]
                    for record, label in train_pairs:
                        contextual_record = _with_prior_action_context(
                            record,
                            previous_label_tokens,
                            source="idm_pseudolabel_previous_teacher_forced",
                        )
                        train_row = _pseudo_record(contextual_record, label, labels_path=labels_path, label_sha256=label_sha256)
                        _write_jsonl_row_with_shards(train_f, train_shard_handles, train_row, row_index=int(counts["train"]))
                        previous_label_tokens = _label_tokens(label)
                        counts["train"] += 1
                    previous_gt_by_sequence: dict[str, list[str]] = {}
                    previous_tokens = ["NOOP"]
                    for record, _label in sorted(group, key=lambda item: (int(item[0].get("timestamp_ns", 0)), str(item[0].get("sequence_id", "")))):
                        previous_gt_by_sequence[str(record.get("sequence_id"))] = list(previous_tokens)
                        previous_tokens = [str(token) for token in record.get("ground_truth_tokens", []) or ["NOOP"]]
                    for record in target_records:
                        target_row = _with_prior_action_context(
                            record,
                            previous_gt_by_sequence.get(str(record.get("sequence_id")), ["NOOP"]),
                            source="d2e_ground_truth_previous_teacher_forced",
                        )
                        _write_jsonl_row_with_shards(target_f, target_shard_handles, target_row, row_index=int(counts["target"]))
                        counts["target"] += 1
                        observe_record(record, target=True)
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
    finally:
        _close_all([*train_shard_handles, *target_shard_handles])

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
        "train_record_paths": [str(path) for path in train_shard_paths] if train_shard_paths else [str(train_records_path)],
        "target_record_paths": [str(path) for path in target_shard_paths] if target_shard_paths else [str(target_records_path)],
        "train_records_glob": str(output_dir / "fdm_train_shards" / "shard_*.jsonl") if train_shard_paths else None,
        "target_records_glob": str(output_dir / "fdm_target_shards" / "shard_*.jsonl") if target_shard_paths else None,
        "output_shards": {
            "enabled": bool(train_shard_paths or target_shard_paths),
            "num_shards": num_output_shards if train_shard_paths or target_shard_paths else 1,
            "train_record_paths": [str(path) for path in train_shard_paths],
            "target_record_paths": [str(path) for path in target_shard_paths],
        },
        "fdm_train_fraction": train_fraction,
        "min_target_per_recording": min_target_per_recording,
        "counts": counts,
        "prior_action_context": {
            "train_source": "idm_pseudolabel_previous_teacher_forced",
            "target_source": "d2e_ground_truth_previous_teacher_forced",
            "first_action_tokens": ["NOOP"],
            "claim_boundary": "Offline FDM evaluation is teacher-forced on previous actions only; closed-loop stability remains a separate G008 live-suite requirement.",
        },
        "dataset_fingerprint": stable_hash_json(
            {
                "labels_path": str(labels_path),
                "labels_sha256": label_sha256,
                "records_path": str(records_path),
                "target_records_source_path": str(explicit_target_records_path) if explicit_target_records_path is not None else str(records_path),
                "train_fraction": train_fraction,
                "min_target_per_recording": min_target_per_recording,
                "counts": counts,
                "prior_action_context": {
                    "train_source": "idm_pseudolabel_previous_teacher_forced",
                    "target_source": "d2e_ground_truth_previous_teacher_forced",
                },
            }
        ),
    }
    write_json(output_dir / "fdm_streaming_split_summary.json", payload)
    return payload


def _fdm_torch_config(config: dict[str, Any], split_summary: dict[str, Any], *, output_dir: Path) -> tuple[dict[str, Any], list[str], list[str], str]:
    model_name = str(config.get("model_name", "streaming_compact_fdm"))
    torch_cfg = dict(config.get("torch_idm_config", {}))
    train_record_paths = [str(path) for path in split_summary.get("train_record_paths", []) if path]
    target_record_paths = [str(path) for path in split_summary.get("target_record_paths", []) if path]
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
    if len(train_record_paths) > 1:
        torch_cfg["train_record_paths"] = train_record_paths
        torch_cfg["train_records_glob"] = split_summary.get("train_records_glob")
    if len(target_record_paths) > 1:
        torch_cfg["target_record_paths"] = target_record_paths
        torch_cfg["target_records_glob"] = split_summary.get("target_records_glob")
    return torch_cfg, train_record_paths, target_record_paths, model_name


def _write_fdm_summary_from_torch_summary(
    config: dict[str, Any],
    *,
    split_summary: dict[str, Any],
    torch_summary: dict[str, Any],
    output_dir: Path,
    train_record_paths: list[str],
    target_record_paths: list[str],
    model_name: str,
    recovered_from_torch_checkpoint: str | None = None,
) -> dict[str, Any]:
    label_hash = str(split_summary["labels_sha256"])
    config_fingerprint = stable_hash_json(config)
    resolved_config_path = output_dir / "resolved_config.json"
    resolved_payload = {
        "schema": "streaming_fdm_resolved_config.v1",
        "model": model_name,
        "config": config,
        "config_fingerprint": config_fingerprint,
    }
    if recovered_from_torch_checkpoint:
        resolved_payload["recovered_from_torch_checkpoint"] = recovered_from_torch_checkpoint
    write_json(resolved_config_path, resolved_payload)
    labels_path = Path(config["labels_path"])
    source_idm_metadata_path = config.get("source_idm_metadata") or str(labels_path.parent / "checkpoint_metadata.json")
    data_universe_path = config.get("data_universe")
    split_contract_path = config.get("split_contract")
    split_counts = split_summary.get("counts", {})
    checkpoint = {
        "schema": "fdm_checkpoint_metadata.v1",
        "model": model_name,
        "label_source": "idm_pseudolabel",
        "source_label_artifact": str(config["labels_path"]),
        "source_label_sha256": label_hash,
        "source_idm_metadata": _file_artifact_metadata(source_idm_metadata_path),
        "source_idm_fingerprint": _json_fingerprint(source_idm_metadata_path),
        "config_fingerprint": config_fingerprint,
        "config_path": str(config.get("config_path", "")),
        "resolved_config_path": str(resolved_config_path),
        "data_universe": _file_artifact_metadata(data_universe_path),
        "data_universe_fingerprint": _json_fingerprint(data_universe_path),
        "split_contract": _file_artifact_metadata(split_contract_path),
        "split_contract_fingerprint": _json_fingerprint(split_contract_path),
        "split_id": str(config.get("split_id") or _json_fingerprint(split_contract_path) or "d2e_full_split_contract"),
        "source_namespace": str(config.get("source_namespace", "d2e_full_corpus")),
        "source_ids": sorted((split_counts.get("source_ids") or {}).keys()),
        "resolution_tiers": sorted((split_counts.get("resolution_tiers") or {}).keys()),
        "split_names": sorted((split_counts.get("split_names") or {}).keys()),
        "eval_split_tags": sorted((split_counts.get("eval_split_tags") or {}).keys()),
        "target_source_ids": sorted((split_counts.get("target_source_ids") or {}).keys()),
        "target_resolution_tiers": sorted((split_counts.get("target_resolution_tiers") or {}).keys()),
        "target_split_names": sorted((split_counts.get("target_split_names") or {}).keys()),
        "target_games": sorted((split_counts.get("target_games") or {}).keys()),
        "target_eval_split_tags": sorted((split_counts.get("target_eval_split_tags") or {}).keys()),
        "predictions_path": str(torch_summary["predictions_path"]),
        "num_training_examples": int(split_summary["counts"]["train"]),
        "oracle_ground_truth_control": False,
        "records_path": str(config["records_path"]),
        "train_records_path": str(split_summary["train_records_path"]),
        "target_records_path": str(split_summary["target_records_path"]),
        "train_record_paths": train_record_paths or [str(split_summary["train_records_path"])],
        "target_record_paths": target_record_paths or [str(split_summary["target_records_path"])],
        "train_records_glob": split_summary.get("train_records_glob"),
        "target_records_glob": split_summary.get("target_records_glob"),
        "target_examples": int(split_summary["counts"]["target"]),
        "split_summary_path": str(output_dir / "fdm_streaming_split_summary.json"),
        "torch_checkpoint_metadata": torch_summary["metadata"],
        "statistical_comparison_path": torch_summary["metadata"].get("statistical_comparison_path"),
        "metrics_path": torch_summary["metadata"].get("metrics_path"),
        "convergence_report_path": torch_summary["metadata"].get("convergence_report_path"),
        "convergence_plateau_met": bool(torch_summary["metadata"].get("convergence_plateau_met", False)),
        "dataset_fingerprint": split_summary["dataset_fingerprint"],
    }
    if recovered_from_torch_checkpoint:
        checkpoint["recovery"] = {
            "schema": "streaming_fdm_checkpoint_recovery.v1",
            "source_torch_checkpoint_path": recovered_from_torch_checkpoint,
            "torch_summary_path": str(output_dir / "torch_train_summary.json"),
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
    if recovered_from_torch_checkpoint:
        summary["recovered_from_torch_checkpoint"] = recovered_from_torch_checkpoint
    write_json(config.get("summary_out", output_dir / "summary.json"), summary)
    if config.get("artifact_summary_out"):
        write_json(config["artifact_summary_out"], summary)
    return summary


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
            init_kwargs: dict[str, Any] = {"backend": backend}
            timeout_seconds = (
                torch_cfg_for_dist.get("distributed_timeout_seconds")
                or config.get("distributed_timeout_seconds")
                or os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS")
                or os.environ.get("TORCH_DIST_TIMEOUT_SECONDS")
            )
            if timeout_seconds is not None:
                timeout = float(timeout_seconds)
                if timeout <= 0:
                    raise ValueError("distributed_timeout_seconds must be positive")
                init_kwargs["timeout"] = timedelta(seconds=timeout)
            torch.distributed.init_process_group(**init_kwargs)
    if rank == 0:
        split_summary = materialize_fdm_streaming_splits(config)
    if world_size > 1:
        assert torch is not None
        torch.distributed.barrier()
    if rank != 0:
        split_summary = read_json(split_summary_path)
    torch_cfg, train_record_paths, target_record_paths, model_name = _fdm_torch_config(config, split_summary, output_dir=output_dir)
    torch_summary = train_streaming_idm(torch_cfg)
    if torch_summary.get("schema") == "streaming_idm_worker_summary.v1":
        return {
            "schema": "streaming_fdm_worker_summary.v1",
            "rank": torch_summary.get("rank"),
            "world_size": torch_summary.get("world_size"),
            "status": "worker_complete",
        }
    return _write_fdm_summary_from_torch_summary(
        config,
        split_summary=split_summary,
        torch_summary=torch_summary,
        output_dir=output_dir,
        train_record_paths=train_record_paths,
        target_record_paths=target_record_paths,
        model_name=model_name,
    )


def recover_streaming_fdm_outputs_from_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config.get("output_dir", "outputs/fdm_streaming_d2e_full_compact"))
    split_summary_path = output_dir / "fdm_streaming_split_summary.json"
    if split_summary_path.exists():
        split_summary = read_json(split_summary_path)
    elif bool(config.get("materialize_split_if_missing", False)):
        split_summary = materialize_fdm_streaming_splits(config)
    else:
        raise FileNotFoundError(f"missing FDM split summary for recovery: {split_summary_path}")
    torch_cfg, train_record_paths, target_record_paths, model_name = _fdm_torch_config(config, split_summary, output_dir=output_dir)
    torch_cfg["checkpoint_path"] = str(config.get("torch_checkpoint_path") or Path(torch_cfg["output_dir"]) / "checkpoint.pt")
    torch_cfg["summary_out"] = str(output_dir / "torch_train_summary.json")
    torch_cfg["resume_predictions"] = bool(config.get("resume_predictions", torch_cfg.get("resume_predictions", True)))
    torch_recovery = recover_streaming_idm_outputs_from_checkpoint(torch_cfg)
    torch_summary = read_json(output_dir / "torch_train_summary.json")
    summary = _write_fdm_summary_from_torch_summary(
        config,
        split_summary=split_summary,
        torch_summary=torch_summary,
        output_dir=output_dir,
        train_record_paths=train_record_paths,
        target_record_paths=target_record_paths,
        model_name=model_name,
        recovered_from_torch_checkpoint=str(torch_cfg["checkpoint_path"]),
    )
    return {
        "schema": "streaming_fdm_checkpoint_recovery_summary.v1",
        "status": "pass",
        "torch_recovery": torch_recovery,
        "checkpoint_metadata_path": str(output_dir / "checkpoint_metadata.json"),
        "summary_path": str(config.get("summary_out", output_dir / "summary.json")),
        "artifact_summary_path": str(config.get("artifact_summary_out", "")) if config.get("artifact_summary_out") else None,
        "target_examples": int(summary["checkpoint"]["target_examples"]),
        "prediction_resume": torch_recovery.get("prediction_resume", {}),
    }


def train_streaming_fdm_from_config(path: str | Path) -> dict[str, Any]:
    return train_streaming_fdm(load_config(path))
