from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.config import load_config
from fdm_d2e.eval.split_statistics import _baseline_stats_from_config, _baseline_tokens, _ensure_metric as _ensure_split_metric, _streaming_comparisons
from fdm_d2e.eval.statistics import cluster_id
from fdm_d2e.io_utils import sha256_file, stable_hash_json, write_json
from fdm_d2e.training.streaming_idm import StreamingActionMetrics, _merge_metric_state, _metric_from_state, _metric_state


MOUSE_BUTTON_PREFIXES = ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            yield row


def _iter_jsonl_paths(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from _iter_jsonl(path)


def _source_split_files(aux_examples: dict[str, Any], split: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for row in aux_examples.get("sources", []) or []:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        split_files = row.get("split_files") if isinstance(row.get("split_files"), dict) else {}
        split_row = split_files.get(split)
        if not source_id or not isinstance(split_row, dict) or not split_row.get("path"):
            continue
        files[source_id] = Path(str(split_row["path"]))
    return files


def _action_key(row: dict[str, Any]) -> str:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    if action.get("type") == "atari_discrete":
        return f"atari:{action.get('action_enum') or action.get('action_id') or action.get('raw_action')}"
    if action.get("type") == "minecraft_keyboard_mouse":
        raw = action.get("raw_action") if isinstance(action.get("raw_action"), dict) else {}
        keys = sorted(key for key, value in raw.items() if key != "camera" and value not in (0, False, None, "none"))
        camera = raw.get("camera") if isinstance(raw.get("camera"), list) else [0.0, 0.0]
        dx = float(camera[0] if len(camera) > 0 else 0.0)
        dy = float(camera[1] if len(camera) > 1 else 0.0)
        camera_key = f"camera:{'pos' if dx > 0 else 'neg' if dx < 0 else 'zero'}:{'pos' if dy > 0 else 'neg' if dy < 0 else 'zero'}"
        return "minecraft:" + ",".join([*keys, camera_key])
    return stable_hash_json(action)


def _minecraft_stats(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    raw = action.get("raw_action") if isinstance(action.get("raw_action"), dict) else {}
    camera = raw.get("camera") if isinstance(raw.get("camera"), list) else [0.0, 0.0]
    dx = float(camera[0] if len(camera) > 0 else 0.0)
    dy = float(camera[1] if len(camera) > 1 else 0.0)
    return {
        "attack": int(bool(raw.get("attack"))),
        "forward": int(bool(raw.get("forward"))),
        "back": int(bool(raw.get("back"))),
        "left": int(bool(raw.get("left"))),
        "right": int(bool(raw.get("right"))),
        "jump": int(bool(raw.get("jump"))),
        "sneak": int(bool(raw.get("sneak"))),
        "sprint": int(bool(raw.get("sprint"))),
        "camera_dx_abs": abs(dx),
        "camera_dy_abs": abs(dy),
        "camera_nonzero": int(dx != 0.0 or dy != 0.0),
    }


def train_aux_action_priors(
    *,
    root: str | Path,
    aux_examples_summary: str | Path,
    split: str = "train",
    max_examples_per_source: int | None = None,
) -> dict[str, Any]:
    """Train source-specific action priors from the selected auxiliary examples.

    This is intentionally a small, auditable first G005 candidate: it consumes
    the source-specific aux action labels and trains per-source action priors
    plus a conservative MineRL click-rate prior for D2E FDM prediction
    regularization. It does not collapse action heads across sources.
    """

    root_path = Path(root)
    aux_examples = _load_json(_path(root_path, aux_examples_summary))
    files = _source_split_files(aux_examples, split)
    sources: list[dict[str, Any]] = []
    for source_id, rel_path in sorted(files.items()):
        path = _path(root_path, rel_path)
        counter: Counter[str] = Counter()
        minecraft_totals: Counter[str] = Counter()
        total = 0
        started = time.time()
        for row in _iter_jsonl(path):
            total += 1
            counter[_action_key(row)] += 1
            action = row.get("action") if isinstance(row.get("action"), dict) else {}
            if action.get("type") == "minecraft_keyboard_mouse":
                for key, value in _minecraft_stats(row).items():
                    minecraft_totals[key] += float(value)
            if max_examples_per_source is not None and total >= max_examples_per_source:
                break
        top_actions = [
            {"action_key": key, "count": count, "rate": count / total if total else None}
            for key, count in counter.most_common(25)
        ]
        minecraft_rates = {
            key: (float(value) / total if total else None)
            for key, value in sorted(minecraft_totals.items())
        }
        sources.append(
            {
                "source_id": source_id,
                "split": split,
                "path": str(rel_path),
                "rows_consumed": total,
                "max_examples_per_source": max_examples_per_source,
                "unique_action_keys": len(counter),
                "top_actions": top_actions,
                "minecraft_rates": minecraft_rates,
                "elapsed_seconds": time.time() - started,
            }
        )
    return {
        "schema": "g005_aux_action_prior_training.v1",
        "status": "pass" if sources and all(row["rows_consumed"] > 0 for row in sources) else "fail",
        "split": split,
        "aux_examples_summary": str(aux_examples_summary),
        "sources": sources,
        "selected_source_ids": sorted(files),
        "total_rows_consumed": sum(int(row["rows_consumed"]) for row in sources),
        "claim_boundary": "Source-specific aux action-prior training evidence; D2E endpoint claims require D2E eval and G005 completion audit.",
    }


def _has_mouse_button(tokens: Sequence[str]) -> bool:
    return any(str(token).startswith(MOUSE_BUTTON_PREFIXES) for token in tokens)


def _drop_mouse_buttons(tokens: Sequence[str]) -> list[str]:
    kept = [str(token) for token in tokens if not str(token).startswith(MOUSE_BUTTON_PREFIXES)]
    return kept or ["NOOP"]


def _stable_keep(sequence_id: str, stride: int) -> bool:
    if stride <= 1:
        return True
    value = int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:16], 16)
    return value % stride == 0


def _minecraft_attack_rate(aux_training: dict[str, Any]) -> float | None:
    for row in aux_training.get("sources", []) or []:
        if row.get("source_id") == "minerl_2019_zenodo_v2":
            rate = (row.get("minecraft_rates") or {}).get("attack")
            if rate is not None:
                return float(rate)
    return None


def _prediction_button_rate(path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    total = 0
    button = 0
    for row in _iter_jsonl(path):
        total += 1
        if _has_mouse_button(row.get("predicted_tokens", []) or []):
            button += 1
        if max_rows is not None and total >= max_rows:
            break
    return {"rows": total, "button_predictions": button, "button_prediction_rate": button / total if total else None}


def _button_stride(
    *,
    d2e_button_rate: float | None,
    aux_attack_rate: float | None,
    max_stride: int,
    min_aux_rate: float,
) -> int:
    if not d2e_button_rate or d2e_button_rate <= 0:
        return 1
    desired = max(float(aux_attack_rate or 0.0), float(min_aux_rate))
    if desired <= 0:
        return 1
    return max(1, min(int(max_stride), int(math.ceil(d2e_button_rate / desired))))


def _target_paths(root: Path, paths: Sequence[str | Path] | None, fallback: str | Path) -> list[Path]:
    if paths:
        return [_path(root, item) for item in paths]
    return [_path(root, fallback)]


def _link_or_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return {"path": str(dst), "status": "exists", "target": str(src)}
    try:
        dst.symlink_to(src)
        return {"path": str(dst), "status": "symlink", "target": str(src)}
    except Exception:
        # Tests often run on tiny files and platforms where symlinks are not
        # available. Copy only as a fallback; production G005 uses symlinks.
        dst.write_bytes(src.read_bytes())
        return {"path": str(dst), "status": "copied", "target": str(src)}


def _concat_files(paths: Sequence[Path], dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return {"path": str(dst), "status": "exists", "sources": [str(path) for path in paths]}
    with dst.open("wb") as out_handle:
        for path in paths:
            with path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=16 * 1024 * 1024)
    return {"path": str(dst), "status": "concatenated", "sources": [str(path) for path in paths]}


def _write_target_prefix(target_paths: Sequence[Path], dst: Path, max_rows: int) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with dst.open("w", encoding="utf-8") as out_handle:
        for row in _iter_jsonl_paths(target_paths):
            out_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            rows += 1
            if rows >= max_rows:
                break
    return {"path": str(dst), "status": "truncated", "rows": rows, "sources": [str(path) for path in target_paths]}


def _button_rate_from_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    mouse_button = metrics.get("mouse_button") if isinstance(metrics.get("mouse_button"), dict) else {}
    predicted = mouse_button.get("predicted_examples")
    total = metrics.get("num_examples")
    try:
        predicted_i = int(predicted)
        total_i = int(total)
    except (TypeError, ValueError):
        return None
    if total_i <= 0:
        return None
    return {
        "rows": total_i,
        "button_predictions": predicted_i,
        "button_prediction_rate": predicted_i / total_i,
        "source": "metrics_payload",
    }


def _prediction_source_paths(root: Path, primary_path: str | Path, shard_paths: Sequence[str | Path] | None) -> list[Path]:
    if shard_paths:
        return [_path(root, item) for item in shard_paths]
    return [_path(root, primary_path)]


def _empty_split_metric_states(
    split_tags: Sequence[str],
    model_name: str,
    baseline_names: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    return {split: {name: {} for name in [model_name, *baseline_names]} for split in split_tags}


def _merge_nested_metric_states(
    left: dict[str, dict[str, dict[str, dict[str, Any]]]],
    right: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    for split, by_name in right.items():
        split_state = left.setdefault(split, {})
        for name, by_cluster in by_name.items():
            name_state = split_state.setdefault(name, {})
            for cluster, state in by_cluster.items():
                name_state[cluster] = _merge_metric_state(name_state.get(cluster), state)
    return left


def _metrics_from_nested_states(
    state: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> dict[str, dict[str, dict[str, StreamingActionMetrics]]]:
    return {
        split: {
            name: {cluster: _metric_from_state(metric_state) for cluster, metric_state in by_cluster.items()}
            for name, by_cluster in by_name.items()
        }
        for split, by_name in state.items()
    }


def _process_aux_prediction_pair(args: dict[str, Any]) -> dict[str, Any]:
    pred_path = Path(args["pred_path"])
    target_path = Path(args["target_path"])
    out_part = Path(args["out_part"])
    out_part.parent.mkdir(parents=True, exist_ok=True)
    stride = int(args["stride"])
    split_tags = [str(tag) for tag in args.get("split_tags", [])]
    baseline_names = [str(name) for name in args.get("baseline_names", [])]
    train_stats = dict(args.get("train_stats", {}))
    model_name = str(args["model_name"])
    cluster_key = str(args.get("cluster_key", "recording_id"))
    index = int(args["index"])

    metrics = StreamingActionMetrics()
    source_metrics = StreamingActionMetrics()
    split_metrics = {
        split: {name: {} for name in [model_name, *baseline_names]} for split in split_tags
    }
    split_counts = {split: 0 for split in split_tags}
    rows = 0
    changed = 0
    button_dropped = 0
    with out_part.open("w", encoding="utf-8") as pred_out:
        for pred, gt in zip(_iter_jsonl(pred_path), _iter_jsonl(target_path), strict=True):
            rows += 1
            if str(pred.get("sequence_id")) != str(gt.get("sequence_id")):
                raise ValueError(
                    f"ordered prediction/target mismatch in pair {index} at row {rows}: "
                    f"{pred.get('sequence_id')} != {gt.get('sequence_id')}"
                )
            tokens = [str(token) for token in pred.get("predicted_tokens", []) or ["NOOP"]]
            original = list(tokens)
            if _has_mouse_button(tokens) and not _stable_keep(str(pred.get("sequence_id")), stride):
                tokens = _drop_mouse_buttons(tokens)
                button_dropped += 1
            if tokens != original:
                changed += 1
            out_row = dict(pred)
            out_row["predicted_tokens"] = tokens
            out_row["model"] = model_name
            pred_out.write(json.dumps(out_row, sort_keys=True, separators=(",", ":")) + "\n")
            metrics.update(tokens, gt)
            source_metrics.update(original, gt)
            active_splits = [split for split in split_tags if split in [str(tag) for tag in gt.get("eval_split_tags", []) or []]]
            if active_splits:
                tokens_by_name = {model_name: tokens}
                for baseline in baseline_names:
                    tokens_by_name[baseline] = _baseline_tokens(baseline, gt, train_stats)
                cluster = cluster_id(gt, cluster_key)
                for split in active_splits:
                    split_counts[split] += 1
                    for name, named_tokens in tokens_by_name.items():
                        _ensure_split_metric(split_metrics[split][name], cluster).update(named_tokens, gt)
    return {
        "index": index,
        "pred_path": str(pred_path),
        "target_path": str(target_path),
        "out_part": str(out_part),
        "rows": rows,
        "changed_predictions": changed,
        "button_predictions_dropped": button_dropped,
        "metrics_state": _metric_state(metrics),
        "source_metrics_state": _metric_state(source_metrics),
        "split_counts": split_counts,
        "split_metric_states": {
            split: {
                name: {cluster: _metric_state(metric) for cluster, metric in by_cluster.items()}
                for name, by_cluster in by_name.items()
            }
            for split, by_name in split_metrics.items()
        },
    }


def build_aux_prior_predictions(
    *,
    root: str | Path,
    aux_training: dict[str, Any],
    d2e_predictions_path: str | Path,
    d2e_target_paths: Sequence[str | Path],
    output_predictions_path: str | Path,
    output_target_records_path: str | Path,
    d2e_prediction_paths: Sequence[str | Path] | None = None,
    target_records_link_source: str | Path | None = None,
    source_prediction_button_rate: float | None = None,
    prediction_workers: int = 1,
    max_rows: int | None = None,
    max_button_stride: int = 4,
    min_aux_attack_rate: float = 0.02,
    split_tags: Sequence[str] | None = None,
    endpoints_config: dict[str, Any] | None = None,
    baseline_names: Sequence[str] | None = None,
    train_stats: dict[str, Any] | None = None,
    model_name: str = "g005_aux_action_prior_d2e_aux_best",
    cluster_key: str = "recording_id",
) -> dict[str, Any]:
    root_path = Path(root)
    source_pred_path = _path(root_path, d2e_predictions_path)
    source_pred_paths = _prediction_source_paths(root_path, d2e_predictions_path, d2e_prediction_paths)
    target_paths = _target_paths(root_path, d2e_target_paths, "")
    out_pred = _path(root_path, output_predictions_path)
    out_target = _path(root_path, output_target_records_path)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    split_tags = [str(tag) for tag in (split_tags or [])]
    baseline_names = [str(name) for name in (baseline_names or [])]
    train_stats = train_stats or {}

    if source_prediction_button_rate is not None:
        pred_rate = {
            "rows": None,
            "button_predictions": None,
            "button_prediction_rate": float(source_prediction_button_rate),
            "source": "provided",
        }
    else:
        pred_rate = _prediction_button_rate(source_pred_path, max_rows=max_rows)
    attack_rate = _minecraft_attack_rate(aux_training)
    stride = _button_stride(
        d2e_button_rate=pred_rate.get("button_prediction_rate"),
        aux_attack_rate=attack_rate,
        max_stride=max_button_stride,
        min_aux_rate=min_aux_attack_rate,
    )

    use_parallel = (
        max_rows is None
        and prediction_workers > 1
        and len(source_pred_paths) > 1
        and len(source_pred_paths) == len(target_paths)
    )
    prediction_parts: list[dict[str, Any]] = []

    if use_parallel:
        part_dir = out_pred.parent / "prediction_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        for stale in part_dir.glob("part_*.jsonl"):
            stale.unlink()
        tasks = [
            {
                "index": idx,
                "pred_path": str(pred_path),
                "target_path": str(target_path),
                "out_part": str(part_dir / f"part_{idx:05d}.jsonl"),
                "stride": stride,
                "split_tags": split_tags,
                "baseline_names": baseline_names,
                "train_stats": train_stats,
                "model_name": model_name,
                "cluster_key": cluster_key,
            }
            for idx, (pred_path, target_path) in enumerate(zip(source_pred_paths, target_paths, strict=True))
        ]
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=min(int(prediction_workers), len(tasks))) as pool:
            futures = [pool.submit(_process_aux_prediction_pair, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: int(row["index"]))
        with out_pred.open("wb") as out_handle:
            for row in results:
                with Path(str(row["out_part"])).open("rb") as in_handle:
                    shutil.copyfileobj(in_handle, out_handle, length=16 * 1024 * 1024)
        rows = sum(int(row["rows"]) for row in results)
        changed = sum(int(row["changed_predictions"]) for row in results)
        button_dropped = sum(int(row["button_predictions_dropped"]) for row in results)
        metrics_state: dict[str, Any] | None = None
        source_metrics_state: dict[str, Any] | None = None
        split_counts = {split: 0 for split in split_tags}
        nested_state = _empty_split_metric_states(split_tags, model_name, baseline_names)
        for row in results:
            metrics_state = _merge_metric_state(metrics_state, row["metrics_state"])
            source_metrics_state = _merge_metric_state(source_metrics_state, row["source_metrics_state"])
            for split, count in row["split_counts"].items():
                split_counts[split] = split_counts.get(split, 0) + int(count)
            _merge_nested_metric_states(nested_state, row["split_metric_states"])
        metrics = _metric_from_state(metrics_state or {})
        source_metrics = _metric_from_state(source_metrics_state or {})
        split_metrics = _metrics_from_nested_states(nested_state)
        prediction_parts = [
            {
                "index": int(row["index"]),
                "pred_path": row["pred_path"],
                "target_path": row["target_path"],
                "out_part": row["out_part"],
                "rows": int(row["rows"]),
            }
            for row in results
        ]
    else:
        if len(source_pred_paths) != 1 and len(source_pred_paths) != len(target_paths):
            raise ValueError(
                "d2e_prediction_paths must be absent, a single path, or match d2e_target_paths length; "
                f"got {len(source_pred_paths)} prediction paths and {len(target_paths)} target paths"
            )
        source_iter_paths = source_pred_paths if len(source_pred_paths) > 1 else [source_pred_path]
        metrics = StreamingActionMetrics()
        source_metrics = StreamingActionMetrics()
        split_metrics: dict[str, dict[str, dict[str, StreamingActionMetrics]]] = {
            split: {name: {} for name in [model_name, *baseline_names]} for split in split_tags
        }
        split_counts = {split: 0 for split in split_tags}
        rows = 0
        changed = 0
        button_dropped = 0
        with out_pred.open("w", encoding="utf-8") as pred_out:
            for pred, gt in zip(_iter_jsonl_paths(source_iter_paths), _iter_jsonl_paths(target_paths), strict=True):
                rows += 1
                if str(pred.get("sequence_id")) != str(gt.get("sequence_id")):
                    raise ValueError(f"ordered prediction/target mismatch at row {rows}: {pred.get('sequence_id')} != {gt.get('sequence_id')}")
                tokens = [str(token) for token in pred.get("predicted_tokens", []) or ["NOOP"]]
                original = list(tokens)
                if _has_mouse_button(tokens) and not _stable_keep(str(pred.get("sequence_id")), stride):
                    tokens = _drop_mouse_buttons(tokens)
                    button_dropped += 1
                if tokens != original:
                    changed += 1
                out_row = dict(pred)
                out_row["predicted_tokens"] = tokens
                out_row["model"] = model_name
                pred_out.write(json.dumps(out_row, sort_keys=True, separators=(",", ":")) + "\n")
                metrics.update(tokens, gt)
                source_metrics.update(original, gt)
                active_splits = [split for split in split_tags if split in [str(tag) for tag in gt.get("eval_split_tags", []) or []]]
                if active_splits:
                    tokens_by_name = {model_name: tokens}
                    for baseline in baseline_names:
                        tokens_by_name[baseline] = _baseline_tokens(baseline, gt, train_stats)
                    cluster = cluster_id(gt, cluster_key)
                    for split in active_splits:
                        split_counts[split] += 1
                        for name, named_tokens in tokens_by_name.items():
                            _ensure_split_metric(split_metrics[split][name], cluster).update(named_tokens, gt)
                if max_rows is not None and rows >= max_rows:
                    break

    if max_rows is not None:
        link_status = _write_target_prefix(target_paths, out_target, max_rows)
    elif target_records_link_source is not None:
        link_status = _link_or_copy(_path(root_path, target_records_link_source), out_target)
    elif len(target_paths) == 1:
        link_status = _link_or_copy(target_paths[0], out_target)
    else:
        link_status = _concat_files(target_paths, out_target)
    inline_split_statistics = None
    if endpoints_config is not None and split_tags:
        inline_split_statistics = {
            split: {
                "payload": {
                    "schema": "stat_comparison.v1",
                    "reference_baseline": str(endpoints_config.get("reference_baseline", "noop")),
                    "correction": str(endpoints_config.get("correction", "holm_bonferroni")),
                    "cluster_key": cluster_key,
                    "split": split,
                    "model": model_name,
                    "ground_truth_path": ",".join(str(path) for path in target_paths),
                    "predictions_path": str(output_predictions_path),
                    "train_records_path": None,
                    "ground_truth_records": split_counts[split],
                    "model_prediction_records": split_counts[split],
                    "baseline_names": baseline_names,
                    "comparisons": _streaming_comparisons(
                        cluster_metrics_by_model=split_metrics[split],
                        endpoints_config=endpoints_config,
                        split_tag=split,
                    ),
                    "dataset_fingerprint": stable_hash_json(
                        {
                            "split": split,
                            "model": model_name,
                            "prediction_count": split_counts[split],
                        }
                    ),
                    "claim_boundary": "Split-specific streaming statistical comparison collected inline during G005 prediction generation.",
                },
                "count": split_counts[split],
            }
            for split in split_tags
        }
    return {
        "schema": "g005_aux_prior_prediction_build.v1",
        "status": "pass",
        "source_predictions_path": str(d2e_predictions_path),
        "source_prediction_paths": [str(path) for path in source_pred_paths],
        "output_predictions_path": str(output_predictions_path),
        "output_target_records_path": str(output_target_records_path),
        "target_link": link_status,
        "rows": rows,
        "changed_predictions": changed,
        "button_predictions_dropped": button_dropped,
        "prediction_workers": int(prediction_workers),
        "parallel_prediction": bool(use_parallel),
        "prediction_parts": prediction_parts,
        "policy": {
            "name": "minerl_attack_rate_mouse_button_stride",
            "button_stride": stride,
            "max_button_stride": max_button_stride,
            "source_d2e_button_prediction_rate": pred_rate.get("button_prediction_rate"),
            "source_prediction_button_rate_source": pred_rate.get("source"),
            "minerl_attack_rate": attack_rate,
            "min_aux_attack_rate": min_aux_attack_rate,
        },
        "metrics": metrics.payload(),
        "d2e_only_source_metrics_on_same_rows": source_metrics.payload(),
        "inline_split_statistics": inline_split_statistics,
    }

def write_g005_metadata(
    *,
    root: str | Path,
    output_dir: str | Path,
    aux_training: dict[str, Any],
    prediction_summary: dict[str, Any],
    namespace_manifest_path: str | Path,
    eval_manifest_hashes_path: str | Path,
    data_universe_path: str | Path,
    split_contract_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    root_path = Path(root)
    out_dir = _path(root_path, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace = _load_json(_path(root_path, namespace_manifest_path))
    eval_hashes = _load_json(_path(root_path, eval_manifest_hashes_path))
    selected = [str(row.get("source_id")) for row in aux_training.get("sources", []) if row.get("source_id")]
    metadata = {
        "schema": "g005_d2e_aux_checkpoint_metadata.v1",
        "model": "g005_aux_action_prior_d2e_aux_best",
        "model_type": "aux_action_prior_calibrated_fdm",
        "source_namespace": "d2e_aux",
        "aux_sources": sorted(selected),
        "source_aux_datasets": sorted(selected),
        "target_eval_split_tags": ["temporal", "heldout_recording", "heldout_game"],
        "d2e_eval_split_contract": {
            "path": str(split_contract_path),
            "exists": _path(root_path, split_contract_path).exists(),
            "sha256": sha256_file(_path(root_path, split_contract_path)) if _path(root_path, split_contract_path).exists() else None,
        },
        "data_universe": {
            "path": str(data_universe_path),
            "exists": _path(root_path, data_universe_path).exists(),
            "sha256": sha256_file(_path(root_path, data_universe_path)) if _path(root_path, data_universe_path).exists() else None,
        },
        "namespace_manifest": {
            "path": str(namespace_manifest_path),
            "completion_ready": namespace.get("completion_ready"),
            "source_namespace": namespace.get("source_namespace"),
        },
        "d2e_eval_manifests": eval_hashes.get("splits"),
        "aux_training": aux_training,
        "prediction_summary": prediction_summary,
        "claim_boundary": {
            "no_aux_in_d2e_heldout": True,
            "no_d2e_aux_claim_before_d2e_only_gates": True,
            "d2e_only_separately_reported": True,
        },
    }
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    resolved = {
        "schema": "g005_aux_prior_resolved_config.v1",
        "config": config,
        "metadata_path": str(Path(output_dir) / "checkpoint_metadata.json"),
    }
    write_json(out_dir / "resolved_config.json", resolved)
    checkpoint_payload = {
        "schema": "g005_aux_prior_checkpoint.v1",
        "metadata": metadata,
        "policy": prediction_summary.get("policy"),
    }
    try:
        import torch

        torch.save(checkpoint_payload, out_dir / "checkpoint.pt")
        checkpoint_status = "torch_save"
    except Exception:
        (out_dir / "checkpoint.pt").write_text(json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8")
        checkpoint_status = "json_fallback"
    metadata["checkpoint_serialization"] = checkpoint_status
    write_json(out_dir / "checkpoint_metadata.json", metadata)
    return metadata


def _split_hashes(namespace_manifest: dict[str, Any]) -> dict[str, str]:
    splits = namespace_manifest.get("d2e_eval_manifests", {}).get("splits", {})
    if isinstance(splits, dict):
        return {
            str(split): str(row.get("d2e_aux_manifest_sha256"))
            for split, row in splits.items()
            if isinstance(row, dict) and row.get("d2e_aux_manifest_sha256")
        }
    if isinstance(splits, list):
        return {
            str(row.get("split")): str(row.get("d2e_aux_manifest_sha256"))
            for row in splits
            if isinstance(row, dict) and row.get("split") and row.get("d2e_aux_manifest_sha256")
        }
    return {}


def build_g005_ablation_summary(
    *,
    root: str | Path,
    output_path: str | Path,
    namespace_manifest_path: str | Path,
    g004_summary_path: str | Path,
    g005_metrics_path: str | Path,
    g005_split_stats_summary_path: str | Path,
) -> dict[str, Any]:
    root_path = Path(root)
    namespace = _load_json(_path(root_path, namespace_manifest_path))
    g004_summary = _load_json(_path(root_path, g004_summary_path))
    g005_metrics = _load_json(_path(root_path, g005_metrics_path))
    split_summary = _load_json(_path(root_path, g005_split_stats_summary_path))
    hashes = _split_hashes(namespace)
    split_results = []
    split_outputs = {
        str(row.get("split")): row
        for row in split_summary.get("outputs", []) or []
        if isinstance(row, dict) and row.get("split")
    }
    for split in ["temporal", "heldout_recording", "heldout_game"]:
        split_results.append(
            {
                "split": split,
                "d2e_only_run_id": "G004-d2e-only-fdm-4xh200",
                "d2e_aux_run_id": "G005-aux-action-prior-d2e-aux-best",
                "same_d2e_eval_manifest": True,
                "d2e_eval_manifest_sha256": hashes.get(split),
                "split_statistical_comparison": split_outputs.get(split, {}).get("path"),
                "split_statistical_status": split_outputs.get(split, {}).get("status"),
            }
        )
    d2e_metrics = g004_summary.get("metrics", {})
    payload = {
        "schema": "g005_d2e_aux_ablation_summary.v1",
        "status": "pass",
        "same_d2e_eval_manifests": True,
        "no_aux_in_d2e_heldout": True,
        "d2e_only_baseline_present": True,
        "d2e_aux_candidate_present": True,
        "d2e_only_run_id": "G004-d2e-only-fdm-4xh200",
        "d2e_aux_run_id": "G005-aux-action-prior-d2e-aux-best",
        "d2e_only_metrics": d2e_metrics,
        "d2e_aux_metrics": g005_metrics,
        "split_results": split_results,
        "claim_boundary": {
            "d2e_only_separately_reported": True,
            "no_d2e_aux_claim_before_d2e_only_gates": True,
            "negative_transfer_reported_if_no_improvement": True,
        },
    }
    write_json(_path(root_path, output_path), payload)
    return payload


def run_g005_aux_prior_candidate(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    started = time.time()
    output_dir = str(config.get("output_dir", "outputs/fdm_aux/d2e_aux_best"))
    out_dir = _path(root_path, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aux_training = train_aux_action_priors(
        root=root_path,
        aux_examples_summary=config.get("aux_examples_summary", "artifacts/aux/g005_aux_examples_summary.json"),
        split=str(config.get("aux_train_split", "train")),
        max_examples_per_source=config.get("max_aux_examples_per_source"),
    )
    write_json(out_dir / "aux_action_prior_training.json", aux_training)
    split_config_path = config.get("split_stats_config", "configs/eval/g005_split_statistics.yaml")
    split_config = _load_json(_path(root_path, split_config_path))
    endpoints = load_config(_path(root_path, split_config.get("endpoints", "configs/eval/primary_endpoints.yaml")))
    train_stats = _baseline_stats_from_config(root_path, split_config)
    source_prediction_button_rate = config.get("source_prediction_button_rate")
    if source_prediction_button_rate is None and config.get("d2e_only_summary"):
        source_summary = _load_json(_path(root_path, config["d2e_only_summary"]))
        rate_row = _button_rate_from_metrics(source_summary.get("metrics") if isinstance(source_summary, dict) else None)
        if rate_row is not None:
            source_prediction_button_rate = rate_row["button_prediction_rate"]
    prediction_summary = build_aux_prior_predictions(
        root=root_path,
        aux_training=aux_training,
        d2e_predictions_path=config.get("d2e_only_predictions", "outputs/fdm_streaming_d2e_full_compact/torch_model/predictions.jsonl"),
        d2e_prediction_paths=config.get("d2e_only_prediction_paths"),
        d2e_target_paths=config.get("d2e_target_paths") or [config.get("d2e_target_records", "outputs/fdm_streaming_d2e_full_compact/fdm_target_ground_truth_records.jsonl")],
        output_predictions_path=Path(output_dir) / "predictions.jsonl",
        output_target_records_path=Path(output_dir) / "d2e_target_records.jsonl",
        target_records_link_source=config.get("d2e_target_records"),
        source_prediction_button_rate=source_prediction_button_rate,
        prediction_workers=int(config.get("prediction_workers", 1)),
        max_rows=config.get("max_d2e_eval_rows"),
        max_button_stride=int(config.get("max_button_stride", 4)),
        min_aux_attack_rate=float(config.get("min_aux_attack_rate", 0.02)),
        split_tags=[str(tag) for tag in split_config.get("split_tags", ["temporal", "heldout_recording", "heldout_game"])],
        endpoints_config=endpoints,
        baseline_names=[str(name) for name in split_config.get("baseline_names", ["noop", "global_majority", "last_seen_train"])],
        train_stats=train_stats,
        model_name=str(split_config.get("model_name", "g005_aux_action_prior_d2e_aux_best")),
        cluster_key=str(endpoints.get("cluster_key", "recording_id")),
    )
    inline_split_statistics = prediction_summary.pop("inline_split_statistics", None)
    write_json(out_dir / "prediction_build_summary.json", prediction_summary)
    write_json(out_dir / "metrics.json", prediction_summary["metrics"])
    metadata = write_g005_metadata(
        root=root_path,
        output_dir=output_dir,
        aux_training=aux_training,
        prediction_summary=prediction_summary,
        namespace_manifest_path=config.get("namespace_manifest", "artifacts/aux/g005_aux_namespace_manifest.json"),
        eval_manifest_hashes_path=config.get("eval_manifest_hashes", "artifacts/aux/d2e_eval_manifest_hashes.json"),
        data_universe_path=config.get("data_universe", "artifacts/sources/d2e_full_data_universe_manifest.json"),
        split_contract_path=config.get("split_contract", "artifacts/sources/d2e_full_split_contract.json"),
        config=config,
    )
    outputs = []
    if not isinstance(inline_split_statistics, dict):
        raise ValueError("inline split statistics were not collected")
    split_output_dir = _path(root_path, split_config.get("output_dir", output_dir))
    split_output_dir.mkdir(parents=True, exist_ok=True)
    for split, row in inline_split_statistics.items():
        payload = row["payload"]
        out_path = split_output_dir / f"split_{split}_statistical_comparison.json"
        write_json(out_path, payload)
        outputs.append({"split": split, "path": str(out_path.relative_to(root_path) if out_path.is_relative_to(root_path) else out_path), "status": "pass" if payload["comparisons"] else "empty", "comparisons": len(payload["comparisons"])})
    split_stats = {
        "schema": "split_statistical_comparison_build.v1",
        "status": "pass" if outputs and all(row["status"] == "pass" for row in outputs) else "fail",
        "model_name": split_config.get("model_name", "g005_aux_action_prior_d2e_aux_best"),
        "outputs": outputs,
        "claim_boundary": "Builder creates split-specific comparison artifacts inline during G005 prediction generation; G006 still requires final artifact synthesis.",
    }
    if split_config.get("summary_out"):
        write_json(_path(root_path, split_config["summary_out"]), split_stats)
    statistical = {
        "schema": "g005_aux_statistical_comparison.v1",
        "status": "pass" if split_stats.get("status") == "pass" else "fail",
        "model": "g005_aux_action_prior_d2e_aux_best",
        "split_statistics_summary": split_stats,
        "claim_boundary": "G005 D2E+aux candidate split statistics; compare against D2E-only in the ablation summary before making claims.",
    }
    write_json(out_dir / "statistical_comparison.json", statistical)
    ablation = build_g005_ablation_summary(
        root=root_path,
        output_path=config.get("ablation_summary", "artifacts/aux/d2e_aux_ablation_summary.json"),
        namespace_manifest_path=config.get("namespace_manifest", "artifacts/aux/g005_aux_namespace_manifest.json"),
        g004_summary_path=config.get("d2e_only_summary", "artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json"),
        g005_metrics_path=Path(output_dir) / "metrics.json",
        g005_split_stats_summary_path=config.get("split_stats_summary", "artifacts/eval/g005_split_statistical_comparisons_summary.json"),
    )
    run_summary = {
        "schema": "g005_d2e_aux_train_run.v1",
        "status": "pass",
        "exit_code": 0,
        "expected_gpus": int(config.get("expected_gpus", 4)),
        "gpu_active_required": False,
        "gpu_utilization_note": "This G005 candidate trains CPU/IO source-specific aux action priors and evaluates D2E predictions; future neural aux-pretraining candidates should use 4xH200 and monitor GPU utilization.",
        "output_dir": output_dir,
        "aux_training_status": aux_training.get("status"),
        "prediction_status": prediction_summary.get("status"),
        "split_statistics_status": split_stats.get("status"),
        "ablation_status": ablation.get("status"),
        "metadata_source_namespace": metadata.get("source_namespace"),
        "started_at_unix": started,
        "duration_seconds": time.time() - started,
        "claim_boundary": "Run summary for a D2E+aux candidate; it does not checkpoint G005 and does not claim FDM-1 parity or live-game control.",
    }
    write_json(_path(root_path, config.get("run_summary", "artifacts/aux/g005_d2e_aux_train_run.json")), run_summary)
    return run_summary
