#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics
from fdm_d2e.io_utils import ensure_dir, write_json
from fdm_d2e.training.temporal_masked_diffusion_idm_trainer import (
    _build_temporal_model,
    _adapt_temporal_family_budget_to_unlabeled_distribution,
    _build_temporal_retrieval_prior_index,
    _calibrate_temporal_family_non_noop_budget,
    _calibrate_temporal_non_noop_budget,
    _candidate_family_diagnostics,
    _candidate_token_prior_weights,
    _collect_temporal_probability_rows,
    _configured_video_feature_dim,
    _expand_paths,
    _iter_jsonl,
    _precompute_features,
    _precompute_video_cache_features,
    _predict_temporal_tokens_batch,
    _raw_video_frame_offsets,
)
from fdm_d2e.training.torch_idm import require_torch, torch_available


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set override has empty key: {item!r}")
        merged[key] = _parse_value(raw_value)
    return merged


def _load_checkpoint(path: Path, torch: Any, *, device: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_temporal_masked_diffusion_idm(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    summary_out: Path | None,
    config_path: Path | None = None,
    overrides: list[str],
    force_cpu: bool = False,
) -> dict[str, Any]:
    if not torch_available():
        raise RuntimeError("torch unavailable; run `uv sync --extra train` or use the MLXP training image")
    torch = require_torch()
    start = time.time()
    output_dir = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path, torch, device=device)
    config = dict(checkpoint.get("config") or {})
    if config_path is not None:
        config.update(json.loads(config_path.read_text()))
    config = _apply_overrides(config, overrides)
    config["output_dir"] = str(output_dir)
    if summary_out is not None:
        config["summary_out"] = str(summary_out)
    if force_cpu:
        config["force_cpu"] = True

    train_paths = _expand_paths(config.get("train_records")) + _expand_paths(config.get("train_record_paths"))
    target_paths = _expand_paths(config.get("target_records")) + _expand_paths(config.get("target_record_paths"))
    max_train_rows = int(config["max_train_rows"]) if config.get("max_train_rows") is not None else None
    max_target_rows = int(config["max_target_rows"]) if config.get("max_target_rows") is not None else None
    train_rows = list(_iter_jsonl(train_paths, max_rows=max_train_rows))
    target_rows = list(_iter_jsonl(target_paths, max_rows=max_target_rows))
    if not train_rows:
        raise ValueError("no train rows found for temporal masked-diffusion IDM prediction")
    if not target_rows:
        raise ValueError("no target rows found for temporal masked-diffusion IDM prediction")

    calibration_rows: list[dict[str, Any]] = []
    fit_rows = train_rows
    if bool(config.get("calibrate_non_noop_budget", config.get("non_noop_budgeted_unmasking", False))) and len(train_rows) >= 10:
        calibration_fraction = float(config.get("temporal_calibration_fraction", config.get("factorized_calibration_fraction", 0.0)) or 0.0)
        calibration_max_rows = int(config.get("temporal_calibration_max_rows", config.get("factorized_calibration_max_rows", 2000)))
        if calibration_fraction > 0.0:
            calibration_count = min(calibration_max_rows, max(1, int(len(train_rows) * calibration_fraction)))
            calibration_rows = train_rows[-calibration_count:]
            fit_rows = train_rows[:-calibration_count] or train_rows

    vocab = list(checkpoint["vocab"])
    max_slots = int(checkpoint.get("max_slots", config.get("max_action_tokens_per_bin", config.get("max_slots", 16))))
    offsets = [int(value) for value in checkpoint.get("temporal_offsets", config.get("temporal_offsets", [-2, -1, 0, 1, 2]))]
    feature_dim = int(checkpoint.get("feature_dim", _configured_video_feature_dim(config)))
    config = {**config, "video_feature_dim": feature_dim, "max_slots": max_slots, "temporal_offsets": offsets}
    feature_source = str(config.get("video_feature_source", "json")).lower()
    if feature_source in {"video_idm_cache", "raw_video_cache"}:
        train_features = _precompute_video_cache_features(train_paths, split_name="train", config=config, max_rows=len(train_rows))
        fit_features = train_features[: len(fit_rows)]
        calibration_features = train_features[len(fit_rows) : len(fit_rows) + len(calibration_rows)] if calibration_rows else []
        target_features = _precompute_video_cache_features(target_paths, split_name="target", config=config, max_rows=len(target_rows))
    else:
        fit_features = _precompute_features(fit_rows, config=config)
        calibration_features = _precompute_features(calibration_rows, config=config) if calibration_rows else []
        target_features = _precompute_features(target_rows, config=config)

    model = _build_temporal_model(
        torch,
        video_dim=feature_dim,
        vocab_size=len(vocab),
        max_slots=max_slots,
        offsets=offsets,
        config=config,
        vocab=vocab,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    retrieval_index = _build_temporal_retrieval_prior_index(
        model,
        torch,
        fit_rows,
        fit_features,
        config=config,
        vocab=vocab,
        device=device,
    )
    retrieval_summary = {key: value for key, value in retrieval_index.items() if key not in {"embeddings", "tokens"}}
    candidate_token_prior_weights, candidate_token_prior_summary = _candidate_token_prior_weights(
        fit_rows,
        vocab=vocab,
        max_slots=max_slots,
        preserve_pad_slots=bool(config.get("preserve_pad_action_slots", config.get("pad_action_slots_as_pad", False))),
        config=config,
    )

    probability_rows: list[dict[str, Any]] | None = None
    needs_probability_rows = (
        bool(config.get("calibrate_non_noop_budget", config.get("non_noop_budgeted_unmasking", False)))
        or bool(config.get("calibrate_family_non_noop_budget", config.get("family_non_noop_budgeted_unmasking", False)))
    )
    if needs_probability_rows and calibration_rows:
        probability_rows = _collect_temporal_probability_rows(
            model,
            torch,
            calibration_rows,
            calibration_features,
            config=config,
            vocab=vocab,
            device=device,
            retrieval_index=retrieval_index,
            token_prior_weights=candidate_token_prior_weights,
        )
    non_noop_budget: dict[str, Any] = {"status": "skipped", "reason": "disabled"}
    family_non_noop_budget: dict[str, Any] = {"status": "skipped", "reason": "disabled"}
    if bool(config.get("calibrate_non_noop_budget", config.get("non_noop_budgeted_unmasking", False))) and probability_rows:
        non_noop_budget = _calibrate_temporal_non_noop_budget(probability_rows, config=config)
        if non_noop_budget.get("status") == "pass":
            config["non_noop_budgeted_unmasking"] = True
            config["non_noop_budget_score_threshold"] = float(non_noop_budget["selected_threshold"])
            config["non_noop_budget_max_tokens_per_row"] = int(non_noop_budget["max_tokens_per_row"])
    if bool(config.get("calibrate_family_non_noop_budget", config.get("family_non_noop_budgeted_unmasking", False))) and probability_rows:
        family_non_noop_budget = _calibrate_temporal_family_non_noop_budget(probability_rows, config=config)
        if family_non_noop_budget.get("status") == "pass":
            config["family_non_noop_budgeted_unmasking"] = True
            config["family_non_noop_budget"] = family_non_noop_budget

    candidate_family_diagnostics: dict[str, Any] = {
        "calibration": _candidate_family_diagnostics(probability_rows or [], config=config),
        "target_prefix": {"status": "skipped", "reason": "disabled"},
    }
    target_diagnostic_rows = max(0, int(config.get("candidate_diagnostics_target_max_rows", 0) or 0))
    adapt_family_budget = bool(config.get("adaptive_family_budget_to_unlabeled_target", False))
    target_probability_rows: list[dict[str, Any]] | None = None
    if target_diagnostic_rows > 0 or adapt_family_budget:
        adaptive_limit = int(config.get("adaptive_family_budget_max_rows", len(target_rows)) or len(target_rows))
        limit = min(
            len(target_rows),
            max(target_diagnostic_rows, adaptive_limit if adapt_family_budget else 0),
        )
        target_probability_rows = _collect_temporal_probability_rows(
            model,
            torch,
            target_rows[:limit],
            target_features[:limit],
            config=config,
            vocab=vocab,
            device=device,
            retrieval_index=retrieval_index,
            token_prior_weights=candidate_token_prior_weights,
        )
        if target_diagnostic_rows > 0:
            candidate_family_diagnostics["target_prefix"] = _candidate_family_diagnostics(
                target_probability_rows[: min(target_diagnostic_rows, len(target_probability_rows))],
                config=config,
            )
    if adapt_family_budget and target_probability_rows and family_non_noop_budget.get("status") == "pass":
        family_non_noop_budget = _adapt_temporal_family_budget_to_unlabeled_distribution(
            family_non_noop_budget,
            target_probability_rows,
            config=config,
        )
        config["family_non_noop_budget"] = family_non_noop_budget

    predictions_path = Path(output_dir) / "predictions.jsonl"
    prediction_batch_size = max(1, int(config.get("prediction_batch_size", config.get("batch_size", 64))))
    with predictions_path.open("w", encoding="utf-8") as handle:
        for start_idx in range(0, len(target_rows), prediction_batch_size):
            batch_rows = target_rows[start_idx : start_idx + prediction_batch_size]
            batch_predictions = _predict_temporal_tokens_batch(
                model,
                torch,
                batch_rows,
                target_features[start_idx : start_idx + prediction_batch_size],
                start_index=start_idx,
                all_features=target_features,
                config=config,
                vocab=vocab,
                device=device,
                retrieval_index=retrieval_index,
                token_prior_weights=candidate_token_prior_weights,
            )
            for row, predicted_tokens in zip(batch_rows, batch_predictions):
                handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": predicted_tokens}, sort_keys=True) + "\n")

    metrics_path = Path(output_dir) / "paper_metrics.json"
    write_paper_idm_metrics(
        prediction_paths=[predictions_path],
        target_paths=target_paths,
        output_path=metrics_path,
        model_name=str(config.get("model_name", "temporal_masked_diffusion_idm")) + "_predict",
        max_rows=len(target_rows),
    )
    summary = {
        "schema": "temporal_masked_diffusion_idm_prediction_summary.v1",
        "status": "pass",
        "checkpoint_path": str(checkpoint_path),
        "model_name": str(config.get("model_name", "temporal_masked_diffusion_idm")),
        "train_rows": len(train_rows),
        "fit_rows": len(fit_rows),
        "calibration_rows": len(calibration_rows),
        "target_rows": len(target_rows),
        "vocab_size": len(vocab),
        "max_slots": max_slots,
        "temporal_offsets": offsets,
        "video_feature_source": feature_source,
        "raw_video_frame_offsets": _raw_video_frame_offsets(config)
        if feature_source in {"raw_frames", "raw_video_frames", "frame_provider", "video_idm_cache", "raw_video_cache"}
        else None,
        "non_noop_budget": non_noop_budget,
        "family_non_noop_budget": family_non_noop_budget,
        "candidate_family_diagnostics": candidate_family_diagnostics,
        "retrieval_action_prior": retrieval_summary,
        "candidate_token_prior": candidate_token_prior_summary,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "resolved_config_path": str(Path(output_dir) / "resolved_config.json"),
        "device": str(device),
        "wall_clock_seconds": time.time() - start,
        "claim_boundary": "Prediction-only sweep from a temporal masked-diffusion IDM checkpoint; no target-label calibration and no G005 completion claim.",
    }
    write_json(Path(output_dir) / "resolved_config.json", config)
    if summary_out is None:
        summary_out = Path(output_dir) / "prediction_summary.json"
    write_json(summary_out, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prediction/calibration sweeps from a temporal masked-diffusion IDM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional JSON/YAML config whose keys override the checkpoint config before --set overrides.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config key with JSON-compatible key=value.")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    summary = predict_temporal_masked_diffusion_idm(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        summary_out=Path(args.summary_out) if args.summary_out else None,
        config_path=Path(args.config) if args.config else None,
        overrides=args.overrides,
        force_cpu=args.force_cpu,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "target_rows": summary["target_rows"],
                "metrics_path": summary["metrics_path"],
                "summary_path": args.summary_out or str(Path(args.output_dir) / "prediction_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
