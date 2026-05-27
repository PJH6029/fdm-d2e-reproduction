#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.eval.paper_idm_metrics import write_paper_idm_metrics
from fdm_d2e.io_utils import write_json
from fdm_d2e.training.streaming_idm import predict_streaming_idm_checkpoint


def _parse_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("threshold grid must not be empty")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("thresholds must be in [0, 1]")
    return values


def _threshold_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").rstrip("0").rstrip("p")


def _is_keyboard_token(token: str) -> bool:
    return token.startswith("KEY_")


def _is_mouse_button_token(token: str) -> bool:
    return token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))


def _thresholds_for_vocab(
    vocab: Iterable[str],
    *,
    keyboard_threshold: float,
    button_threshold: float,
    default_threshold: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in vocab:
        token = str(raw)
        if _is_keyboard_token(token):
            thresholds[token] = float(keyboard_threshold)
        elif _is_mouse_button_token(token):
            thresholds[token] = float(button_threshold)
        else:
            thresholds[token] = float(default_threshold)
    return thresholds


def _load_category_vocab(checkpoint_path: Path) -> list[str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without train deps.
        raise RuntimeError("torch is required to load the streaming IDM checkpoint vocabulary") from exc
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stats = checkpoint.get("stats", {}) if isinstance(checkpoint, dict) else {}
    vocab = checkpoint.get("category_vocab") if isinstance(checkpoint, dict) else None
    if not vocab:
        vocab = stats.get("category_vocab")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError(f"checkpoint has no category_vocab: {checkpoint_path}")
    return [str(token) for token in vocab]


def _metric_value(metrics: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if value is not None else None


def _combo_score(metrics: dict[str, Any]) -> float:
    all_group = metrics.get("groups", {}).get("all", {})
    paper = all_group.get("paper_compatible", {})
    strict = all_group.get("strict_local", {})
    fpr = _metric_value(strict, ("mouse_button", "no_button_false_positive_rate"))
    if fpr is not None and fpr > 0.10:
        return -1.0 + (0.10 - fpr)
    values = [
        _metric_value(paper, ("keyboard", "key_accuracy")),
        _metric_value(paper, ("mouse_button", "button_accuracy")),
        _metric_value(paper, ("mouse_move", "pearson_x")),
        _metric_value(paper, ("mouse_move", "pearson_y")),
    ]
    return sum(value for value in values if value is not None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prefix sweep of group-specific keyboard/button category thresholds "
            "for an existing streaming IDM checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records-glob", required=True)
    parser.add_argument("--target-glob", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--endpoints", default="configs/eval/primary_endpoints.yaml")
    parser.add_argument("--model-prefix", default="streaming_idm_threshold_sweep")
    parser.add_argument("--keyboard-thresholds", type=_parse_grid, default="0.05,0.1,0.15,0.2,0.35")
    parser.add_argument("--button-thresholds", type=_parse_grid, default="0.05,0.1,0.15,0.2,0.35")
    parser.add_argument("--default-threshold", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=320000)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--prediction-workers", type=int, default=1)
    parser.add_argument("--prediction-cuda-devices", default="")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--empty-bins-as-correct", action="store_true")
    args = parser.parse_args()

    started = time.time()
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    metrics_root = Path(args.metrics_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    vocab = _load_category_vocab(checkpoint_path)

    combos = [
        (key_threshold, button_threshold)
        for key_threshold in args.keyboard_thresholds
        for button_threshold in args.button_thresholds
    ]
    if args.max_combos and args.max_combos > 0:
        combos = combos[: int(args.max_combos)]

    results: list[dict[str, Any]] = []
    cuda_devices = [device.strip() for device in str(args.prediction_cuda_devices).split(",") if device.strip()]
    for index, (key_threshold, button_threshold) in enumerate(combos):
        slug = f"key{_threshold_slug(key_threshold)}_button{_threshold_slug(button_threshold)}"
        model_name = f"{args.model_prefix}_{slug}"
        combo_output = output_root / slug
        thresholds = _thresholds_for_vocab(
            vocab,
            keyboard_threshold=float(key_threshold),
            button_threshold=float(button_threshold),
            default_threshold=float(args.default_threshold),
        )
        predict_config: dict[str, Any] = {
            "schema": "streaming_idm_predict_config.v1",
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path),
            "endpoints": str(args.endpoints),
            "records_path": str(args.records_glob),
            "records_glob": str(args.records_glob),
            "output_dir": str(combo_output),
            "max_target_examples": int(args.max_rows),
            "prediction_workers": int(args.prediction_workers),
            "eval_batch_size": int(args.eval_batch_size),
            "resume_predictions": False,
            "category_threshold": float(args.default_threshold),
            "category_thresholds": thresholds,
            "force_cpu": bool(args.force_cpu),
            "claim_boundary": "Threshold-sweep prefix diagnostic only; not G005 completion evidence.",
        }
        if cuda_devices:
            predict_config["prediction_cuda_devices"] = cuda_devices
        prediction_summary = predict_streaming_idm_checkpoint(predict_config)
        metrics_path = metrics_root / f"{slug}_paper_metrics.json"
        progress_path = metrics_root / f"{slug}_paper_metrics_progress.json"
        paper_metrics = write_paper_idm_metrics(
            prediction_paths=[prediction_summary["predictions_path"]],
            target_paths=[str(args.target_glob)],
            output_path=metrics_path,
            progress_output_path=progress_path,
            progress_rows=100000,
            model_name=model_name,
            max_rows=int(args.max_rows),
            empty_bins_as_correct=bool(args.empty_bins_as_correct),
        )
        all_group = paper_metrics.get("groups", {}).get("all", {})
        paper = all_group.get("paper_compatible", {})
        strict = all_group.get("strict_local", {})
        result = {
            "index": index,
            "slug": slug,
            "model_name": model_name,
            "keyboard_threshold": float(key_threshold),
            "button_threshold": float(button_threshold),
            "default_threshold": float(args.default_threshold),
            "output_dir": str(combo_output),
            "prediction_summary_path": str(combo_output / "prediction_summary.json"),
            "predictions_path": prediction_summary["predictions_path"],
            "metrics_path": str(metrics_path),
            "rows": paper_metrics.get("alignment", {}).get("rows_seen"),
            "status": paper_metrics.get("status"),
            "paper_keyboard_accuracy": _metric_value(paper, ("keyboard", "key_accuracy")),
            "paper_mouse_button_accuracy": _metric_value(paper, ("mouse_button", "button_accuracy")),
            "paper_mouse_pearson_x": _metric_value(paper, ("mouse_move", "pearson_x")),
            "paper_mouse_pearson_y": _metric_value(paper, ("mouse_move", "pearson_y")),
            "strict_mouse_button_f1": _metric_value(strict, ("mouse_button", "f1")),
            "strict_no_button_fpr": _metric_value(strict, ("mouse_button", "no_button_false_positive_rate")),
            "score": _combo_score(paper_metrics),
        }
        write_json(combo_output / "prediction_summary.json", prediction_summary)
        results.append(result)
        write_json(
            args.summary,
            {
                "schema": "streaming_idm_group_threshold_sweep.v1",
                "status": "running",
                "completed": len(results),
                "total": len(combos),
                "results": results,
            },
        )

    best = max(results, key=lambda row: float(row.get("score") or -999.0)) if results else None
    summary = {
        "schema": "streaming_idm_group_threshold_sweep.v1",
        "status": "pass" if results else "fail",
        "checkpoint": str(checkpoint_path),
        "records_glob": str(args.records_glob),
        "target_glob": str(args.target_glob),
        "max_rows": int(args.max_rows),
        "empty_bins_as_correct": bool(args.empty_bins_as_correct),
        "keyboard_thresholds": [float(value) for value in args.keyboard_thresholds],
        "button_thresholds": [float(value) for value in args.button_thresholds],
        "default_threshold": float(args.default_threshold),
        "combo_count": len(results),
        "best": best,
        "results": results,
        "wall_clock_seconds": time.time() - started,
        "claim_boundary": "Threshold-sweep prefix diagnostic only; not G005 completion evidence.",
    }
    write_json(args.summary, summary)
    print(json.dumps({"status": summary["status"], "combo_count": len(results), "best": best}, sort_keys=True))
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
