#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.data.frame_embedding_materializer import (  # noqa: E402
    FrameEmbeddingMaterializerConfig,
    materialize_frame_embedding_features,
    parse_offsets,
    parse_path_remaps,
)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize D2E JSONL rows with __streaming_idm_features built from frozen frame embeddings. "
            "Use --backend dummy-stat for dependency-light tests and --backend hf-vision for the G005 "
            "prefix-gated pretrained-vision branch."
        )
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--backend", default="dummy-stat", choices=["dummy-stat", "hf-vision", "dinov2-torchhub"])
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--frame-offsets", default="0,2", help="Comma-separated 50ms row/frame offsets, e.g. 0,1,2.")
    parser.add_argument(
        "--frame-source",
        default="video",
        choices=["video", "compact-luma"],
        help="Use video/PPM frame.path decoding or compact luma16 fields already present in D2E JSONL rows.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-fps", type=int, default=20)
    parser.add_argument("--missing-frame-policy", default="zero", choices=["zero", "error"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-pooling", default="cls", choices=["cls", "mean", "pooler", "image"])
    parser.add_argument(
        "--hf-preprocess",
        default="manual-imagenet",
        choices=["manual-imagenet", "auto"],
        help="manual-imagenet avoids torchvision; auto uses AutoImageProcessor when its optional deps are installed.",
    )
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--no-embedding-deltas", action="store_true")
    parser.add_argument("--no-summary-features", action="store_true")
    parser.add_argument(
        "--summary-feature-mode",
        default="summary_compact_luma16_pair_shift_time_state_duration_prior_action",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--round-digits", default="6", help="Set to -1 to disable rounding.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional frame.path prefix remap, repeated as FROM=TO (e.g. /root/work=/mnt/ddn/prod-runs/user).",
    )
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--progress-rows", type=int, default=50_000)
    parser.add_argument("--source-label", default="g005_frozen_frame_embedding_materialization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrameEmbeddingMaterializerConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_out=args.summary_out,
        backend=args.backend,
        model_id=args.model_id,
        frame_offsets=parse_offsets(args.frame_offsets),
        frame_source=args.frame_source,
        image_size=args.image_size,
        frame_fps=args.frame_fps,
        missing_frame_policy=args.missing_frame_policy,
        batch_size=args.batch_size,
        device=args.device,
        embedding_pooling=args.embedding_pooling,
        hf_preprocess=args.hf_preprocess,
        normalize_embeddings=not args.no_normalize_embeddings,
        include_embedding_deltas=not args.no_embedding_deltas,
        include_summary_features=not args.no_summary_features,
        summary_feature_mode=args.summary_feature_mode,
        max_rows=args.max_rows,
        round_digits=_optional_int(args.round_digits),
        trust_remote_code=bool(args.trust_remote_code),
        path_remaps=parse_path_remaps(args.path_map),
        progress_output=args.progress_output,
        progress_rows=args.progress_rows,
        source_label=args.source_label,
    )
    summary = materialize_frame_embedding_features(config)
    print(json.dumps({"status": summary["status"], "rows": summary["rows_written"], "feature_dim": summary["feature_dim"]}, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
