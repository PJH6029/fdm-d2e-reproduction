from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator, write_paper_idm_metrics
from fdm_d2e.io_utils import ensure_dir, write_json
from fdm_d2e.training.masked_diffusion_idm import (
    FDM1_ACTION_PAD,
    FDM1_ACTION_MASK,
    FDM1_ACTION_NOOP,
    canonical_action_slot_record,
    canonical_fdm1_action_tokens,
    corrupt_action_slots,
    d2e_metric_tokens_from_fdm1_tokens,
    iterative_unmask_counts,
    select_topk_masked,
)
from fdm_d2e.training.masked_diffusion_idm_trainer import _screen_size, video_feature_vector
from fdm_d2e.training.torch_idm import require_torch, torch_available


def _expand_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths: list[Path] = []
    for item in values:
        matches = sorted(glob.glob(str(item)))
        paths.extend(Path(match) for match in matches)
    return paths


def _iter_jsonl(paths: Sequence[Path], *, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    emitted = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and emitted >= max_rows:
                    return
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
                emitted += 1
                yield row


def _target_slots(row: dict[str, Any], *, max_slots: int, preserve_pad_slots: bool = False) -> list[str]:
    record = canonical_action_slot_record(row, max_slots=max_slots)
    if preserve_pad_slots:
        return list(record.padded_tokens)
    return [FDM1_ACTION_NOOP if token.startswith("<FDM1_ACTION_PAD") else token for token in record.padded_tokens]


def _action_mouse_token_mode(config: dict[str, Any]) -> str:
    return str(config.get("action_mouse_tokenization", config.get("mouse_token_mode", "fdm1_49")))


def _target_slots_for_config(
    row: dict[str, Any],
    *,
    max_slots: int,
    config: dict[str, Any],
    preserve_pad_slots: bool = False,
) -> list[str]:
    record = canonical_action_slot_record(row, max_slots=max_slots, mouse_token_mode=_action_mouse_token_mode(config))
    if preserve_pad_slots:
        return list(record.padded_tokens)
    return [FDM1_ACTION_NOOP if token.startswith("<FDM1_ACTION_PAD") else token for token in record.padded_tokens]


def _build_vocab(rows: Sequence[dict[str, Any]], *, max_slots: int, min_count: int = 1, preserve_pad_slots: bool = False) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for token in _target_slots(row, max_slots=max_slots, preserve_pad_slots=preserve_pad_slots):
            counts[token] = counts.get(token, 0) + 1
    counts.setdefault(FDM1_ACTION_NOOP, 1)
    vocab = ["<FDM1_ACTION_PAD>", FDM1_ACTION_MASK]
    vocab.extend(sorted(token for token, count in counts.items() if count >= min_count and token not in vocab))
    return vocab


def _build_vocab_for_config(
    rows: Sequence[dict[str, Any]],
    *,
    max_slots: int,
    config: dict[str, Any],
    min_count: int = 1,
    preserve_pad_slots: bool = False,
) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for token in _target_slots_for_config(row, max_slots=max_slots, config=config, preserve_pad_slots=preserve_pad_slots):
            counts[token] = counts.get(token, 0) + 1
    counts.setdefault(FDM1_ACTION_NOOP, 1)
    vocab = ["<FDM1_ACTION_PAD>", FDM1_ACTION_MASK]
    vocab.extend(sorted(token for token, count in counts.items() if count >= min_count and token not in vocab))
    return vocab


def _temporal_offsets(config: dict[str, Any]) -> list[int]:
    raw = config.get("temporal_offsets", [-2, -1, 0, 1, 2])
    offsets = [int(value) for value in raw] if isinstance(raw, list) and raw else [-2, -1, 0, 1, 2]
    if 0 not in offsets:
        offsets.append(0)
    return sorted(dict.fromkeys(offsets))


def _center_index(offsets: Sequence[int]) -> int:
    return list(offsets).index(0) if 0 in offsets else len(offsets) // 2


def _raw_video_frame_offsets(config: dict[str, Any]) -> list[int]:
    raw = config.get("raw_video_frame_offsets", config.get("video_frame_offsets"))
    if raw is None:
        raw = [0, int(config.get("next_frame_offset", 1))]
    offsets = [int(value) for value in raw] if isinstance(raw, list) else [0, int(raw)]
    if not offsets:
        offsets = [0]
    return list(dict.fromkeys(offsets))


def _raw_video_feature_dim(config: dict[str, Any]) -> int:
    if config.get("video_feature_dim") is not None:
        return int(config["video_feature_dim"])
    image_size = int(config.get("raw_video_image_size", config.get("video_image_size", 96)))
    frame_dim = len(_raw_video_frame_offsets(config)) * image_size * image_size
    aux_dim = int(config.get("raw_video_aux_feature_dim", 0) or 0)
    return int(frame_dim + max(0, aux_dim))


def _configured_video_feature_dim(config: dict[str, Any]) -> int:
    source = str(config.get("video_feature_source", "json")).lower()
    if source in {"raw_frames", "raw_video_frames", "frame_provider", "video_idm_cache", "raw_video_cache"}:
        return _raw_video_feature_dim(config)
    return int(config.get("video_feature_dim", 64))


def _precompute_raw_video_features(rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> list[list[float]]:
    """Load downsampled raw screen-video frames for temporal IDM conditioning.

    This is the bridge from the compact-luma diagnostic probes back to the
    public FDM-1 recipe shape: the IDM remains a non-causal masked action-token
    diffusion model, but the conditioning tokens now come directly from D2E
    frame/video references instead of pre-materialized 16x16 summary features.
    Full-corpus promotion should replace this in-memory prefix path with the
    existing tensor-cache infrastructure, but the feature contract is identical:
    a deterministic sequence of normalized screen-video tokens plus optional
    train/eval row metadata features.
    """

    from fdm_d2e.training.video_idm import _FramePairProvider

    image_size = int(config.get("raw_video_image_size", config.get("video_image_size", 96)))
    offsets = _raw_video_frame_offsets(config)
    expected_frame_dim = len(offsets) * image_size * image_size
    aux_paths = list(config.get("raw_video_aux_feature_paths", []))
    aux_dim = int(config.get("raw_video_aux_feature_dim", 0) or 0)
    feature_dim = _raw_video_feature_dim(config)
    storage = str(config.get("raw_video_feature_storage", "list")).lower()
    tensor_storage = storage in {"tensor", "torch", "float16_tensor", "fp16_tensor"}
    torch = require_torch() if tensor_storage else None
    tensor_dtype = None
    if tensor_storage:
        dtype_name = str(config.get("raw_video_feature_tensor_dtype", "float16")).lower()
        tensor_dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
    provider = _FramePairProvider(
        root=Path(config.get("root", ".")).resolve(),
        image_size=image_size,
        fps=int(config.get("raw_video_frame_fps", config.get("video_frame_fps", 20))),
        next_frame_offset=int(config.get("next_frame_offset", 1)),
        missing_frame_policy=str(config.get("raw_video_missing_frame_policy", config.get("missing_frame_policy", "error"))),
    )
    features: list[list[float]] = []
    tensor_features: list[Any] = []
    try:
        for row in rows:
            frame_bytes = b"".join(provider.frames(row, offsets=offsets))
            if tensor_storage:
                raw = torch.frombuffer(bytearray(frame_bytes[:expected_frame_dim]), dtype=torch.uint8).to(dtype=torch.float32).div_(255.0)
                if int(raw.numel()) < expected_frame_dim:
                    raw = torch.cat([raw, torch.zeros(expected_frame_dim - int(raw.numel()), dtype=torch.float32)])
                parts = [raw[:expected_frame_dim]]
                if aux_dim > 0 and aux_paths:
                    parts.append(torch.tensor(video_feature_vector(row, feature_paths=aux_paths, dim=aux_dim), dtype=torch.float32))
                values_tensor = torch.cat(parts) if len(parts) > 1 else parts[0]
                if int(values_tensor.numel()) < feature_dim:
                    values_tensor = torch.cat([values_tensor, torch.zeros(feature_dim - int(values_tensor.numel()), dtype=torch.float32)])
                tensor_features.append(values_tensor[:feature_dim].to(dtype=tensor_dtype).contiguous())
            else:
                values = [float(value) / 255.0 for value in frame_bytes[:expected_frame_dim]]
                if len(values) < expected_frame_dim:
                    values.extend([0.0] * (expected_frame_dim - len(values)))
                if aux_dim > 0 and aux_paths:
                    values.extend(video_feature_vector(row, feature_paths=aux_paths, dim=aux_dim))
                if len(values) < feature_dim:
                    values.extend([0.0] * (feature_dim - len(values)))
                features.append(values[:feature_dim])
    finally:
        provider.close()
    if tensor_storage:
        return tensor_features
    return features


def _precompute_features(rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> list[list[float]]:
    source = str(config.get("video_feature_source", "json")).lower()
    if source in {"raw_frames", "raw_video_frames", "frame_provider"}:
        return _precompute_raw_video_features(rows, config=config)
    feature_paths = list(config.get("video_feature_paths", ["compact_luma_window", "compact_luma_window_mask", "frame.features", "next_frame_features", "frame_delta_features"]))
    feature_dim = _configured_video_feature_dim(config)
    return [video_feature_vector(row, feature_paths=feature_paths, dim=feature_dim) for row in rows]


def _maybe_tensorize_features(torch: Any, features: Any, *, config: dict[str, Any], split_name: str) -> Any:
    """Optionally store precomputed JSON features as one CPU tensor.

    Long-context FDM-1-style probes repeatedly gather dozens of neighboring
    screen-video rows per training example.  Keeping compact-luma features as
    Python lists forces every ``__getitem__`` call to re-box thousands of floats
    before a GPU step.  Tensorizing once preserves the same recipe/objective
    while avoiding sustained H200 idle caused by Python data marshaling.
    """

    del split_name  # reserved for future per-split diagnostics without API churn
    if not bool(config.get("precompute_features_as_tensor", config.get("tensorize_precomputed_features", False))):
        return features
    if hasattr(features, "detach") and hasattr(features, "to"):
        dtype_name = str(config.get("precompute_feature_tensor_dtype", "float16")).lower()
        dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
        return features.to(dtype=dtype).contiguous()
    if not features:
        return features
    dtype_name = str(config.get("precompute_feature_tensor_dtype", "float16")).lower()
    dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
    return torch.tensor(features, dtype=dtype).contiguous()


def _precompute_video_cache_features(
    record_paths: Sequence[Path],
    *,
    split_name: str,
    config: dict[str, Any],
    max_rows: int,
) -> list[Any]:
    """Load raw frame-token features from existing video-IDM tensor caches.

    This avoids reserving H200s for slow ffmpeg decode when a prior raw-video
    cache already exists on the PVC.  The action model is still the FDM-1-shaped
    temporal masked-diffusion IDM; the cache only supplies decoded screen-video
    tensors in the same normalized flattened format as ``raw_frames``.
    """

    from fdm_d2e.io_utils import read_json
    from fdm_d2e.training.video_idm import _load_cache_chunk, load_video_idm_cache_manifests

    if max_rows <= 0:
        return []
    torch = require_torch()
    stats_path = Path(config.get("video_cache_stats_path", config.get("stats_path", "")))
    if not stats_path.exists():
        raise FileNotFoundError(f"missing video cache stats_path for temporal IDM: {stats_path}")
    stats = read_json(stats_path)
    manifests = load_video_idm_cache_manifests(record_paths, stats=stats, config=config, split_name=split_name)
    feature_dim = _raw_video_feature_dim(config)
    dtype_name = str(config.get("raw_video_feature_tensor_dtype", "float16")).lower()
    tensor_dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
    features: list[Any] = []
    remaining = int(max_rows)
    for manifest in manifests:
        for chunk in manifest.get("chunks", []):
            if remaining <= 0:
                break
            payload = _load_cache_chunk(torch, chunk["path"])
            frames = payload["frames"]
            take = min(int(frames.shape[0]), remaining)
            flat = frames[:take].reshape(take, -1).to(dtype=torch.float32).div_(255.0)
            if int(flat.shape[1]) < feature_dim:
                flat = torch.cat([flat, torch.zeros((take, feature_dim - int(flat.shape[1])), dtype=torch.float32)], dim=1)
            flat = flat[:, :feature_dim].to(dtype=tensor_dtype).contiguous()
            features.extend(flat[idx].clone() for idx in range(take))
            remaining -= take
        if remaining <= 0:
            break
    if len(features) < max_rows:
        raise ValueError(f"video cache split {split_name} provided {len(features)} rows, expected {max_rows}")
    return features


def _precompute_target_ids(rows: Sequence[dict[str, Any]], *, max_slots: int, token_to_index: dict[str, int], preserve_pad_slots: bool = False) -> list[list[int]]:
    noop = token_to_index[FDM1_ACTION_NOOP]
    return [
        [token_to_index.get(token, noop) for token in _target_slots(row, max_slots=max_slots, preserve_pad_slots=preserve_pad_slots)]
        for row in rows
    ]


def _precompute_target_ids_for_config(
    rows: Sequence[dict[str, Any]],
    *,
    max_slots: int,
    token_to_index: dict[str, int],
    config: dict[str, Any],
    preserve_pad_slots: bool = False,
) -> list[list[int]]:
    noop = token_to_index[FDM1_ACTION_NOOP]
    return [
        [
            token_to_index.get(token, noop)
            for token in _target_slots_for_config(
                row,
                max_slots=max_slots,
                config=config,
                preserve_pad_slots=preserve_pad_slots,
            )
        ]
        for row in rows
    ]


class _TemporalMaskedDiffusionDataset:
    def __init__(
        self,
        *,
        features: Sequence[Sequence[float]],
        target_ids: Sequence[Sequence[int]],
        config: dict[str, Any],
        vocab: Sequence[str],
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.features = features if hasattr(features, "detach") and hasattr(features, "to") else list(features)
        self.features_are_tensor = hasattr(self.features, "detach") and hasattr(self.features, "to")
        self.target_ids = [list(row) for row in target_ids]
        self.vocab = list(vocab)
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
        self.mask_probability = float(config.get("mask_probability", 0.65))
        self.random_token_probability = float(config.get("random_token_probability", 0.10))
        self.full_action_mask_probability = max(
            0.0,
            min(
                1.0,
                float(
                    config.get(
                        "full_action_mask_probability",
                        config.get("all_action_mask_probability", config.get("all_mask_probability", 0.0)),
                    )
                    or 0.0
                ),
            ),
        )
        self.seed = int(config.get("seed", 7))
        self.offsets = _temporal_offsets(config)
        self.loss_offsets = set(int(value) for value in config.get("temporal_loss_offsets", self.offsets))

    def __len__(self) -> int:
        return len(self.features)

    def _row_index(self, idx: int, offset: int) -> int:
        return max(0, min(len(self.features) - 1, idx + offset))

    def _feature_tensor(self, row_index: int) -> Any:
        value = self.features[row_index]
        if hasattr(value, "detach") and hasattr(value, "to"):
            return value.to(dtype=self.torch.float32)
        return self.torch.tensor(list(value), dtype=self.torch.float32)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, Any]:
        corrupted_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        index_to_token = {idx_: token for token, idx_ in self.token_to_index.items()}
        mask_index = self.token_to_index[FDM1_ACTION_MASK]
        force_full_mask = random.Random(self.seed + idx * 104729).random() < self.full_action_mask_probability
        row_indices: list[int] = []
        for offset_position, offset in enumerate(self.offsets):
            row_index = self._row_index(idx, offset)
            row_indices.append(row_index)
            target = list(self.target_ids[row_index])
            target_tokens = [index_to_token.get(token_id, FDM1_ACTION_NOOP) for token_id in target]
            if force_full_mask:
                # Public FDM-1 inference starts from interleaved frame tokens
                # plus masked action-token positions.  Mixing full-mask rows
                # into training reduces the teacher-forcing mismatch from
                # ordinary BERT-style partial corruption while staying within
                # the same masked-diffusion action-token objective.
                corrupted_tokens = [
                    FDM1_ACTION_MASK if token != FDM1_ACTION_PAD else token
                    for token in target_tokens
                ]
                loss_mask = [token != FDM1_ACTION_PAD for token in target_tokens]
            else:
                corrupted_tokens, loss_mask = corrupt_action_slots(
                    target_tokens,
                    vocab=self.vocab,
                    mask_probability=self.mask_probability,
                    random_token_probability=self.random_token_probability,
                    rng=random.Random(self.seed + idx * 1009 + offset_position),
                )
            if offset not in self.loss_offsets:
                loss_mask = [False for _ in loss_mask]
            corrupted_rows.append([self.token_to_index.get(token, mask_index) for token in corrupted_tokens])
            target_rows.append(target)
            mask_rows.append(loss_mask)
        if self.features_are_tensor:
            index_tensor = self.torch.tensor(row_indices, dtype=self.torch.long, device=self.features.device)
            feature_tensor = self.features.index_select(0, index_tensor).to(dtype=self.torch.float32)
        else:
            feature_tensor = self.torch.stack([self._feature_tensor(row_index) for row_index in row_indices], dim=0)
        return (
            feature_tensor,
            self.torch.tensor(corrupted_rows, dtype=self.torch.long),
            self.torch.tensor(target_rows, dtype=self.torch.long),
            self.torch.tensor(mask_rows, dtype=self.torch.bool),
        )


def _button_class_vocab(vocab: Sequence[str]) -> list[str]:
    return [str(token) for token in vocab if _action_family(str(token)) == "mouse_button"]


def _key_class_vocab(vocab: Sequence[str]) -> list[str]:
    return [str(token) for token in vocab if _action_family(str(token)) == "keyboard"]


def _mouse_move_class_vocab(vocab: Sequence[str]) -> list[str]:
    return [str(token) for token in vocab if _action_family(str(token)) == "mouse_move"]


def _mouse_axis_class_vocab(vocab: Sequence[str], axis: str) -> list[str]:
    normalized = str(axis).lower()
    if normalized in {"x", "dx"}:
        prefixes = ("FDM1_MOUSE_DX_", "MOUSE_DX_")
    elif normalized in {"y", "dy"}:
        prefixes = ("FDM1_MOUSE_DY_", "MOUSE_DY_")
    else:
        raise ValueError(f"unsupported mouse axis: {axis}")
    return [str(token) for token in vocab if str(token).startswith(prefixes)]


def _mouse_axis_for_token(token: str) -> str | None:
    token = str(token)
    if token.startswith(("FDM1_MOUSE_DX_", "MOUSE_DX_")):
        return "x"
    if token.startswith(("FDM1_MOUSE_DY_", "MOUSE_DY_")):
        return "y"
    return None


def _build_temporal_model(
    torch: Any,
    *,
    video_dim: int,
    vocab_size: int,
    max_slots: int,
    offsets: Sequence[int],
    config: dict[str, Any],
    vocab: Sequence[str] | None = None,
) -> Any:
    nn = torch.nn
    hidden_dim = int(config.get("hidden_dim", 256))
    layers = int(config.get("transformer_layers", 4))
    heads = int(config.get("transformer_heads", 4))
    dropout = float(config.get("dropout", 0.1))
    video_encoder_arch = str(config.get("video_encoder_arch", "flat_mlp")).lower()
    luma_window_frames = int(config.get("luma_window_frames", 5))
    luma_window_size = int(config.get("luma_window_size", 16))
    luma_window_dim = max(0, luma_window_frames * luma_window_size * luma_window_size)
    raw_video_frames = len(_raw_video_frame_offsets(config))
    raw_video_size = int(config.get("raw_video_image_size", config.get("video_image_size", 96)))
    raw_video_dim = max(0, raw_video_frames * raw_video_size * raw_video_size)
    button_vocab = _button_class_vocab(vocab or [])
    key_vocab = _key_class_vocab(vocab or [])
    mouse_move_vocab = _mouse_move_class_vocab(vocab or [])
    mouse_dx_vocab = _mouse_axis_class_vocab(vocab or [], "x")
    mouse_dy_vocab = _mouse_axis_class_vocab(vocab or [], "y")

    class CompactLumaWindowEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(config.get("luma_encoder_channels", 32))
            pooled_hw = int(config.get("luma_encoder_pool_hw", 2))
            self.luma_dim = min(video_dim, luma_window_dim)
            self.aux_dim = max(0, video_dim - self.luma_dim)
            self.frames = max(1, luma_window_frames)
            self.size = max(1, luma_window_size)
            self.conv = nn.Sequential(
                nn.Conv3d(1, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
                nn.Conv3d(channels, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((1, pooled_hw, pooled_hw)),
                nn.Flatten(),
            )
            conv_dim = channels * 2 * pooled_hw * pooled_hw
            aux_hidden = int(config.get("luma_aux_hidden_dim", min(hidden_dim, 128)))
            self.aux_proj = nn.Sequential(nn.Linear(self.aux_dim, aux_hidden), nn.GELU()) if self.aux_dim else None
            merged_dim = conv_dim + (aux_hidden if self.aux_proj is not None else 0)
            self.out = nn.Sequential(nn.Linear(merged_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        def forward(self, video_features: Any) -> Any:
            batch = int(video_features.shape[0])
            expected = self.frames * self.size * self.size
            luma = video_features[:, : self.luma_dim]
            if self.luma_dim < expected:
                pad = torch.zeros((batch, expected - self.luma_dim), device=video_features.device, dtype=video_features.dtype)
                luma = torch.cat([luma, pad], dim=1)
            luma = luma[:, :expected].reshape(batch, 1, self.frames, self.size, self.size)
            parts = [self.conv(luma)]
            if self.aux_proj is not None:
                parts.append(self.aux_proj(video_features[:, self.luma_dim :]))
            return self.out(torch.cat(parts, dim=1))

    class RawVideoFrameEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(config.get("raw_video_encoder_channels", config.get("luma_encoder_channels", 32)))
            pooled_hw = int(config.get("raw_video_encoder_pool_hw", config.get("luma_encoder_pool_hw", 2)))
            self.raw_dim = min(video_dim, raw_video_dim)
            self.aux_dim = max(0, video_dim - self.raw_dim)
            self.frames = max(1, raw_video_frames)
            self.size = max(1, raw_video_size)
            self.conv = nn.Sequential(
                nn.Conv3d(1, channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
                nn.GELU(),
                nn.Conv3d(channels, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2)),
                nn.GELU(),
                nn.Conv3d(channels * 2, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((1, pooled_hw, pooled_hw)),
                nn.Flatten(),
            )
            conv_dim = channels * 2 * pooled_hw * pooled_hw
            aux_hidden = int(config.get("raw_video_aux_hidden_dim", config.get("luma_aux_hidden_dim", min(hidden_dim, 128))))
            self.aux_proj = nn.Sequential(nn.Linear(self.aux_dim, aux_hidden), nn.GELU()) if self.aux_dim else None
            merged_dim = conv_dim + (aux_hidden if self.aux_proj is not None else 0)
            self.out = nn.Sequential(nn.Linear(merged_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        def forward(self, video_features: Any) -> Any:
            batch = int(video_features.shape[0])
            expected = self.frames * self.size * self.size
            raw = video_features[:, : self.raw_dim]
            if self.raw_dim < expected:
                pad = torch.zeros((batch, expected - self.raw_dim), device=video_features.device, dtype=video_features.dtype)
                raw = torch.cat([raw, pad], dim=1)
            raw = raw[:, :expected].reshape(batch, 1, self.frames, self.size, self.size)
            parts = [self.conv(raw)]
            if self.aux_proj is not None:
                parts.append(self.aux_proj(video_features[:, self.raw_dim :]))
            return self.out(torch.cat(parts, dim=1))

    class RawVideoPatchTokenEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(config.get("raw_video_encoder_channels", config.get("luma_encoder_channels", 32)))
            pooled_hw = int(config.get("raw_video_encoder_token_hw", config.get("raw_video_encoder_pool_hw", 2)))
            pooled_frames = int(config.get("raw_video_encoder_token_frames", 1))
            self.raw_dim = min(video_dim, raw_video_dim)
            self.aux_dim = max(0, video_dim - self.raw_dim)
            self.frames = max(1, raw_video_frames)
            self.size = max(1, raw_video_size)
            self.token_frames = max(1, pooled_frames)
            self.token_hw = max(1, pooled_hw)
            self.tokens_per_offset = self.token_frames * self.token_hw * self.token_hw
            self.conv = nn.Sequential(
                nn.Conv3d(1, channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
                nn.GELU(),
                nn.Conv3d(channels, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 2, 2)),
                nn.GELU(),
                nn.Conv3d(channels * 2, channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool3d((self.token_frames, self.token_hw, self.token_hw))
            self.token_proj = nn.Linear(channels * 2, hidden_dim)
            aux_hidden = int(config.get("raw_video_aux_hidden_dim", config.get("luma_aux_hidden_dim", min(hidden_dim, 128))))
            self.aux_proj = nn.Sequential(nn.Linear(self.aux_dim, aux_hidden), nn.GELU(), nn.Linear(aux_hidden, hidden_dim)) if self.aux_dim else None
            self.out_norm = nn.LayerNorm(hidden_dim)

        def forward(self, video_features: Any) -> Any:
            batch = int(video_features.shape[0])
            expected = self.frames * self.size * self.size
            raw = video_features[:, : self.raw_dim]
            if self.raw_dim < expected:
                pad = torch.zeros((batch, expected - self.raw_dim), device=video_features.device, dtype=video_features.dtype)
                raw = torch.cat([raw, pad], dim=1)
            raw = raw[:, :expected].reshape(batch, 1, self.frames, self.size, self.size)
            encoded = self.pool(self.conv(raw))
            tokens = encoded.permute(0, 2, 3, 4, 1).reshape(batch, self.tokens_per_offset, -1)
            projected = self.token_proj(tokens)
            if self.aux_proj is not None:
                projected = projected + self.aux_proj(video_features[:, self.raw_dim :]).unsqueeze(1)
            return self.out_norm(torch.nn.functional.gelu(projected))

    class TemporalMaskedDiffusionIDM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offsets = list(offsets)
            if video_encoder_arch in {"raw_video_cnn", "raw_frame_cnn", "raw_video_frame_cnn"}:
                self.video_reconstruction_dim = raw_video_dim
            elif video_encoder_arch in {"raw_video_patch_cnn", "raw_video_token_cnn", "raw_video_patch_tokens"}:
                self.video_reconstruction_dim = raw_video_dim
            elif video_encoder_arch in {"compact_luma_window_cnn", "luma_window_cnn", "video_luma_cnn"}:
                self.video_reconstruction_dim = luma_window_dim
            else:
                self.video_reconstruction_dim = 0
            if video_encoder_arch in {"raw_video_cnn", "raw_frame_cnn", "raw_video_frame_cnn"}:
                self.video_proj = RawVideoFrameEncoder()
            elif video_encoder_arch in {"raw_video_patch_cnn", "raw_video_token_cnn", "raw_video_patch_tokens"}:
                self.video_proj = RawVideoPatchTokenEncoder()
            elif self.video_reconstruction_dim > 0:
                self.video_proj = CompactLumaWindowEncoder()
            else:
                self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
            self.video_tokens_per_offset = int(getattr(self.video_proj, "tokens_per_offset", 1))
            self.video_token_embed = (
                nn.Embedding(self.video_tokens_per_offset, hidden_dim)
                if self.video_tokens_per_offset > 1
                else None
            )
            self.video_reconstruction_head = (
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.video_reconstruction_dim))
                if self.video_reconstruction_dim > 0
                else None
            )
            self.action_embed = nn.Embedding(vocab_size, hidden_dim)
            self.slot_embed = nn.Embedding(max_slots, hidden_dim)
            self.offset_embed = nn.Embedding(len(offsets), hidden_dim)
            self.type_embed = nn.Embedding(2, hidden_dim)  # 0=video, 1=action
            self.event_auxiliary = bool(config.get("temporal_event_auxiliary", config.get("event_auxiliary", False)))
            self.button_class_auxiliary = bool(config.get("temporal_button_class_auxiliary", False)) and bool(button_vocab)
            self.button_class_count = len(button_vocab)
            self.key_class_auxiliary = bool(config.get("temporal_key_class_auxiliary", False)) and bool(key_vocab)
            self.key_class_count = len(key_vocab)
            self.key_token_presence_auxiliary = bool(config.get("temporal_key_token_presence_auxiliary", False)) and bool(key_vocab)
            self.button_token_presence_auxiliary = bool(config.get("temporal_button_token_presence_auxiliary", False)) and bool(button_vocab)
            self.mouse_move_token_presence_auxiliary = bool(config.get("temporal_mouse_move_token_presence_auxiliary", False)) and bool(mouse_move_vocab)
            self.mouse_axis_class_auxiliary = (
                bool(config.get("temporal_mouse_axis_class_auxiliary", False))
                and bool(mouse_dx_vocab)
                and bool(mouse_dy_vocab)
            )
            self.mouse_dx_class_count = len(mouse_dx_vocab)
            self.mouse_dy_class_count = len(mouse_dy_vocab)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(hidden_dim, vocab_size)
            self.key_event_head = nn.Linear(hidden_dim, 1) if self.event_auxiliary else None
            self.button_event_head = nn.Linear(hidden_dim, 1) if self.event_auxiliary else None
            self.button_class_head = nn.Linear(hidden_dim, self.button_class_count + 1) if self.button_class_auxiliary else None
            self.key_class_head = nn.Linear(hidden_dim, self.key_class_count + 1) if self.key_class_auxiliary else None
            self.key_token_presence_head = nn.Linear(hidden_dim, len(key_vocab)) if self.key_token_presence_auxiliary else None
            self.button_token_presence_head = (
                nn.Linear(hidden_dim, len(button_vocab)) if self.button_token_presence_auxiliary else None
            )
            self.mouse_move_token_presence_head = (
                nn.Linear(hidden_dim, len(mouse_move_vocab)) if self.mouse_move_token_presence_auxiliary else None
            )
            self.mouse_dx_class_head = (
                nn.Linear(hidden_dim, self.mouse_dx_class_count + 1) if self.mouse_axis_class_auxiliary else None
            )
            self.mouse_dy_class_head = (
                nn.Linear(hidden_dim, self.mouse_dy_class_count + 1) if self.mouse_axis_class_auxiliary else None
            )
            self.token_presence_auxiliary = bool(config.get("temporal_token_presence_auxiliary", config.get("token_presence_auxiliary", False)))
            self.token_presence_head = nn.Linear(hidden_dim, vocab_size) if self.token_presence_auxiliary else None

        def video_embedding(self, video_features: Any) -> Any:
            flat = video_features.reshape(-1, video_features.shape[-1])
            encoded = self.video_proj(flat)
            if encoded.dim() == 3:
                return encoded.reshape(video_features.shape[0], video_features.shape[1], encoded.shape[1], encoded.shape[2])
            return encoded.reshape(video_features.shape[0], video_features.shape[1], -1)

        def video_summary_embedding(self, video_features: Any) -> Any:
            encoded = self.video_embedding(video_features)
            if encoded.dim() == 4:
                return encoded.mean(dim=2)
            return encoded

        def reconstruct_video(self, video_features: Any) -> Any:
            if self.video_reconstruction_head is None:
                raise RuntimeError("video reconstruction head unavailable")
            flat = video_features.reshape(-1, video_features.shape[-1])
            encoded = self.video_proj(flat)
            if encoded.dim() == 3:
                encoded = encoded.mean(dim=1)
            pred = self.video_reconstruction_head(encoded)
            return pred.reshape(video_features.shape[0], video_features.shape[1], -1)

        def _forward_impl(self, video_features: Any, corrupted_ids: Any) -> dict[str, Any]:
            batch, window, _ = video_features.shape
            slots = corrupted_ids.shape[-1]
            device = video_features.device
            offset_positions = torch.arange(window, device=device)
            video_embeddings = self.video_embedding(video_features)
            if video_embeddings.dim() == 4:
                video_token_count = int(video_embeddings.shape[2])
                video_token_positions = torch.arange(video_token_count, device=device)
                video_position_embeddings = (
                    self.video_token_embed(video_token_positions).view(1, 1, video_token_count, -1)
                    if self.video_token_embed is not None
                    else torch.zeros((1, 1, video_token_count, video_embeddings.shape[-1]), device=device)
                )
                video_tokens = (
                    video_embeddings
                    + self.offset_embed(offset_positions).view(1, window, 1, -1)
                    + video_position_embeddings
                    + self.type_embed(torch.zeros((), dtype=torch.long, device=device)).view(1, 1, 1, -1)
                ).reshape(batch, window * video_token_count, -1)
            else:
                video_tokens = video_embeddings + self.offset_embed(offset_positions).unsqueeze(0) + self.type_embed(torch.zeros(window, dtype=torch.long, device=device)).unsqueeze(0)
            action = self.action_embed(corrupted_ids)
            slot_positions = torch.arange(slots, device=device)
            action = action + self.offset_embed(offset_positions).view(1, window, 1, -1) + self.slot_embed(slot_positions).view(1, 1, slots, -1) + self.type_embed(torch.ones((), dtype=torch.long, device=device)).view(1, 1, 1, -1)
            sequence = torch.cat([video_tokens, action.reshape(batch, window * slots, -1)], dim=1)
            encoded = self.encoder(sequence)
            action_start = int(video_tokens.shape[1])
            action_encoded = encoded[:, action_start:, :].reshape(batch, window, slots, -1)
            payload: dict[str, Any] = {"action_logits": self.head(action_encoded)}
            if self.event_auxiliary and self.key_event_head is not None and self.button_event_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["key_event_logits"] = self.key_event_head(pooled).squeeze(-1)
                payload["button_event_logits"] = self.button_event_head(pooled).squeeze(-1)
            if self.button_class_auxiliary and self.button_class_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["button_class_logits"] = self.button_class_head(pooled)
            if self.key_class_auxiliary and self.key_class_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["key_class_logits"] = self.key_class_head(pooled)
            if self.key_token_presence_auxiliary and self.key_token_presence_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["key_token_presence_logits"] = self.key_token_presence_head(pooled)
            if self.button_token_presence_auxiliary and self.button_token_presence_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["button_token_presence_logits"] = self.button_token_presence_head(pooled)
            if self.mouse_move_token_presence_auxiliary and self.mouse_move_token_presence_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["mouse_move_token_presence_logits"] = self.mouse_move_token_presence_head(pooled)
            if self.mouse_axis_class_auxiliary and self.mouse_dx_class_head is not None and self.mouse_dy_class_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["mouse_dx_class_logits"] = self.mouse_dx_class_head(pooled)
                payload["mouse_dy_class_logits"] = self.mouse_dy_class_head(pooled)
            if self.token_presence_auxiliary and self.token_presence_head is not None:
                pooled = action_encoded.mean(dim=2)
                payload["token_presence_logits"] = self.token_presence_head(pooled)
            return payload

        def forward(self, video_features: Any, corrupted_ids: Any) -> Any:
            return self._forward_impl(video_features, corrupted_ids)["action_logits"]

        def forward_with_aux(self, video_features: Any, corrupted_ids: Any) -> dict[str, Any]:
            return self._forward_impl(video_features, corrupted_ids)

    return TemporalMaskedDiffusionIDM()


def _masked_video_reconstruction_loss(model: Any, torch: Any, features: Any, *, config: dict[str, Any]) -> Any:
    if not hasattr(model, "reconstruct_video"):
        return torch.tensor(0.0, device=features.device)
    recon_dim = int(getattr(model, "video_reconstruction_dim", 0) or 0)
    if recon_dim <= 0:
        return torch.tensor(0.0, device=features.device)
    target = features[:, :, : min(recon_dim, features.shape[-1])]
    if target.shape[-1] < recon_dim:
        pad = torch.zeros((*target.shape[:2], recon_dim - target.shape[-1]), device=features.device, dtype=features.dtype)
        target = torch.cat([target, pad], dim=-1)
    corrupted = features.clone()
    mask_probability = float(config.get("video_encoder_mask_probability", config.get("video_encoder_pretrain_mask_probability", 0.65)))
    mask = torch.rand(target.shape, device=features.device) < mask_probability
    if not bool(mask.any()):
        mask = torch.ones_like(target, dtype=torch.bool)
    source = corrupted[:, :, : min(recon_dim, corrupted.shape[-1])]
    source_mask = mask[:, :, : source.shape[-1]]
    source[source_mask] = 0.0
    corrupted[:, :, : source.shape[-1]] = source
    pred = model.reconstruct_video(corrupted)
    if bool(config.get("video_encoder_reconstruct_masked_only", True)):
        return torch.nn.functional.mse_loss(pred[mask], target[mask])
    return torch.nn.functional.mse_loss(pred, target)


def _rank_progress_path(config: dict[str, Any], *, output_dir: Path) -> Path:
    progress_dir = Path(config.get("rank_progress_dir", output_dir / "rank_progress"))
    return progress_dir / "train_rank0.json"


def _write_rank_progress(config: dict[str, Any], *, output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(
        _rank_progress_path(config, output_dir=output_dir),
        {
            "schema": "temporal_masked_diffusion_rank_progress.v1",
            "rank": 0,
            "updated_at_epoch": time.time(),
            **payload,
        },
    )


def _pretrain_video_encoder(model: Any, torch: Any, loader: Any, *, config: dict[str, Any], device: Any) -> list[dict[str, Any]]:
    epochs = int(config.get("video_encoder_pretrain_epochs", 0) or 0)
    if epochs <= 0 or not hasattr(model, "reconstruct_video") or int(getattr(model, "video_reconstruction_dim", 0) or 0) <= 0:
        return []
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("video_encoder_pretrain_lr", config.get("lr", 2e-4))),
        weight_decay=float(config.get("video_encoder_pretrain_weight_decay", config.get("weight_decay", 0.01))),
    )
    history: list[dict[str, Any]] = []
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_temporal_masked_diffusion_d2e"))
    progress_every = max(1, int(config.get("rank_progress_every_batches", 50)))
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    max_batches = int(config.get("video_encoder_pretrain_max_batches", 0) or 0)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for batch_index, (features, _corrupted, _targets, _mask) in enumerate(loader, 1):
            features = features.to(device=device, dtype=torch.float32)
            loss = _masked_video_reconstruction_loss(model, torch, features, config=config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            batch = int(features.shape[0])
            total += float(loss.detach().cpu()) * batch
            examples += batch
            if batch_index == 1 or batch_index % progress_every == 0 or (total_batches and batch_index == total_batches):
                _write_rank_progress(
                    config,
                    output_dir=output_dir,
                    payload={
                        "phase": "video_pretrain",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "batch": batch_index,
                        "batches": total_batches,
                        "examples": examples,
                        "loss": total / max(1, examples),
                    },
                )
            if max_batches > 0 and batch_index >= max_batches:
                break
        history.append(
            {
                "epoch": epoch + 1,
                "video_reconstruction_loss": total / max(1, examples),
                "examples": examples,
                "batches": batch_index if "batch_index" in locals() else 0,
                "max_batches": max_batches or None,
                "truncated": bool(max_batches > 0 and (total_batches is None or max_batches < total_batches)),
            }
        )
    return history


def _class_weights(torch: Any, vocab: Sequence[str], config: dict[str, Any], *, device: Any) -> Any:
    weights = torch.ones(len(vocab), dtype=torch.float32, device=device)
    for idx, token in enumerate(vocab):
        if token == FDM1_ACTION_NOOP:
            weights[idx] = float(config.get("noop_loss_weight", 1.0))
        elif token == "<FDM1_ACTION_PAD>":
            weights[idx] = float(config.get("pad_loss_weight", 0.0))
        elif token.startswith("KEY_"):
            weights[idx] = float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            weights[idx] = float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith(("FDM1_MOUSE_DX_", "FDM1_MOUSE_DY_", "MOUSE_DX_", "MOUSE_DY_")):
            weights[idx] = float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0)))
        elif token.startswith("SCROLL_"):
            weights[idx] = float(config.get("scroll_loss_weight", config.get("action_loss_weight", 1.0)))
    return weights


def _temporal_loss_offset_mask(torch: Any, offsets: Sequence[int], config: dict[str, Any], *, device: Any) -> Any:
    loss_offsets = config.get("temporal_loss_offsets", offsets)
    if loss_offsets is None:
        loss_offsets = offsets
    selected = {int(value) for value in loss_offsets}
    return torch.tensor([int(offset) in selected for offset in offsets], dtype=torch.bool, device=device)


def _temporal_event_targets(torch: Any, target_ids: Any, vocab: Sequence[str], prefixes: tuple[str, ...]) -> Any:
    indices = [idx for idx, token in enumerate(vocab) if str(token).startswith(prefixes)]
    if not indices:
        return torch.zeros(target_ids.shape[:2], dtype=torch.float32, device=target_ids.device)
    index_tensor = torch.tensor(indices, dtype=target_ids.dtype, device=target_ids.device)
    return (target_ids.unsqueeze(-1) == index_tensor.view(1, 1, 1, -1)).any(dim=(-1, -2)).float()


def _temporal_button_class_targets(torch: Any, target_ids: Any, vocab: Sequence[str], button_vocab: Sequence[str]) -> Any:
    """Return 0=no-button, 1..N=button action-token class per temporal row."""

    return _temporal_family_class_targets(torch, target_ids, vocab, button_vocab)


def _temporal_family_class_targets(torch: Any, target_ids: Any, vocab: Sequence[str], family_vocab: Sequence[str]) -> Any:
    """Return 0=no-family-token, 1..N=family action-token class per temporal row."""

    if not family_vocab:
        return torch.zeros(target_ids.shape[:2], dtype=torch.long, device=target_ids.device)
    class_by_token = {str(token): idx + 1 for idx, token in enumerate(family_vocab)}
    mapping = torch.zeros(len(vocab), dtype=torch.long, device=target_ids.device)
    for token_idx, token in enumerate(vocab):
        class_idx = class_by_token.get(str(token))
        if class_idx is not None:
            mapping[token_idx] = int(class_idx)
    mapped = mapping[target_ids]
    # D2E bins normally have at most one token per auxiliary class family; max
    # keeps the target deterministic if a rare multi-token bin appears.
    return mapped.max(dim=2).values


def _temporal_family_token_presence_targets(torch: Any, target_ids: Any, vocab: Sequence[str], family_vocab: Sequence[str]) -> Any:
    """Return multi-hot token-presence targets over a family-specific vocab.

    This keeps sparse key/button identity learning inside the public FDM-1
    masked action-token denoising recipe while avoiding a full-vocabulary
    auxiliary dominated by mouse/no-op negatives.
    """

    batch, window, slots = target_ids.shape
    if not family_vocab:
        return torch.zeros((batch, window, 0), dtype=torch.float32, device=target_ids.device)
    mapping = torch.zeros((len(vocab),), dtype=torch.long, device=target_ids.device)
    index_by_token = {str(token): idx for idx, token in enumerate(vocab)}
    for class_idx, token in enumerate(family_vocab, start=1):
        token_idx = index_by_token.get(str(token))
        if token_idx is None:
            continue
        mapping[token_idx] = int(class_idx)
    mapped = mapping[target_ids]
    # Channel 0 is the sentinel for non-family tokens; max over slots gives a
    # multi-hot family-token set without looping over every class per batch.
    return torch.nn.functional.one_hot(mapped, num_classes=len(family_vocab) + 1).amax(dim=2)[:, :, 1:].float()


def _event_auxiliary_bce_loss(torch: Any, logits: Any, targets: Any, offset_mask: Any, *, pos_weight: float) -> Any:
    if logits is None:
        return torch.tensor(0.0, device=targets.device)
    selected_logits = logits[:, offset_mask]
    selected_targets = targets[:, offset_mask]
    if selected_targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    weight = torch.tensor(float(pos_weight), dtype=selected_logits.dtype, device=selected_logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=weight)


def _button_class_auxiliary_loss(torch: Any, logits: Any, targets: Any, offset_mask: Any, config: dict[str, Any]) -> Any:
    return _family_class_auxiliary_loss(
        torch,
        logits,
        targets,
        offset_mask,
        no_family_weight=float(config.get("button_class_no_button_weight", 0.05)),
        family_weight=float(config.get("button_class_button_weight", config.get("button_event_pos_weight", 16.0))),
        focal_gamma=float(config.get("button_class_focal_gamma", 0.0) or 0.0),
    )


def _family_class_auxiliary_loss(
    torch: Any,
    logits: Any,
    targets: Any,
    offset_mask: Any,
    *,
    no_family_weight: float,
    family_weight: float,
    focal_gamma: float = 0.0,
) -> Any:
    if logits is None or logits.shape[-1] <= 1:
        return torch.tensor(0.0, device=targets.device)
    selected_logits = logits[:, offset_mask, :].reshape(-1, logits.shape[-1])
    selected_targets = targets[:, offset_mask].reshape(-1)
    if selected_targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    weights = torch.ones(logits.shape[-1], dtype=selected_logits.dtype, device=selected_logits.device) * float(family_weight)
    weights[0] = float(no_family_weight)
    loss = torch.nn.functional.cross_entropy(selected_logits, selected_targets, weight=weights, reduction="none")
    gamma = float(focal_gamma or 0.0)
    if gamma > 0.0:
        probs = torch.softmax(selected_logits, dim=-1)
        pt = probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        loss = ((1.0 - pt) ** gamma) * loss
    return loss.mean()


def _family_token_presence_bce_loss(
    torch: Any,
    logits: Any,
    targets: Any,
    offset_mask: Any,
    *,
    pos_weight: float,
    negative_weight: float,
) -> Any:
    if logits is None or logits.shape[-1] == 0:
        return torch.tensor(0.0, device=targets.device)
    selected_logits = logits[:, offset_mask, :]
    selected_targets = targets[:, offset_mask, :]
    if selected_targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    pos = torch.tensor(float(pos_weight), dtype=selected_logits.dtype, device=selected_logits.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=pos,
        reduction="none",
    )
    neg_weight = float(negative_weight)
    weights = torch.where(selected_targets > 0.0, torch.ones_like(selected_targets), torch.full_like(selected_targets, neg_weight))
    return (loss * weights).mean()


def _family_token_presence_rank_loss(
    torch: Any,
    logits: Any,
    targets: Any,
    offset_mask: Any,
    *,
    margin: float,
    top_negatives: int = 1,
) -> Any:
    """Pairwise confidence-ranking loss for sparse action-token presence heads.

    Public FDM-1 describes iterative unmasking of the highest-confidence action
    tokens but not the internal confidence objective.  This loss remains within
    the masked action-token recipe: it trains auxiliary confidence heads, from
    fit/train labels only, to rank true sparse action tokens above same-family
    negatives.  It does not replace the masked token CE objective and is not
    calibrated on target labels.
    """

    if logits is None or logits.shape[-1] <= 1:
        return torch.tensor(0.0, device=targets.device)
    selected_logits = logits[:, offset_mask, :].reshape(-1, logits.shape[-1])
    selected_targets = targets[:, offset_mask, :].reshape(-1, logits.shape[-1])
    if selected_targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    pos_mask = selected_targets > 0.0
    neg_mask = ~pos_mask
    valid_rows = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return torch.tensor(0.0, device=targets.device)
    row_logits = selected_logits[valid_rows]
    row_pos = pos_mask[valid_rows]
    row_neg = neg_mask[valid_rows]
    neg_logits = row_logits.masked_fill(~row_neg, float("-inf"))
    k = max(1, min(int(top_negatives), int(row_logits.shape[1]) - 1))
    top_neg = torch.topk(neg_logits, k=k, dim=1).values
    if k > 1:
        top_neg = top_neg.mean(dim=1)
    else:
        top_neg = top_neg.squeeze(1)
    pos_logits = row_logits[row_pos]
    pos_row_idx = row_pos.nonzero(as_tuple=False)[:, 0]
    paired_neg = top_neg[pos_row_idx]
    return torch.nn.functional.softplus(paired_neg - pos_logits + float(margin)).mean()


def _token_presence_targets(torch: Any, target_ids: Any, vocab: Sequence[str], *, include_noop: bool = False) -> Any:
    """Return multi-hot action-token-set targets for each temporal row.

    This auxiliary target keeps the public FDM-1 shape (masked action-token
    denoising) but adds a set-level token identity signal so sparse key/button
    actions are not learned only through fixed slot positions.
    """

    batch, window, slots = target_ids.shape
    vocab_size = len(vocab)
    targets = torch.zeros((batch, window, vocab_size), dtype=torch.float32, device=target_ids.device)
    targets.scatter_(2, target_ids.reshape(batch, window * slots).reshape(batch, window, slots), 1.0)
    excluded = [
        idx
        for idx, token in enumerate(vocab)
        if token in {"<FDM1_ACTION_PAD>", FDM1_ACTION_MASK} or (not include_noop and token == FDM1_ACTION_NOOP)
    ]
    if excluded:
        targets[:, :, torch.tensor(excluded, dtype=torch.long, device=target_ids.device)] = 0.0
    return targets


def _token_presence_pos_weights(torch: Any, vocab: Sequence[str], config: dict[str, Any], *, device: Any) -> Any:
    weights = torch.ones(len(vocab), dtype=torch.float32, device=device)
    for idx, token in enumerate(vocab):
        token = str(token)
        if token in {"<FDM1_ACTION_PAD>", FDM1_ACTION_MASK}:
            weights[idx] = 0.0
        elif token == FDM1_ACTION_NOOP:
            weights[idx] = float(config.get("token_presence_noop_pos_weight", 1.0))
        elif token.startswith("KEY_"):
            weights[idx] = float(config.get("token_presence_keyboard_pos_weight", config.get("key_event_pos_weight", 8.0)))
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            weights[idx] = float(config.get("token_presence_mouse_button_pos_weight", config.get("button_event_pos_weight", 16.0)))
        elif token.startswith(("FDM1_MOUSE_DX_", "FDM1_MOUSE_DY_", "MOUSE_DX_", "MOUSE_DY_")):
            weights[idx] = float(config.get("token_presence_mouse_move_pos_weight", 2.0))
        elif token.startswith("SCROLL_"):
            weights[idx] = float(config.get("token_presence_scroll_pos_weight", 4.0))
        else:
            weights[idx] = float(config.get("token_presence_other_pos_weight", 2.0))
    return weights


def _token_presence_bce_loss(torch: Any, logits: Any, targets: Any, offset_mask: Any, pos_weights: Any) -> Any:
    if logits is None:
        return torch.tensor(0.0, device=targets.device)
    selected_logits = logits[:, offset_mask, :]
    selected_targets = targets[:, offset_mask, :]
    if selected_targets.numel() == 0:
        return torch.tensor(0.0, device=targets.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=pos_weights)


def _candidate_token_prior_weights(
    rows: Sequence[dict[str, Any]],
    *,
    vocab: Sequence[str],
    max_slots: int,
    preserve_pad_slots: bool,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build train-only action-token prior weights for candidate ranking.

    The public FDM-1 report does not disclose the IDM candidate scoring details,
    but it does describe masked-diffusion action-token prediction and iterative
    confidence unmasking.  This helper stays inside that recipe: it never reads
    target/eval labels, and only adjusts train-fit action-token candidate scores
    by the empirical sparsity of action tokens in the fit split.
    """

    if not bool(config.get("candidate_token_prior_correction", False)):
        return {}, {"status": "skipped", "reason": "disabled"}
    families = config.get("candidate_token_prior_families", ["keyboard", "mouse_button"])
    if not isinstance(families, list) or not families:
        families = ["keyboard", "mouse_button"]
    selected_families = {str(family) for family in families}
    token_set = {str(token) for token in vocab}
    counts: dict[str, int] = {
        str(token): 0
        for token in vocab
        if _is_predictable_action_token(str(token)) and _action_family(str(token)) in selected_families
    }
    for row in rows:
        seen_in_row: set[str] = set()
        for token in _target_slots_for_config(
            row,
            max_slots=max_slots,
            config=config,
            preserve_pad_slots=preserve_pad_slots,
        ):
            token = str(token)
            if token not in token_set or token not in counts:
                continue
            if bool(config.get("candidate_token_prior_count_once_per_row", True)):
                if token in seen_in_row:
                    continue
                seen_in_row.add(token)
            counts[token] += 1
    if not counts:
        return {}, {"status": "skipped", "reason": "no_selected_family_tokens"}
    smoothing = max(0.0, float(config.get("candidate_token_prior_smoothing", 8.0)))
    strength = max(0.0, float(config.get("candidate_token_prior_strength", 0.5)))
    min_weight = max(0.0, float(config.get("candidate_token_prior_min_weight", 0.25)))
    max_weight = max(min_weight, float(config.get("candidate_token_prior_max_weight", 8.0)))
    unseen_weight = max(
        min_weight,
        min(max_weight, float(config.get("candidate_token_prior_unseen_weight", 1.0))),
    )
    weights: dict[str, float] = {}
    family_payloads: dict[str, Any] = {}

    def median(values: Sequence[float]) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    for family in sorted(selected_families):
        family_tokens = [token for token in counts if _action_family(token) == family]
        if not family_tokens:
            family_payloads[family] = {"status": "skipped", "reason": "no_family_tokens"}
            continue
        smoothed = {token: float(counts[token]) + smoothing for token in family_tokens}
        positive_values = [value for token, value in smoothed.items() if counts[token] > 0]
        reference = median(positive_values or list(smoothed.values()))
        if reference <= 0.0 or strength <= 0.0:
            for token in family_tokens:
                weights[token] = 1.0
        else:
            for token in family_tokens:
                if counts[token] <= 0:
                    weights[token] = unseen_weight
                    continue
                raw = (reference / max(1e-12, smoothed[token])) ** strength
                weights[token] = min(max_weight, max(min_weight, float(raw)))
        family_payloads[family] = {
            "status": "pass",
            "tokens": len(family_tokens),
            "observed_tokens": sum(1 for token in family_tokens if counts[token] > 0),
            "unseen_tokens": sum(1 for token in family_tokens if counts[token] <= 0),
            "total_count": sum(counts[token] for token in family_tokens),
            "reference_smoothed_count": reference,
            "count": _numeric_summary([float(counts[token]) for token in family_tokens]),
            "weight": _numeric_summary([weights[token] for token in family_tokens]),
        }
    boosted = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    suppressed = sorted(weights.items(), key=lambda item: (item[1], item[0]))
    limit = max(1, int(config.get("candidate_token_prior_summary_limit", 12)))
    summary = {
        "schema": "temporal_candidate_token_prior.v1",
        "status": "pass",
        "rows": len(rows),
        "families": family_payloads,
        "strength": strength,
        "smoothing": smoothing,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "unseen_weight": unseen_weight,
        "count_once_per_row": bool(config.get("candidate_token_prior_count_once_per_row", True)),
        "top_boosted": [{"token": token, "weight": weight, "count": counts.get(token, 0)} for token, weight in boosted[:limit]],
        "top_suppressed": [{"token": token, "weight": weight, "count": counts.get(token, 0)} for token, weight in suppressed[:limit]],
        "claim_boundary": "Train-fit action-token prior correction only. Target/eval labels are never used; unseen-in-fit tokens receive a fixed neutral/suppression weight rather than an inverse-frequency boost.",
    }
    return weights, summary


def _is_predictable_action_token(token: str) -> bool:
    return token not in {"<FDM1_ACTION_PAD>", FDM1_ACTION_MASK, FDM1_ACTION_NOOP} and not token.startswith("<FDM1_ACTION_PAD")


def _action_family(token: str) -> str:
    token = str(token)
    if token.startswith("KEY_"):
        return "keyboard"
    if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
        return "mouse_button"
    if token.startswith(("FDM1_MOUSE_DX_", "FDM1_MOUSE_DY_", "MOUSE_DX_", "MOUSE_DY_")):
        return "mouse_move"
    if token.startswith("SCROLL_"):
        return "scroll"
    return "other"


def _temporal_window_features_for_batch(
    *,
    rows: Sequence[dict[str, Any]],
    start_index: int,
    all_features: Sequence[Sequence[float]],
    offsets: Sequence[int],
) -> list[list[list[float]]]:
    window_features: list[list[Any]] = []
    for local_idx, _row in enumerate(rows):
        global_idx = start_index + local_idx
        window_features.append([all_features[max(0, min(len(all_features) - 1, global_idx + offset))] for offset in offsets])
    return window_features


def _feature_sequence_tensor(torch: Any, feature_rows: Sequence[Any], *, device: Any) -> Any:
    if hasattr(feature_rows, "detach") and hasattr(feature_rows, "to"):
        if int(feature_rows.shape[0]) == 0:
            return torch.empty((0, 0), dtype=torch.float32, device=device)
        return feature_rows.to(device=device, dtype=torch.float32)
    if not feature_rows:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    first = feature_rows[0]
    if hasattr(first, "detach") and hasattr(first, "to"):
        return torch.stack([row.to(dtype=torch.float32) for row in feature_rows], dim=0).to(device=device)
    return torch.tensor([list(row) for row in feature_rows], dtype=torch.float32, device=device)


def _feature_window_tensor(torch: Any, window_features: Sequence[Sequence[Any]], *, device: Any) -> Any:
    if hasattr(window_features, "detach") and hasattr(window_features, "to"):
        if int(window_features.shape[0]) == 0:
            return torch.empty((0, 0, 0), dtype=torch.float32, device=device)
        return window_features.to(device=device, dtype=torch.float32)
    if not window_features:
        return torch.empty((0, 0, 0), dtype=torch.float32, device=device)
    first = window_features[0][0] if window_features[0] else None
    if hasattr(first, "detach") and hasattr(first, "to"):
        return torch.stack(
            [torch.stack([feature.to(dtype=torch.float32) for feature in row], dim=0) for row in window_features],
            dim=0,
        ).to(device=device)
    return torch.tensor(window_features, dtype=torch.float32, device=device)


def _temporal_final_center_probabilities(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    *,
    start_index: int,
    all_features: Sequence[Sequence[float]],
    config: dict[str, Any],
    vocab: Sequence[str],
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    offsets = _temporal_offsets(config)
    center = _center_index(offsets)
    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    mask_index = token_to_index[FDM1_ACTION_MASK]
    window_features = _temporal_window_features_for_batch(rows=rows, start_index=start_index, all_features=all_features, offsets=offsets)
    if not window_features:
        return None, None, {}
    model.eval()
    feature_tensor = _feature_window_tensor(torch, window_features, device=device)
    batch = int(feature_tensor.shape[0])
    corrupted = torch.full((batch, len(offsets), max_slots), mask_index, dtype=torch.long, device=device)
    masked = torch.ones((batch, len(offsets), max_slots), dtype=torch.bool, device=device)
    counts = iterative_unmask_counts(len(offsets) * max_slots, steps=int(config.get("diffusion_steps", 16)))
    with torch.no_grad():
        for count in counts:
            if not bool(masked.any()):
                break
            logits = model(feature_tensor, corrupted)
            probs = torch.softmax(logits, dim=-1)
            best_prob, best_id = torch.max(probs, dim=-1)
            for batch_idx in range(batch):
                flat_probs = [float(value) for value in best_prob[batch_idx].reshape(-1).detach().cpu().tolist()]
                flat_masked = [bool(value) for value in masked[batch_idx].reshape(-1).detach().cpu().tolist()]
                selected = select_topk_masked(flat_probs, flat_masked, k=count)
                for flat_idx in selected:
                    off_idx = flat_idx // max_slots
                    slot_idx = flat_idx % max_slots
                    corrupted[batch_idx, off_idx, slot_idx] = best_id[batch_idx, off_idx, slot_idx]
                    masked[batch_idx, off_idx, slot_idx] = False
        if bool(masked.any()):
            logits = model(feature_tensor, corrupted)
            best_id = torch.argmax(logits, dim=-1)
            corrupted = torch.where(masked, best_id, corrupted)
        wants_aux_payload = (
            bool(config.get("event_auxiliary_bias_candidates", config.get("temporal_event_auxiliary", config.get("event_auxiliary", False))))
            or bool(config.get("button_class_bias_candidates", config.get("temporal_button_class_auxiliary", False)))
            or bool(config.get("key_class_bias_candidates", config.get("temporal_key_class_auxiliary", False)))
            or bool(config.get("key_token_presence_bias_candidates", config.get("temporal_key_token_presence_auxiliary", False)))
            or bool(config.get("button_token_presence_bias_candidates", config.get("temporal_button_token_presence_auxiliary", False)))
            or bool(config.get("mouse_move_token_presence_bias_candidates", config.get("temporal_mouse_move_token_presence_auxiliary", False)))
            or bool(config.get("mouse_axis_class_bias_candidates", config.get("temporal_mouse_axis_class_auxiliary", False)))
            or bool(config.get("token_presence_bias_candidates", config.get("temporal_token_presence_auxiliary", config.get("token_presence_auxiliary", False))))
        )
        if wants_aux_payload and hasattr(model, "forward_with_aux"):
            payload = model.forward_with_aux(feature_tensor, corrupted)
            final_logits = payload["action_logits"]
            event_probabilities = {
                "keyboard": torch.sigmoid(payload["key_event_logits"][:, center]) if "key_event_logits" in payload else None,
                "mouse_button": torch.sigmoid(payload["button_event_logits"][:, center]) if "button_event_logits" in payload else None,
            }
            if "token_presence_logits" in payload:
                event_probabilities["token_presence"] = torch.sigmoid(payload["token_presence_logits"][:, center, :])
            if "button_class_logits" in payload:
                event_probabilities["button_class"] = torch.softmax(payload["button_class_logits"][:, center, :], dim=-1)
            if "key_class_logits" in payload:
                event_probabilities["key_class"] = torch.softmax(payload["key_class_logits"][:, center, :], dim=-1)
            if "key_token_presence_logits" in payload:
                event_probabilities["key_token_presence"] = torch.sigmoid(payload["key_token_presence_logits"][:, center, :])
            if "button_token_presence_logits" in payload:
                event_probabilities["button_token_presence"] = torch.sigmoid(payload["button_token_presence_logits"][:, center, :])
            if "mouse_move_token_presence_logits" in payload:
                event_probabilities["mouse_move_token_presence"] = torch.sigmoid(payload["mouse_move_token_presence_logits"][:, center, :])
            if "mouse_dx_class_logits" in payload:
                event_probabilities["mouse_dx_class"] = torch.softmax(payload["mouse_dx_class_logits"][:, center, :], dim=-1)
            if "mouse_dy_class_logits" in payload:
                event_probabilities["mouse_dy_class"] = torch.softmax(payload["mouse_dy_class_logits"][:, center, :], dim=-1)
        else:
            final_logits = model(feature_tensor, corrupted)
            event_probabilities = {}
        final_probs = torch.softmax(final_logits[:, center, :, :], dim=-1)
    return final_probs, corrupted[:, center, :], event_probabilities


def _temporal_center_candidates(
    probabilities: Any,
    *,
    vocab: Sequence[str],
    config: dict[str, Any],
    event_probabilities: dict[str, Any] | None = None,
    retrieval_priors: Sequence[dict[str, float]] | None = None,
    token_prior_weights: dict[str, float] | None = None,
) -> list[list[dict[str, Any]]]:
    if probabilities is None:
        return []
    probabilities_cpu = probabilities.detach().cpu()
    event_probabilities_cpu: dict[str, Any] = {}
    if event_probabilities:
        for key, value in event_probabilities.items():
            event_probabilities_cpu[key] = value.detach().cpu() if value is not None and hasattr(value, "detach") else value
    max_slots = int(probabilities.shape[1])
    max_candidates = int(config.get("non_noop_budget_candidates_per_row", max_slots * 8))
    min_candidates_per_family = max(
        0,
        int(
            config.get(
                "non_noop_budget_min_candidates_per_family",
                config.get("candidate_min_candidates_per_family", 0),
            )
            or 0
        ),
    )
    candidate_families = config.get(
        "non_noop_budget_candidate_families",
        config.get("family_non_noop_budget_families", ["keyboard", "mouse_button", "mouse_move"]),
    )
    if not isinstance(candidate_families, list) or not candidate_families:
        candidate_families = ["keyboard", "mouse_button", "mouse_move"]
    candidate_families = [str(family) for family in candidate_families]
    min_probability = float(config.get("non_noop_candidate_min_probability", 0.0))
    key_vocab = _key_class_vocab(vocab)
    key_presence_index = {token: idx for idx, token in enumerate(key_vocab)}
    key_class_index = {token: idx + 1 for idx, token in enumerate(key_vocab)}
    button_vocab = _button_class_vocab(vocab)
    button_class_index = {token: idx + 1 for idx, token in enumerate(button_vocab)}
    button_presence_index = {token: idx for idx, token in enumerate(button_vocab)}
    mouse_move_vocab = _mouse_move_class_vocab(vocab)
    mouse_move_presence_index = {token: idx for idx, token in enumerate(mouse_move_vocab)}
    mouse_dx_vocab = _mouse_axis_class_vocab(vocab, "x")
    mouse_dy_vocab = _mouse_axis_class_vocab(vocab, "y")
    mouse_dx_class_index = {token: idx + 1 for idx, token in enumerate(mouse_dx_vocab)}
    mouse_dy_class_index = {token: idx + 1 for idx, token in enumerate(mouse_dy_vocab)}
    direct_aux_families = config.get("direct_auxiliary_candidate_families", [])
    if not isinstance(direct_aux_families, list):
        direct_aux_families = []
    direct_aux_families = {str(item) for item in direct_aux_families}
    direct_aux_min_score = max(0.0, float(config.get("direct_auxiliary_candidate_min_score", 0.0) or 0.0))
    batch_candidates: list[list[dict[str, Any]]] = []
    for batch_idx in range(int(probabilities_cpu.shape[0])):
        candidates: list[dict[str, Any]] = []
        for slot_idx in range(max_slots):
            for token_idx, token in enumerate(vocab):
                token = str(token)
                if not _is_predictable_action_token(token):
                    continue
                token_score = float(probabilities_cpu[batch_idx, slot_idx, token_idx])
                score = token_score
                family = _action_family(token)
                if event_probabilities_cpu and family in {"keyboard", "mouse_button"} and event_probabilities_cpu.get(family) is not None:
                    event_score = float(event_probabilities_cpu[family][batch_idx])
                    blend = max(0.0, min(1.0, float(config.get("event_auxiliary_candidate_score_blend", 0.5))))
                    score = (1.0 - blend) * token_score + blend * event_score
                if event_probabilities_cpu and event_probabilities_cpu.get("token_presence") is not None:
                    presence_score = float(event_probabilities_cpu["token_presence"][batch_idx, token_idx])
                    blend = max(0.0, min(1.0, float(config.get("token_presence_candidate_score_blend", 0.0))))
                    if blend > 0.0:
                        score = (1.0 - blend) * score + blend * presence_score
                key_presence_score = 0.0
                if family == "keyboard" and event_probabilities_cpu and event_probabilities_cpu.get("key_token_presence") is not None:
                    class_idx = key_presence_index.get(token)
                    if class_idx is not None and class_idx < int(event_probabilities_cpu["key_token_presence"].shape[1]):
                        key_presence_score = float(event_probabilities_cpu["key_token_presence"][batch_idx, class_idx])
                        blend = max(0.0, min(1.0, float(config.get("key_token_presence_candidate_score_blend", 0.0))))
                        if blend > 0.0:
                            score = (1.0 - blend) * score + blend * key_presence_score
                key_class_score = 0.0
                if family == "keyboard" and event_probabilities_cpu and event_probabilities_cpu.get("key_class") is not None:
                    class_idx = key_class_index.get(token)
                    if class_idx is not None and class_idx < int(event_probabilities_cpu["key_class"].shape[1]):
                        key_class_score = float(event_probabilities_cpu["key_class"][batch_idx, class_idx])
                        blend = max(0.0, min(1.0, float(config.get("key_class_candidate_score_blend", 0.0))))
                        if blend > 0.0:
                            score = (1.0 - blend) * score + blend * key_class_score
                button_class_score = 0.0
                if family == "mouse_button" and event_probabilities_cpu and event_probabilities_cpu.get("button_class") is not None:
                    class_idx = button_class_index.get(token)
                    if class_idx is not None and class_idx < int(event_probabilities_cpu["button_class"].shape[1]):
                        button_class_score = float(event_probabilities_cpu["button_class"][batch_idx, class_idx])
                        blend = max(0.0, min(1.0, float(config.get("button_class_candidate_score_blend", 0.0))))
                        if blend > 0.0:
                            score = (1.0 - blend) * score + blend * button_class_score
                button_presence_score = 0.0
                if family == "mouse_button" and event_probabilities_cpu and event_probabilities_cpu.get("button_token_presence") is not None:
                    class_idx = button_presence_index.get(token)
                    if class_idx is not None and class_idx < int(event_probabilities_cpu["button_token_presence"].shape[1]):
                        button_presence_score = float(event_probabilities_cpu["button_token_presence"][batch_idx, class_idx])
                        blend = max(0.0, min(1.0, float(config.get("button_token_presence_candidate_score_blend", 0.0))))
                        if blend > 0.0:
                            score = (1.0 - blend) * score + blend * button_presence_score
                mouse_move_presence_score = 0.0
                if family == "mouse_move" and event_probabilities_cpu and event_probabilities_cpu.get("mouse_move_token_presence") is not None:
                    class_idx = mouse_move_presence_index.get(token)
                    if class_idx is not None and class_idx < int(event_probabilities_cpu["mouse_move_token_presence"].shape[1]):
                        mouse_move_presence_score = float(event_probabilities_cpu["mouse_move_token_presence"][batch_idx, class_idx])
                        blend = max(0.0, min(1.0, float(config.get("mouse_move_token_presence_candidate_score_blend", 0.0))))
                        if blend > 0.0:
                            score = (1.0 - blend) * score + blend * mouse_move_presence_score
                mouse_axis_class_score = 0.0
                axis = _mouse_axis_for_token(token)
                if family == "mouse_move" and axis is not None and event_probabilities_cpu:
                    class_key = "mouse_dx_class" if axis == "x" else "mouse_dy_class"
                    class_index = mouse_dx_class_index if axis == "x" else mouse_dy_class_index
                    if event_probabilities_cpu.get(class_key) is not None:
                        class_idx = class_index.get(token)
                        if class_idx is not None and class_idx < int(event_probabilities_cpu[class_key].shape[1]):
                            mouse_axis_class_score = float(event_probabilities_cpu[class_key][batch_idx, class_idx])
                            blend = max(0.0, min(1.0, float(config.get("mouse_axis_class_candidate_score_blend", 0.0))))
                            if blend > 0.0:
                                score = (1.0 - blend) * score + blend * mouse_axis_class_score
                retrieval_score = 0.0
                if retrieval_priors and batch_idx < len(retrieval_priors):
                    retrieval_score = float(retrieval_priors[batch_idx].get(token, 0.0))
                    if retrieval_score > 0.0:
                        blend = max(0.0, min(1.0, float(config.get("retrieval_action_prior_blend", 0.35))))
                        score = (1.0 - blend) * score + blend * retrieval_score
                prior_weight = 1.0
                if token_prior_weights:
                    prior_weight = float(token_prior_weights.get(token, 1.0))
                    score = max(0.0, min(1.0, score * prior_weight))
                event_gate_multiplier = 1.0
                gate_families = config.get("event_auxiliary_candidate_gate_families", ["keyboard", "mouse_button"])
                if not isinstance(gate_families, list):
                    gate_families = ["keyboard", "mouse_button"]
                if (
                    event_probabilities_cpu
                    and family in {str(item) for item in gate_families}
                    and event_probabilities_cpu.get(family) is not None
                ):
                    event_score = float(event_probabilities_cpu[family][batch_idx])
                    gate_power = max(
                        0.0,
                        float(
                            config.get(
                                f"{family}_event_candidate_gate_power",
                                config.get("event_auxiliary_candidate_gate_power", 0.0),
                            )
                            or 0.0
                        ),
                    )
                    if gate_power > 0.0:
                        gate_floor = max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    config.get(
                                        f"{family}_event_candidate_gate_floor",
                                        config.get("event_auxiliary_candidate_gate_floor", 0.0),
                                    )
                                    or 0.0
                                ),
                            ),
                        )
                        event_gate_multiplier *= gate_floor + (1.0 - gate_floor) * (max(0.0, min(1.0, event_score)) ** gate_power)
                button_class_no_button_gate_score = 1.0
                if family == "mouse_button" and event_probabilities_cpu and event_probabilities_cpu.get("button_class") is not None:
                    button_class_probs = event_probabilities_cpu["button_class"]
                    if int(button_class_probs.shape[1]) > 0:
                        no_button_prob = float(button_class_probs[batch_idx, 0])
                        button_class_no_button_gate_score = max(0.0, min(1.0, 1.0 - no_button_prob))
                        gate_power = max(0.0, float(config.get("button_class_no_button_gate_power", 0.0) or 0.0))
                        if gate_power > 0.0:
                            gate_floor = max(0.0, min(1.0, float(config.get("button_class_no_button_gate_floor", 0.0) or 0.0)))
                            event_gate_multiplier *= gate_floor + (1.0 - gate_floor) * (button_class_no_button_gate_score ** gate_power)
                if event_gate_multiplier < 1.0:
                    score = max(0.0, min(1.0, score * event_gate_multiplier))
                if score < min_probability:
                    continue
                candidates.append(
                    {
                        "score": score,
                        "token_probability": token_score,
                        "retrieval_score": retrieval_score,
                        "prior_weight": prior_weight,
                        "event_gate_multiplier": event_gate_multiplier,
                        "key_presence_score": key_presence_score,
                        "key_class_score": key_class_score,
                        "button_class_score": button_class_score,
                        "button_class_no_button_gate_score": button_class_no_button_gate_score,
                        "button_presence_score": button_presence_score,
                        "mouse_move_presence_score": mouse_move_presence_score,
                        "mouse_axis_class_score": mouse_axis_class_score,
                        "slot": slot_idx,
                        "token_index": token_idx,
                        "token": token,
                        "family": family,
                    }
                )
        if direct_aux_families and event_probabilities_cpu:
            if "keyboard" in direct_aux_families and event_probabilities_cpu.get("key_token_presence") is not None:
                key_presence = event_probabilities_cpu["key_token_presence"]
                key_class_probs = event_probabilities_cpu.get("key_class")
                for token, class_idx in key_presence_index.items():
                    if class_idx >= int(key_presence.shape[1]):
                        continue
                    presence_score = float(key_presence[batch_idx, class_idx])
                    class_prob_idx = key_class_index.get(token)
                    class_score = (
                        float(key_class_probs[batch_idx, class_prob_idx])
                        if key_class_probs is not None and class_prob_idx is not None and class_prob_idx < int(key_class_probs.shape[1])
                        else 0.0
                    )
                    blend = max(
                        0.0,
                        min(1.0, float(config.get("direct_auxiliary_key_class_blend", 0.0) or 0.0)),
                    )
                    score = (1.0 - blend) * presence_score + blend * class_score
                    prior_weight = float(token_prior_weights.get(token, 1.0)) if token_prior_weights else 1.0
                    if bool(config.get("direct_auxiliary_candidate_apply_token_prior", False)):
                        score = max(0.0, min(1.0, score * prior_weight))
                    if score < direct_aux_min_score:
                        continue
                    candidates.append(
                        {
                            "score": score,
                            "token_probability": 0.0,
                            "retrieval_score": 0.0,
                            "prior_weight": prior_weight,
                            "event_gate_multiplier": 1.0,
                            "key_presence_score": presence_score,
                            "key_class_score": class_score,
                            "button_class_score": 0.0,
                            "button_class_no_button_gate_score": 1.0,
                            "button_presence_score": 0.0,
                            "mouse_move_presence_score": 0.0,
                            "mouse_axis_class_score": 0.0,
                            "slot": -1,
                            "token_index": key_presence_index[token],
                            "token": token,
                            "family": "keyboard",
                            "direct_auxiliary_candidate": "key_token_presence",
                        }
                    )
            if "mouse_button" in direct_aux_families:
                button_presence = event_probabilities_cpu.get("button_token_presence")
                button_class_probs = event_probabilities_cpu.get("button_class")
                no_button_gate = 1.0
                if button_class_probs is not None and int(button_class_probs.shape[1]) > 0:
                    no_button_gate = max(0.0, min(1.0, 1.0 - float(button_class_probs[batch_idx, 0])))
                for token in button_vocab:
                    class_idx = button_presence_index.get(token)
                    class_prob_idx = button_class_index.get(token)
                    presence_score = (
                        float(button_presence[batch_idx, class_idx])
                        if button_presence is not None and class_idx is not None and class_idx < int(button_presence.shape[1])
                        else 0.0
                    )
                    class_score = (
                        float(button_class_probs[batch_idx, class_prob_idx])
                        if button_class_probs is not None and class_prob_idx is not None and class_prob_idx < int(button_class_probs.shape[1])
                        else 0.0
                    )
                    blend = max(
                        0.0,
                        min(1.0, float(config.get("direct_auxiliary_button_class_blend", 0.5) or 0.0)),
                    )
                    score = (1.0 - blend) * presence_score + blend * class_score
                    gate_power = max(
                        0.0,
                        float(config.get("direct_auxiliary_button_no_button_gate_power", 0.0) or 0.0),
                    )
                    if gate_power > 0.0:
                        gate_floor = max(
                            0.0,
                            min(1.0, float(config.get("direct_auxiliary_button_no_button_gate_floor", 0.0) or 0.0)),
                        )
                        score *= gate_floor + (1.0 - gate_floor) * (no_button_gate**gate_power)
                    prior_weight = float(token_prior_weights.get(token, 1.0)) if token_prior_weights else 1.0
                    if bool(config.get("direct_auxiliary_candidate_apply_token_prior", False)):
                        score = max(0.0, min(1.0, score * prior_weight))
                    if score < direct_aux_min_score:
                        continue
                    token_index = next((idx for idx, vocab_token in enumerate(vocab) if str(vocab_token) == token), class_idx or 0)
                    candidates.append(
                        {
                            "score": score,
                            "token_probability": 0.0,
                            "retrieval_score": 0.0,
                            "prior_weight": prior_weight,
                            "event_gate_multiplier": 1.0,
                            "key_presence_score": 0.0,
                            "key_class_score": 0.0,
                            "button_class_score": class_score,
                            "button_class_no_button_gate_score": no_button_gate,
                            "button_presence_score": presence_score,
                            "mouse_move_presence_score": 0.0,
                            "mouse_axis_class_score": 0.0,
                            "slot": -1,
                            "token_index": token_index,
                            "token": token,
                            "family": "mouse_button",
                            "direct_auxiliary_candidate": "button_presence_class",
                        }
                    )
            if "mouse_move" in direct_aux_families and (
                event_probabilities_cpu.get("mouse_move_token_presence") is not None
                or event_probabilities_cpu.get("token_presence") is not None
                or event_probabilities_cpu.get("mouse_dx_class") is not None
                or event_probabilities_cpu.get("mouse_dy_class") is not None
            ):
                token_presence = event_probabilities_cpu.get("mouse_move_token_presence")
                direct_source = "mouse_move_token_presence"
                if token_presence is None:
                    token_presence = event_probabilities_cpu.get("token_presence")
                    direct_source = "token_presence_mouse_move"
                for token_idx, token in enumerate(vocab):
                    token = str(token)
                    if _action_family(token) != "mouse_move":
                        continue
                    presence_score = 0.0
                    if direct_source == "mouse_move_token_presence":
                        class_idx = mouse_move_presence_index.get(token)
                        if token_presence is not None and class_idx is not None and class_idx < int(token_presence.shape[1]):
                            presence_score = float(token_presence[batch_idx, class_idx])
                    elif token_presence is not None and token_idx < int(token_presence.shape[1]):
                        presence_score = float(token_presence[batch_idx, token_idx])
                    axis = _mouse_axis_for_token(token)
                    axis_score = 0.0
                    if axis is not None:
                        class_key = "mouse_dx_class" if axis == "x" else "mouse_dy_class"
                        class_index = mouse_dx_class_index if axis == "x" else mouse_dy_class_index
                        axis_probs = event_probabilities_cpu.get(class_key)
                        class_prob_idx = class_index.get(token)
                        if axis_probs is not None and class_prob_idx is not None and class_prob_idx < int(axis_probs.shape[1]):
                            axis_score = float(axis_probs[batch_idx, class_prob_idx])
                    blend = max(
                        0.0,
                        min(1.0, float(config.get("direct_auxiliary_mouse_axis_class_blend", 0.0) or 0.0)),
                    )
                    score = (1.0 - blend) * presence_score + blend * axis_score
                    if score < direct_aux_min_score:
                        continue
                    direct_label = direct_source
                    if blend > 0.0 and axis_score > 0.0:
                        direct_label = f"{direct_source}_axis_class"
                    candidates.append(
                        {
                            "score": score,
                            "token_probability": 0.0,
                            "retrieval_score": 0.0,
                            "prior_weight": 1.0,
                            "event_gate_multiplier": 1.0,
                            "key_presence_score": 0.0,
                            "key_class_score": 0.0,
                            "button_class_score": 0.0,
                            "button_class_no_button_gate_score": 1.0,
                            "button_presence_score": 0.0,
                            "mouse_move_presence_score": presence_score,
                            "mouse_axis_class_score": axis_score,
                            "slot": -1,
                            "token_index": token_idx,
                            "token": token,
                            "family": "mouse_move",
                            "direct_auxiliary_candidate": direct_label,
                        }
                    )
        candidates.sort(key=lambda item: (-float(item["score"]), int(item["slot"]), int(item["token_index"])))
        if min_candidates_per_family > 0:
            selected: list[dict[str, Any]] = []
            seen: set[tuple[int, int]] = set()

            def add_candidate(candidate: dict[str, Any]) -> None:
                key = (int(candidate.get("slot", -1)), int(candidate.get("token_index", -1)))
                if key in seen:
                    return
                selected.append(candidate)
                seen.add(key)

            for family in candidate_families:
                emitted = 0
                for candidate in candidates:
                    if str(candidate.get("family", _action_family(str(candidate.get("token", ""))))) != family:
                        continue
                    add_candidate(candidate)
                    emitted += 1
                    if emitted >= min_candidates_per_family:
                        break
            effective_max_candidates = max(max_candidates, min_candidates_per_family * len(candidate_families))
            for candidate in candidates:
                if len(selected) >= effective_max_candidates:
                    break
                add_candidate(candidate)
            selected.sort(key=lambda item: (-float(item["score"]), int(item["slot"]), int(item["token_index"])))
            batch_candidates.append(selected[:effective_max_candidates])
        else:
            batch_candidates.append(candidates[:max_candidates])
    return batch_candidates


def _retrieval_tokens(row: dict[str, Any], *, max_slots: int, config: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _target_slots_for_config(row, max_slots=max_slots, config=config, preserve_pad_slots=True):
        token = str(token)
        if not _is_predictable_action_token(token) or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def _build_temporal_retrieval_prior_index(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[Sequence[float]],
    *,
    config: dict[str, Any],
    vocab: Sequence[str],
    device: Any,
) -> dict[str, Any]:
    if not bool(config.get("retrieval_action_prior_enabled", False)):
        return {"status": "skipped", "reason": "disabled"}
    max_rows = config.get("retrieval_action_prior_max_rows")
    limit = min(len(rows), int(max_rows)) if max_rows is not None else len(rows)
    if limit <= 0:
        return {"status": "skipped", "reason": "no_rows"}
    batch_size = max(1, int(config.get("retrieval_action_prior_batch_size", config.get("prediction_batch_size", config.get("batch_size", 64)))))
    token_vocab = set(str(token) for token in vocab)
    embeddings: list[Any] = []
    token_rows: list[list[str]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, limit, batch_size):
            batch_features = _feature_sequence_tensor(torch, features[start : start + batch_size], device=device).unsqueeze(1)
            if hasattr(model, "video_summary_embedding"):
                encoded = model.video_summary_embedding(batch_features)[:, 0, :]
            else:
                encoded = model.video_embedding(batch_features)
                if encoded.dim() == 4:
                    encoded = encoded.mean(dim=2)
                encoded = encoded[:, 0, :]
            encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            embeddings.append(encoded.detach())
            for row in rows[start : start + batch_size]:
                token_rows.append(
                    [
                        token
                        for token in _retrieval_tokens(
                            row,
                            max_slots=int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16))),
                            config=config,
                        )
                        if token in token_vocab
                    ]
                )
    if not embeddings:
        return {"status": "skipped", "reason": "no_embeddings"}
    matrix = torch.cat(embeddings, dim=0).to(device)
    return {
        "schema": "temporal_retrieval_action_prior_index.v1",
        "status": "pass",
        "rows": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "tokens": token_rows,
        "embeddings": matrix,
        "top_k": int(config.get("retrieval_action_prior_top_k", 16)),
        "temperature": float(config.get("retrieval_action_prior_temperature", 0.07)),
        "claim_boundary": "Retrieval index uses fit/train rows only as a video/action-token denoising prior; target labels are never indexed.",
    }


def _retrieval_priors_for_batch(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[Sequence[float]],
    *,
    retrieval_index: dict[str, Any] | None,
    config: dict[str, Any],
    device: Any,
) -> list[dict[str, float]] | None:
    del rows
    if not retrieval_index or retrieval_index.get("status") != "pass":
        return None
    embeddings = retrieval_index.get("embeddings")
    token_rows = retrieval_index.get("tokens")
    if embeddings is None or not token_rows:
        return None
    if (hasattr(features, "shape") and int(features.shape[0]) == 0) or (not hasattr(features, "shape") and not features):
        return []
    top_k = max(1, min(int(config.get("retrieval_action_prior_top_k", retrieval_index.get("top_k", 16))), int(embeddings.shape[0])))
    temperature = max(1e-6, float(config.get("retrieval_action_prior_temperature", retrieval_index.get("temperature", 0.07))))
    batch_features = _feature_sequence_tensor(torch, features, device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "video_summary_embedding"):
            query = model.video_summary_embedding(batch_features)[:, 0, :]
        else:
            query = model.video_embedding(batch_features)
            if query.dim() == 4:
                query = query.mean(dim=2)
            query = query[:, 0, :]
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        scores = query @ embeddings.T
        values, indices = torch.topk(scores, k=top_k, dim=1)
        weights = torch.softmax(values / temperature, dim=1)
    priors: list[dict[str, float]] = []
    for row_indices, row_weights in zip(indices.detach().cpu().tolist(), weights.detach().cpu().tolist()):
        acc: dict[str, float] = {}
        for retrieval_row_idx, weight in zip(row_indices, row_weights):
            for token in token_rows[int(retrieval_row_idx)]:
                acc[token] = acc.get(token, 0.0) + float(weight)
        priors.append(acc)
    return priors


def _tokens_from_non_noop_candidates(candidates: Sequence[dict[str, Any]], *, threshold: float, max_tokens: int) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if float(candidate.get("score", 0.0)) < threshold:
            continue
        token = str(candidate.get("token", ""))
        if not _is_predictable_action_token(token) or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
        if len(tokens) >= max_tokens:
            break
    return tokens


def _tokens_from_family_budget_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    family_budgets: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    config = config or {}
    mouse_axis_constrained = bool(config.get("mouse_move_axis_constrained_budget", False))
    families = family_budgets.get("families", family_budgets) if isinstance(family_budgets, dict) else {}
    if not isinstance(families, dict):
        return tokens
    for family, budget in families.items():
        if not isinstance(budget, dict) or budget.get("status") not in {None, "pass"}:
            continue
        threshold = float(budget.get("selected_threshold", budget.get("threshold", 1.1)))
        max_tokens = max(0, int(budget.get("max_tokens_per_row", 0)))
        if max_tokens <= 0:
            continue
        emitted = 0
        emitted_mouse_axes: set[str] = set()
        for candidate in candidates:
            token = str(candidate.get("token", ""))
            if _action_family(token) != str(family):
                continue
            if float(candidate.get("score", 0.0)) < threshold:
                continue
            if not _is_predictable_action_token(token) or token in seen:
                continue
            if str(family) == "mouse_move" and mouse_axis_constrained:
                axis = _mouse_axis_for_token(token)
                if axis is not None and axis in emitted_mouse_axes:
                    continue
                if axis is not None:
                    emitted_mouse_axes.add(axis)
            tokens.append(token)
            seen.add(token)
            emitted += 1
            if emitted >= max_tokens:
                break
    return tokens


def _predict_temporal_tokens_batch(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[Sequence[float]],
    *,
    start_index: int,
    all_features: Sequence[Sequence[float]],
    config: dict[str, Any],
    vocab: Sequence[str],
    device: Any,
    retrieval_index: dict[str, Any] | None = None,
    token_prior_weights: dict[str, float] | None = None,
) -> list[list[str]]:
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    noop_index = token_to_index.get(FDM1_ACTION_NOOP, token_to_index[FDM1_ACTION_MASK])
    center_probs, predicted_center_ids, event_probabilities = _temporal_final_center_probabilities(
        model,
        torch,
        rows,
        start_index=start_index,
        all_features=all_features,
        config=config,
        vocab=vocab,
        device=device,
    )
    if center_probs is None or predicted_center_ids is None:
        return []
    retrieval_priors = _retrieval_priors_for_batch(model, torch, rows, features, retrieval_index=retrieval_index, config=config, device=device)
    candidate_rows = _temporal_center_candidates(
        center_probs,
        vocab=vocab,
        config=config,
        event_probabilities=event_probabilities,
        retrieval_priors=retrieval_priors,
        token_prior_weights=token_prior_weights,
    )
    family_budgeted = bool(config.get("family_non_noop_budgeted_unmasking", False))
    family_budgets = config.get("family_non_noop_budget", {})
    budgeted = bool(config.get("non_noop_budgeted_unmasking", False))
    budget_threshold = float(config.get("non_noop_budget_score_threshold", config.get("non_noop_threshold", 1.1)))
    budget_max_tokens = max(1, int(config.get("non_noop_budget_max_tokens_per_row", config.get("max_predicted_non_noop_tokens", 4))))
    predictions: list[list[str]] = []
    for batch_idx, row in enumerate(rows):
        center_ids = [int(value) for value in predicted_center_ids[batch_idx, :].detach().cpu().tolist()]
        if family_budgeted:
            fdm1_tokens = _tokens_from_family_budget_candidates(
                candidate_rows[batch_idx],
                family_budgets=family_budgets,
                config=config,
            )
        elif budgeted:
            fdm1_tokens = _tokens_from_non_noop_candidates(candidate_rows[batch_idx], threshold=budget_threshold, max_tokens=budget_max_tokens)
        else:
            fdm1_tokens = [str(vocab[idx]) for idx in center_ids if idx != noop_index and _is_predictable_action_token(str(vocab[idx]))]
        width, height = _screen_size(row)
        predictions.append(d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height) or [FDM1_ACTION_NOOP])
    return predictions


def _collect_temporal_probability_rows(
    model: Any,
    torch: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[Sequence[float]],
    *,
    config: dict[str, Any],
    vocab: Sequence[str],
    device: Any,
    retrieval_index: dict[str, Any] | None = None,
    token_prior_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    batch_size = max(1, int(config.get("prediction_batch_size", config.get("batch_size", 64))))
    collected: list[dict[str, Any]] = []
    for start_idx in range(0, len(rows), batch_size):
        batch_rows = list(rows[start_idx : start_idx + batch_size])
        center_probs, _center_ids, event_probabilities = _temporal_final_center_probabilities(
            model,
            torch,
            batch_rows,
            start_index=start_idx,
            all_features=features,
            config=config,
            vocab=vocab,
            device=device,
        )
        retrieval_priors = _retrieval_priors_for_batch(model, torch, batch_rows, features[start_idx : start_idx + batch_size], retrieval_index=retrieval_index, config=config, device=device)
        for row, candidates in zip(
            batch_rows,
            _temporal_center_candidates(
                center_probs,
                vocab=vocab,
                config=config,
                event_probabilities=event_probabilities,
                retrieval_priors=retrieval_priors,
                token_prior_weights=token_prior_weights,
            ),
        ):
            collected.append(
                {
                    "row": row,
                    "candidates": candidates,
                    "ground_truth_tokens": [str(token) for token in row.get("ground_truth_tokens", [])],
                    "ground_truth_fdm1_tokens": [
                        str(token)
                        for token in canonical_fdm1_action_tokens(
                            row,
                            default_width=_screen_size(row)[0],
                            default_height=_screen_size(row)[1],
                            include_noop=False,
                            mouse_token_mode=_action_mouse_token_mode(config),
                        )
                    ],
                }
            )
    return collected


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
    }


def _candidate_family_diagnostics(probability_rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether recipe-shaped action-token candidates exist per family.

    This is an audit/diagnostic surface only.  It may use ground-truth labels for
    reporting positive-row coverage, but it never changes calibration thresholds
    or predictions.
    """

    if not probability_rows:
        return {"status": "skipped", "reason": "no_probability_rows"}
    families = config.get("candidate_diagnostic_families", config.get("family_non_noop_budget_families", ["keyboard", "mouse_button", "mouse_move"]))
    if not isinstance(families, list) or not families:
        families = ["keyboard", "mouse_button", "mouse_move"]
    top_tokens_limit = max(1, int(config.get("candidate_diagnostic_top_tokens", 12)))
    payload: dict[str, Any] = {
        "schema": "temporal_candidate_family_diagnostics.v1",
        "status": "pass",
        "rows": len(probability_rows),
        "families": {},
        "claim_boundary": "Diagnostic-only summary. Ground-truth labels may be used to audit candidate coverage, but candidate diagnostics do not calibrate or mutate predictions.",
    }
    for family in [str(item) for item in families]:
        total_candidates = 0
        rows_with_candidates = 0
        positive_rows = 0
        positive_rows_with_family_candidate = 0
        positive_rows_with_exact_candidate = 0
        exact_ranks: list[float] = []
        top_family_scores: list[float] = []
        all_family_scores: list[float] = []
        top_token_counts: dict[str, int] = {}
        for row in probability_rows:
            candidates = [
                candidate
                for candidate in row.get("candidates", [])
                if str(candidate.get("family", _action_family(str(candidate.get("token", ""))))) == family
            ]
            total_candidates += len(candidates)
            if candidates:
                rows_with_candidates += 1
                top_family_scores.append(float(candidates[0].get("score", 0.0)))
                top_token = str(candidates[0].get("token", ""))
                top_token_counts[top_token] = top_token_counts.get(top_token, 0) + 1
                all_family_scores.extend(float(candidate.get("score", 0.0)) for candidate in candidates)
            # Diagnostics rank recipe-shaped FDM-1 action-token candidates, so
            # compare them against canonical FDM-1 target tokens when available.
            # D2E paper metrics still use raw/coarse metric tokens elsewhere.
            diagnostic_gt = row.get("ground_truth_fdm1_tokens") or row.get("ground_truth_tokens", [])
            gt_tokens = [str(token) for token in diagnostic_gt if _action_family(str(token)) == family]
            if not gt_tokens:
                continue
            positive_rows += 1
            candidate_tokens = [str(candidate.get("token", "")) for candidate in candidates]
            if candidate_tokens:
                positive_rows_with_family_candidate += 1
            candidate_token_set = set(candidate_tokens)
            if any(token in candidate_token_set for token in gt_tokens):
                positive_rows_with_exact_candidate += 1
            for gt_token in dict.fromkeys(gt_tokens):
                for rank, candidate in enumerate(row.get("candidates", []), 1):
                    if str(candidate.get("token", "")) == gt_token:
                        exact_ranks.append(float(rank))
                        break
        top_tokens = [
            {"token": token, "rows_as_top_family_candidate": count}
            for token, count in sorted(top_token_counts.items(), key=lambda item: (-item[1], item[0]))[:top_tokens_limit]
        ]
        payload["families"][family] = {
            "rows": len(probability_rows),
            "total_candidates": total_candidates,
            "rows_with_candidates": rows_with_candidates,
            "candidate_row_rate": rows_with_candidates / max(1, len(probability_rows)),
            "positive_rows": positive_rows,
            "positive_rows_with_family_candidate": positive_rows_with_family_candidate,
            "positive_rows_with_family_candidate_rate": positive_rows_with_family_candidate / max(1, positive_rows),
            "positive_rows_with_exact_candidate": positive_rows_with_exact_candidate,
            "positive_rows_with_exact_candidate_rate": positive_rows_with_exact_candidate / max(1, positive_rows),
            "exact_candidate_rank": _numeric_summary(exact_ranks),
            "top_family_candidate_score": _numeric_summary(top_family_scores),
            "all_family_candidate_score": _numeric_summary(all_family_scores),
            "top_tokens": top_tokens,
        }
    return payload


def _family_budget_score(family: str, metrics: dict[str, Any], *, max_button_fpr: float, beta: float) -> float:
    paper = metrics["paper_compatible"]
    strict = metrics["strict_local"]
    if family == "keyboard":
        return float(paper["keyboard"].get("key_accuracy") or 0.0) + 0.25 * float(strict["keyboard"].get("accuracy") or 0.0)
    if family == "mouse_button":
        strict_button = strict["mouse_button"]
        precision = float(strict_button.get("precision") or 0.0)
        recall = float(strict_button.get("recall") or 0.0)
        if precision + recall > 0:
            beta2 = beta * beta
            fbeta = (1.0 + beta2) * precision * recall / max(1e-12, beta2 * precision + recall)
        else:
            fbeta = 0.0
        fpr = strict_button.get("no_button_false_positive_rate")
        fpr_penalty = 0.0 if fpr is None or float(fpr) <= max_button_fpr else (float(fpr) - max_button_fpr) * 10.0
        return fbeta + 0.25 * float(paper["mouse_button"].get("button_accuracy") or 0.0) - fpr_penalty
    if family == "mouse_move":
        move = paper["mouse_move"]
        px = float(move.get("pearson_x") or 0.0)
        py = float(move.get("pearson_y") or 0.0)
        return 0.5 * (px + py)
    return 0.0


def _calibrate_temporal_family_non_noop_budget(probability_rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    if not probability_rows:
        return {"status": "skipped", "reason": "no_probability_rows"}
    families = config.get("family_non_noop_budget_families", ["keyboard", "mouse_button", "mouse_move"])
    if not isinstance(families, list):
        families = ["keyboard", "mouse_button", "mouse_move"]
    max_button_fpr = float(config.get("family_non_noop_budget_max_no_button_fpr", config.get("non_noop_budget_max_no_button_fpr", 0.10)))
    beta = float(config.get("family_non_noop_budget_beta", config.get("non_noop_budget_beta", 2.0)))
    max_threshold_candidates = max(2, int(config.get("family_non_noop_budget_max_threshold_candidates", config.get("non_noop_budget_max_threshold_candidates", 256))))
    family_payloads: dict[str, Any] = {}
    for family in [str(item) for item in families]:
        family_candidates = [
            float(candidate.get("score", 0.0))
            for row in probability_rows
            for candidate in row.get("candidates", [])
            if _action_family(str(candidate.get("token", ""))) == family
        ]
        if not family_candidates:
            family_payloads[family] = {"status": "skipped", "reason": "no_family_candidates", "max_tokens_per_row": 0}
            continue
        raw_candidates = [
            float(value)
            for value in config.get(f"family_non_noop_budget_{family}_threshold_candidates", config.get("family_non_noop_budget_threshold_candidates", []))
            if isinstance(value, (int, float))
        ]
        raw_candidates.extend(family_candidates)
        raw_candidates.extend(min(1.0, value + 1e-6) for value in family_candidates)
        raw_candidates.append(1.1)
        thresholds = sorted({max(0.0, min(1.1, float(value))) for value in raw_candidates})
        if len(thresholds) > max_threshold_candidates:
            thresholds = [
                thresholds[min(len(thresholds) - 1, int(round(idx * (len(thresholds) - 1) / (max_threshold_candidates - 1))))]
                for idx in range(max_threshold_candidates)
            ]
            thresholds = sorted(set(thresholds + [1.1]))
        max_tokens = max(
            0,
            int(
                config.get(
                    f"family_non_noop_budget_{family}_max_tokens_per_row",
                    config.get("family_non_noop_budget_max_tokens_per_row", config.get("non_noop_budget_max_tokens_per_row", 4)),
                )
            ),
        )
        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            acc = _PaperMetricAccumulator(empty_bins_as_correct=False)
            predicted_non_noop = 0
            for item in probability_rows:
                family_row_candidates = [candidate for candidate in item.get("candidates", []) if _action_family(str(candidate.get("token", ""))) == family]
                fdm1_tokens = _tokens_from_non_noop_candidates(family_row_candidates, threshold=threshold, max_tokens=max_tokens)
                predicted_non_noop += len(fdm1_tokens)
                width, height = _screen_size(item["row"])
                pred_tokens = d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height) or [FDM1_ACTION_NOOP]
                acc.update(pred_tokens, item.get("ground_truth_tokens", []))
            metrics = acc.metrics()
            score = _family_budget_score(family, metrics, max_button_fpr=max_button_fpr, beta=beta)
            strict_button = metrics["strict_local"]["mouse_button"]
            rows.append(
                {
                    "threshold": threshold,
                    "score": score,
                    "predicted_non_noop_tokens": predicted_non_noop,
                    "keyboard_key_accuracy": metrics["paper_compatible"]["keyboard"].get("key_accuracy"),
                    "mouse_button_accuracy": metrics["paper_compatible"]["mouse_button"].get("button_accuracy"),
                    "mouse_button_f1": strict_button.get("f1"),
                    "no_button_false_positive_rate": strict_button.get("no_button_false_positive_rate"),
                    "mouse_move_pearson_x": metrics["paper_compatible"]["mouse_move"].get("pearson_x"),
                    "mouse_move_pearson_y": metrics["paper_compatible"]["mouse_move"].get("pearson_y"),
                }
            )
        selected = max(rows, key=lambda row: (float(row["score"]), int(row["predicted_non_noop_tokens"]), -float(row["threshold"])))
        family_payloads[family] = {
            "status": "pass",
            "family": family,
            "selected_threshold": float(selected["threshold"]),
            "selected_row": selected,
            "candidate_count": len(rows),
            "max_tokens_per_row": max_tokens,
            "calibration_predicted_tokens_per_row": float(selected["predicted_non_noop_tokens"]) / max(1, len(probability_rows)),
            "calibration_positive_row_rate": sum(
                1
                for item in probability_rows
                if any(_action_family(str(token)) == family for token in item.get("ground_truth_tokens", []))
            )
            / max(1, len(probability_rows)),
            "sweep_preview": rows[:5] + ([] if len(rows) <= 10 else [{"omitted_rows": len(rows) - 10}]) + rows[-5:],
        }
    return {
        "schema": "temporal_family_non_noop_budget_calibration.v1",
        "status": "pass" if any(row.get("status") == "pass" for row in family_payloads.values()) else "skipped",
        "rows": len(probability_rows),
        "families": family_payloads,
        "max_no_button_fpr": max_button_fpr,
        "claim_boundary": "Family budgets are calibrated on held-out training rows only; target labels remain evaluation/diagnostic only.",
    }


def _adapt_temporal_family_budget_to_unlabeled_distribution(
    family_budget: dict[str, Any],
    target_probability_rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Adapt fixed family thresholds to an unlabeled target score distribution.

    Public FDM-1 IDM inference iteratively unmasks the highest-confidence action
    tokens rather than relying on a globally calibrated probability scale.  In
    practice our raw-video probes showed large score-distribution shift between
    train-heldout calibration rows and D2E target rows, so a fixed score
    threshold could over-unmask mouse-button events.  This helper keeps the same
    recipe shape by using only unlabeled candidate scores on the target split to
    match a train-heldout action-token emission budget.  It must never inspect
    target ground-truth tokens.
    """

    if not target_probability_rows:
        return family_budget
    if not isinstance(family_budget, dict) or family_budget.get("status") != "pass":
        return family_budget
    families_payload = family_budget.get("families")
    if not isinstance(families_payload, dict):
        return family_budget
    enabled_families = config.get("adaptive_family_budget_families", ["mouse_button"])
    if not isinstance(enabled_families, list) or not enabled_families:
        enabled_families = ["mouse_button"]
    enabled = {str(item) for item in enabled_families}
    multiplier = max(0.0, float(config.get("adaptive_family_budget_rate_multiplier", 1.0) or 0.0))
    epsilon = max(0.0, float(config.get("adaptive_family_budget_threshold_epsilon", 1e-12) or 0.0))
    adapted = json.loads(json.dumps(family_budget))
    adaptation_payload: dict[str, Any] = {
        "schema": "temporal_unlabeled_family_budget_adaptation.v1",
        "status": "pass",
        "rows": len(target_probability_rows),
        "families": {},
        "claim_boundary": "Uses only unlabeled target candidate scores and train-heldout emission budgets; target ground-truth labels are not inspected.",
    }
    adapted_families = adapted.get("families", {})
    for family, budget in list(adapted_families.items()):
        family = str(family)
        if family not in enabled or not isinstance(budget, dict) or budget.get("status") not in {None, "pass"}:
            continue
        max_tokens = max(0, int(budget.get("max_tokens_per_row", 0) or 0))
        if max_tokens <= 0:
            continue
        explicit_rate = config.get(f"adaptive_family_budget_{family}_tokens_per_row")
        if explicit_rate is not None:
            desired_tokens_per_row = max(0.0, float(explicit_rate))
            budget_source = "explicit_config"
        else:
            desired_tokens_per_row = max(
                0.0,
                float(
                    budget.get(
                        "calibration_predicted_tokens_per_row",
                        float((budget.get("selected_row") or {}).get("predicted_non_noop_tokens", 0.0))
                        / max(1, int(family_budget.get("rows", 0) or 0)),
                    )
                    or 0.0
                )
                * multiplier,
            )
            budget_source = "train_heldout_selected_emission_rate"
        desired_tokens_per_row = min(float(max_tokens), desired_tokens_per_row)
        target_token_budget = int(round(desired_tokens_per_row * len(target_probability_rows)))
        scores = sorted(
            (
                float(candidate.get("score", 0.0))
                for item in target_probability_rows
                for candidate in item.get("candidates", [])
                if _action_family(str(candidate.get("token", ""))) == family
            ),
            reverse=True,
        )
        if not scores or target_token_budget <= 0:
            adapted_threshold = 1.1
        else:
            target_token_budget = max(1, min(target_token_budget, len(scores)))
            adapted_threshold = min(1.1, float(scores[target_token_budget - 1]) + epsilon)
        old_threshold = float(budget.get("selected_threshold", budget.get("threshold", 1.1)))
        # Conservative by default: never lower a train-heldout threshold unless
        # explicitly requested, because the observed failure mode is over-
        # emission under score-distribution shift.
        if bool(config.get("adaptive_family_budget_only_raise_threshold", True)):
            adapted_threshold = max(old_threshold, adapted_threshold)
        budget["selected_threshold"] = adapted_threshold
        budget["unlabeled_adapted_threshold"] = adapted_threshold
        budget["unlabeled_pre_adaptation_threshold"] = old_threshold
        adaptation_payload["families"][family] = {
            "status": "pass",
            "family": family,
            "old_threshold": old_threshold,
            "adapted_threshold": adapted_threshold,
            "target_candidate_scores": len(scores),
            "target_token_budget": target_token_budget,
            "desired_tokens_per_row": desired_tokens_per_row,
            "max_tokens_per_row": max_tokens,
            "budget_source": budget_source,
        }
    if not adaptation_payload["families"]:
        adaptation_payload["status"] = "skipped"
        adaptation_payload["reason"] = "no_enabled_family_budget"
    adapted["unlabeled_distribution_adaptation"] = adaptation_payload
    adapted["claim_boundary"] = (
        str(adapted.get("claim_boundary", ""))
        + " Unlabeled distribution adaptation uses target candidate scores only, not target labels."
    ).strip()
    return adapted


def _calibrate_temporal_non_noop_budget(probability_rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    if not probability_rows:
        return {"status": "skipped", "reason": "no_probability_rows"}
    max_tokens = max(1, int(config.get("non_noop_budget_max_tokens_per_row", config.get("max_predicted_non_noop_tokens", 4))))
    max_button_fpr = float(config.get("non_noop_budget_max_no_button_fpr", config.get("calibration_max_no_button_fpr", 0.10)))
    beta = float(config.get("non_noop_budget_beta", 2.0))
    scores = sorted({float(candidate.get("score", 0.0)) for row in probability_rows for candidate in row.get("candidates", [])})
    if not scores:
        return {"status": "skipped", "reason": "no_non_noop_candidates"}
    raw_candidates = [float(value) for value in config.get("non_noop_budget_threshold_candidates", [])] if isinstance(config.get("non_noop_budget_threshold_candidates"), list) else []
    raw_candidates.extend(scores)
    raw_candidates.extend(min(1.0, score + 1e-6) for score in scores)
    raw_candidates.append(1.1)  # explicit abstention candidate
    thresholds = sorted({max(0.0, min(1.1, float(value))) for value in raw_candidates})
    if len(thresholds) > int(config.get("non_noop_budget_max_threshold_candidates", 256)):
        max_candidates = max(2, int(config.get("non_noop_budget_max_threshold_candidates", 256)))
        thresholds = [thresholds[min(len(thresholds) - 1, int(round(idx * (len(thresholds) - 1) / (max_candidates - 1))))] for idx in range(max_candidates)]
        thresholds = sorted(set(thresholds + [1.1]))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        acc = _PaperMetricAccumulator(empty_bins_as_correct=False)
        predicted_non_noop = 0
        for item in probability_rows:
            row = item["row"]
            fdm1_tokens = _tokens_from_non_noop_candidates(item.get("candidates", []), threshold=threshold, max_tokens=max_tokens)
            predicted_non_noop += len(fdm1_tokens)
            width, height = _screen_size(row)
            pred_tokens = d2e_metric_tokens_from_fdm1_tokens(fdm1_tokens, screen_width=width, screen_height=height) or [FDM1_ACTION_NOOP]
            acc.update(pred_tokens, item.get("ground_truth_tokens", []))
        metrics = acc.metrics()
        strict_button = metrics["strict_local"]["mouse_button"]
        paper_button = metrics["paper_compatible"]["mouse_button"]
        paper_key = metrics["paper_compatible"]["keyboard"]
        strict_key = metrics["strict_local"]["keyboard"]
        fpr = strict_button.get("no_button_false_positive_rate")
        precision = float(strict_button.get("precision") or 0.0)
        recall = float(strict_button.get("recall") or 0.0)
        if precision + recall > 0:
            beta2 = beta * beta
            button_fbeta = (1.0 + beta2) * precision * recall / max(1e-12, beta2 * precision + recall)
        else:
            button_fbeta = 0.0
        fpr_penalty = 0.0 if fpr is None or fpr <= max_button_fpr else (float(fpr) - max_button_fpr) * 10.0
        score = (
            float(paper_key.get("key_accuracy") or 0.0)
            + 0.25 * float(strict_key.get("accuracy") or 0.0)
            + button_fbeta
            + 0.25 * float(paper_button.get("button_accuracy") or 0.0)
            - fpr_penalty
        )
        rows.append(
            {
                "threshold": threshold,
                "score": score,
                "keyboard_key_accuracy": paper_key.get("key_accuracy"),
                "keyboard_strict_accuracy": strict_key.get("accuracy"),
                "mouse_button_accuracy": paper_button.get("button_accuracy"),
                "mouse_button_f1": strict_button.get("f1"),
                "mouse_button_fbeta": button_fbeta,
                "no_button_false_positive_rate": fpr,
                "predicted_non_noop_tokens": predicted_non_noop,
            }
        )
    selected = max(rows, key=lambda row: (float(row["score"]), int(row["predicted_non_noop_tokens"]), -float(row["threshold"])))
    return {
        "schema": "temporal_non_noop_budget_calibration.v1",
        "status": "pass",
        "rows": len(probability_rows),
        "candidate_count": len(rows),
        "selected_threshold": float(selected["threshold"]),
        "selected_row": selected,
        "max_tokens_per_row": max_tokens,
        "max_no_button_fpr": max_button_fpr,
        "sweep_preview": rows[:10] + ([] if len(rows) <= 20 else [{"omitted_rows": len(rows) - 20}]) + rows[-10:],
        "claim_boundary": "Calibration uses held-out training rows only; target labels remain evaluation/diagnostic only.",
    }


def _temporal_action_loss(torch: Any, logits: Any, target_ids: Any, loss_mask: Any, class_weights: Any, config: dict[str, Any]) -> Any:
    if not bool(loss_mask.any()):
        return torch.tensor(0.0, device=logits.device)
    selected_logits = logits[loss_mask]
    selected_targets = target_ids[loss_mask]
    loss_type = str(config.get("token_loss_type", "cross_entropy")).lower()
    if loss_type in {"focal", "focal_cross_entropy", "weighted_focal"}:
        ce = torch.nn.functional.cross_entropy(selected_logits, selected_targets, weight=class_weights, reduction="none")
        probs = torch.softmax(selected_logits, dim=-1)
        pt = probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        gamma = float(config.get("token_focal_gamma", config.get("focal_gamma", 2.0)))
        return (((1.0 - pt) ** gamma) * ce).mean()
    return torch.nn.functional.cross_entropy(selected_logits, selected_targets, weight=class_weights)


def train_temporal_masked_diffusion_idm(config: dict[str, Any]) -> dict[str, Any]:
    if not torch_available():
        raise RuntimeError("torch unavailable; run `uv sync --extra train` or use the MLXP training image")
    torch = require_torch()
    start = time.time()
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_temporal_masked_diffusion_d2e"))
    train_paths = _expand_paths(config.get("train_records")) + _expand_paths(config.get("train_record_paths"))
    target_paths = _expand_paths(config.get("target_records")) + _expand_paths(config.get("target_record_paths"))
    train_rows = list(_iter_jsonl(train_paths, max_rows=int(config["max_train_rows"]) if config.get("max_train_rows") is not None else None))
    target_rows = list(_iter_jsonl(target_paths, max_rows=int(config["max_target_rows"]) if config.get("max_target_rows") is not None else None))
    if not train_rows:
        raise ValueError("no train rows found for temporal masked-diffusion IDM")
    if not target_rows:
        raise ValueError("no target rows found for temporal masked-diffusion IDM")
    calibration_rows: list[dict[str, Any]] = []
    fit_rows = train_rows
    if bool(config.get("calibrate_non_noop_budget", config.get("non_noop_budgeted_unmasking", False))) and len(train_rows) >= 10:
        calibration_fraction = float(config.get("temporal_calibration_fraction", config.get("factorized_calibration_fraction", 0.0)) or 0.0)
        calibration_max_rows = int(config.get("temporal_calibration_max_rows", config.get("factorized_calibration_max_rows", 2000)))
        if calibration_fraction > 0.0:
            calibration_count = min(calibration_max_rows, max(1, int(len(train_rows) * calibration_fraction)))
            calibration_rows = train_rows[-calibration_count:]
            fit_rows = train_rows[:-calibration_count] or train_rows
    offsets = _temporal_offsets(config)
    max_slots = int(config.get("max_action_tokens_per_bin", config.get("max_slots", 16)))
    feature_dim = _configured_video_feature_dim(config)
    config = {**config, "video_feature_dim": feature_dim}
    preserve_pad_slots = bool(config.get("preserve_pad_action_slots", config.get("pad_action_slots_as_pad", False)))
    action_mouse_tokenization = _action_mouse_token_mode(config)
    config["action_mouse_tokenization"] = action_mouse_tokenization
    vocab = _build_vocab_for_config(
        train_rows,
        max_slots=max_slots,
        config=config,
        min_count=int(config.get("vocab_min_count", 1)),
        preserve_pad_slots=preserve_pad_slots,
    )
    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    feature_source = str(config.get("video_feature_source", "json")).lower()
    if feature_source in {"video_idm_cache", "raw_video_cache"}:
        train_features = _precompute_video_cache_features(
            train_paths,
            split_name="train",
            config=config,
            max_rows=len(train_rows),
        )
        fit_features = train_features[: len(fit_rows)]
        calibration_features = train_features[len(fit_rows) : len(fit_rows) + len(calibration_rows)] if calibration_rows else []
        target_features = _precompute_video_cache_features(
            target_paths,
            split_name="target",
            config=config,
            max_rows=len(target_rows),
        )
    else:
        fit_features = _precompute_features(fit_rows, config=config)
        calibration_features = _precompute_features(calibration_rows, config=config) if calibration_rows else []
        target_features = _precompute_features(target_rows, config=config)
    fit_features = _maybe_tensorize_features(torch, fit_features, config=config, split_name="fit")
    calibration_features = _maybe_tensorize_features(torch, calibration_features, config=config, split_name="calibration")
    target_features = _maybe_tensorize_features(torch, target_features, config=config, split_name="target")
    fit_target_ids = _precompute_target_ids_for_config(
        fit_rows,
        max_slots=max_slots,
        token_to_index=token_to_index,
        config=config,
        preserve_pad_slots=preserve_pad_slots,
    )
    dataset = _TemporalMaskedDiffusionDataset(features=fit_features, target_ids=fit_target_ids, config={**config, "max_slots": max_slots}, vocab=vocab)
    dataloader_workers = max(0, int(config.get("dataloader_num_workers", config.get("num_workers", 0)) or 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config.get("batch_size", 64)),
        "shuffle": True,
        "num_workers": dataloader_workers,
        "pin_memory": bool(config.get("dataloader_pin_memory", dataloader_workers > 0 and torch.cuda.is_available())),
    }
    if dataloader_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("dataloader_persistent_workers", True))
        prefetch_factor = config.get("dataloader_prefetch_factor")
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("force_cpu") else "cpu")
    model = _build_temporal_model(torch, video_dim=feature_dim, vocab_size=len(vocab), max_slots=max_slots, offsets=offsets, config=config, vocab=vocab).to(device)
    video_pretrain_history = _pretrain_video_encoder(model, torch, loader, config=config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    class_weights = _class_weights(torch, vocab, config, device=device)
    video_reconstruction_aux_weight = float(config.get("video_reconstruction_aux_weight", 0.0) or 0.0)
    event_auxiliary = bool(config.get("temporal_event_auxiliary", config.get("event_auxiliary", False)))
    key_event_aux_weight = float(config.get("key_event_aux_weight", config.get("event_aux_weight", 0.0)) or 0.0)
    button_event_aux_weight = float(config.get("button_event_aux_weight", config.get("event_aux_weight", 0.0)) or 0.0)
    button_class_auxiliary = bool(config.get("temporal_button_class_auxiliary", False))
    button_class_aux_weight = float(config.get("button_class_aux_weight", 0.0) or 0.0)
    button_vocab = _button_class_vocab(vocab)
    key_vocab = _key_class_vocab(vocab)
    mouse_move_vocab = _mouse_move_class_vocab(vocab)
    mouse_dx_vocab = _mouse_axis_class_vocab(vocab, "x")
    mouse_dy_vocab = _mouse_axis_class_vocab(vocab, "y")
    key_class_auxiliary = bool(config.get("temporal_key_class_auxiliary", False))
    key_class_aux_weight = float(config.get("key_class_aux_weight", 0.0) or 0.0)
    mouse_axis_class_auxiliary = bool(config.get("temporal_mouse_axis_class_auxiliary", False))
    mouse_axis_class_aux_weight = float(config.get("mouse_axis_class_aux_weight", 0.0) or 0.0)
    key_token_presence_auxiliary = bool(config.get("temporal_key_token_presence_auxiliary", False))
    key_token_presence_aux_weight = float(config.get("key_token_presence_aux_weight", 0.0) or 0.0)
    key_token_presence_rank_weight = float(config.get("key_token_presence_rank_weight", 0.0) or 0.0)
    button_token_presence_auxiliary = bool(config.get("temporal_button_token_presence_auxiliary", False))
    button_token_presence_aux_weight = float(config.get("button_token_presence_aux_weight", 0.0) or 0.0)
    button_token_presence_rank_weight = float(config.get("button_token_presence_rank_weight", 0.0) or 0.0)
    mouse_move_token_presence_auxiliary = bool(config.get("temporal_mouse_move_token_presence_auxiliary", False))
    mouse_move_token_presence_aux_weight = float(config.get("mouse_move_token_presence_aux_weight", 0.0) or 0.0)
    mouse_move_token_presence_rank_weight = float(config.get("mouse_move_token_presence_rank_weight", 0.0) or 0.0)
    token_presence_auxiliary = bool(config.get("temporal_token_presence_auxiliary", config.get("token_presence_auxiliary", False)))
    token_presence_aux_weight = float(config.get("token_presence_aux_weight", 0.0) or 0.0)
    token_presence_pos_weights = _token_presence_pos_weights(torch, vocab, config, device=device)
    event_offset_mask = _temporal_loss_offset_mask(torch, offsets, config, device=device)
    history: list[dict[str, Any]] = []
    progress_every = max(1, int(config.get("rank_progress_every_batches", 50)))
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        total_loss = 0.0
        total_action = 0.0
        total_video = 0.0
        total_key_event = 0.0
        total_button_event = 0.0
        total_button_class = 0.0
        total_key_class = 0.0
        total_mouse_axis_class = 0.0
        total_key_token_presence = 0.0
        total_key_token_presence_rank = 0.0
        total_button_token_presence = 0.0
        total_button_token_presence_rank = 0.0
        total_mouse_move_token_presence = 0.0
        total_mouse_move_token_presence_rank = 0.0
        total_token_presence = 0.0
        total_targets = 0
        total_examples = 0
        for batch_index, (features, corrupted_ids, target_ids, loss_mask) in enumerate(loader, 1):
            features = features.to(device=device, dtype=torch.float32)
            corrupted_ids = corrupted_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)
            if (
                event_auxiliary
                or button_class_auxiliary
                or key_class_auxiliary
                or mouse_axis_class_auxiliary
                or key_token_presence_auxiliary
                or button_token_presence_auxiliary
                or mouse_move_token_presence_auxiliary
                or token_presence_auxiliary
            ) and hasattr(model, "forward_with_aux"):
                payload = model.forward_with_aux(features, corrupted_ids)
                logits = payload["action_logits"]
            else:
                payload = {}
                logits = model(features, corrupted_ids)
            action_loss = _temporal_action_loss(torch, logits, target_ids, loss_mask, class_weights, config)
            video_loss = _masked_video_reconstruction_loss(model, torch, features, config=config) if video_reconstruction_aux_weight > 0.0 else torch.tensor(0.0, device=device)
            key_event_loss = torch.tensor(0.0, device=device)
            button_event_loss = torch.tensor(0.0, device=device)
            button_class_loss = torch.tensor(0.0, device=device)
            key_class_loss = torch.tensor(0.0, device=device)
            mouse_axis_class_loss = torch.tensor(0.0, device=device)
            key_token_presence_loss = torch.tensor(0.0, device=device)
            key_token_presence_rank_loss = torch.tensor(0.0, device=device)
            button_token_presence_loss = torch.tensor(0.0, device=device)
            button_token_presence_rank_loss = torch.tensor(0.0, device=device)
            mouse_move_token_presence_loss = torch.tensor(0.0, device=device)
            mouse_move_token_presence_rank_loss = torch.tensor(0.0, device=device)
            token_presence_loss = torch.tensor(0.0, device=device)
            if event_auxiliary and (key_event_aux_weight > 0.0 or button_event_aux_weight > 0.0):
                key_targets = _temporal_event_targets(torch, target_ids, vocab, ("KEY_",))
                button_targets = _temporal_event_targets(torch, target_ids, vocab, ("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))
                key_event_loss = _event_auxiliary_bce_loss(
                    torch,
                    payload.get("key_event_logits"),
                    key_targets,
                    event_offset_mask,
                    pos_weight=float(config.get("key_event_pos_weight", 8.0)),
                )
                button_event_loss = _event_auxiliary_bce_loss(
                    torch,
                    payload.get("button_event_logits"),
                    button_targets,
                    event_offset_mask,
                    pos_weight=float(config.get("button_event_pos_weight", 16.0)),
                )
            if button_class_auxiliary and button_class_aux_weight > 0.0 and button_vocab:
                button_class_targets = _temporal_button_class_targets(torch, target_ids, vocab, button_vocab)
                button_class_loss = _button_class_auxiliary_loss(
                    torch,
                    payload.get("button_class_logits"),
                    button_class_targets,
                    event_offset_mask,
                    config,
                )
            if key_class_auxiliary and key_class_aux_weight > 0.0 and key_vocab:
                key_class_targets = _temporal_family_class_targets(torch, target_ids, vocab, key_vocab)
                key_class_loss = _family_class_auxiliary_loss(
                    torch,
                    payload.get("key_class_logits"),
                    key_class_targets,
                    event_offset_mask,
                    no_family_weight=float(config.get("key_class_no_key_weight", 0.05)),
                    family_weight=float(config.get("key_class_key_weight", config.get("key_event_pos_weight", 8.0))),
                    focal_gamma=float(config.get("key_class_focal_gamma", 0.0) or 0.0),
                )
            if mouse_axis_class_auxiliary and mouse_axis_class_aux_weight > 0.0 and mouse_dx_vocab and mouse_dy_vocab:
                mouse_dx_targets = _temporal_family_class_targets(torch, target_ids, vocab, mouse_dx_vocab)
                mouse_dy_targets = _temporal_family_class_targets(torch, target_ids, vocab, mouse_dy_vocab)
                mouse_dx_loss = _family_class_auxiliary_loss(
                    torch,
                    payload.get("mouse_dx_class_logits"),
                    mouse_dx_targets,
                    event_offset_mask,
                    no_family_weight=float(config.get("mouse_axis_class_no_axis_weight", 0.05)),
                    family_weight=float(config.get("mouse_axis_class_axis_weight", config.get("token_presence_mouse_move_pos_weight", 2.0))),
                    focal_gamma=float(config.get("mouse_axis_class_focal_gamma", 0.0) or 0.0),
                )
                mouse_dy_loss = _family_class_auxiliary_loss(
                    torch,
                    payload.get("mouse_dy_class_logits"),
                    mouse_dy_targets,
                    event_offset_mask,
                    no_family_weight=float(config.get("mouse_axis_class_no_axis_weight", 0.05)),
                    family_weight=float(config.get("mouse_axis_class_axis_weight", config.get("token_presence_mouse_move_pos_weight", 2.0))),
                    focal_gamma=float(config.get("mouse_axis_class_focal_gamma", 0.0) or 0.0),
                )
                mouse_axis_class_loss = 0.5 * (mouse_dx_loss + mouse_dy_loss)
            if key_token_presence_auxiliary and key_token_presence_aux_weight > 0.0 and key_vocab:
                key_presence_targets = _temporal_family_token_presence_targets(torch, target_ids, vocab, key_vocab)
                key_token_presence_loss = _family_token_presence_bce_loss(
                    torch,
                    payload.get("key_token_presence_logits"),
                    key_presence_targets,
                    event_offset_mask,
                    pos_weight=float(config.get("key_token_presence_pos_weight", config.get("key_event_pos_weight", 8.0))),
                    negative_weight=float(config.get("key_token_presence_negative_weight", 0.05)),
                )
                if key_token_presence_rank_weight > 0.0:
                    key_token_presence_rank_loss = _family_token_presence_rank_loss(
                        torch,
                        payload.get("key_token_presence_logits"),
                        key_presence_targets,
                        event_offset_mask,
                        margin=float(config.get("key_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
                        top_negatives=int(config.get("key_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
                    )
            if button_token_presence_auxiliary and button_token_presence_aux_weight > 0.0 and button_vocab:
                button_presence_targets = _temporal_family_token_presence_targets(torch, target_ids, vocab, button_vocab)
                button_token_presence_loss = _family_token_presence_bce_loss(
                    torch,
                    payload.get("button_token_presence_logits"),
                    button_presence_targets,
                    event_offset_mask,
                    pos_weight=float(config.get("button_token_presence_pos_weight", config.get("button_event_pos_weight", 16.0))),
                    negative_weight=float(config.get("button_token_presence_negative_weight", 0.05)),
                )
                if button_token_presence_rank_weight > 0.0:
                    button_token_presence_rank_loss = _family_token_presence_rank_loss(
                        torch,
                        payload.get("button_token_presence_logits"),
                        button_presence_targets,
                        event_offset_mask,
                        margin=float(config.get("button_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
                        top_negatives=int(config.get("button_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
                    )
            if mouse_move_token_presence_auxiliary and mouse_move_token_presence_aux_weight > 0.0 and mouse_move_vocab:
                mouse_move_presence_targets = _temporal_family_token_presence_targets(torch, target_ids, vocab, mouse_move_vocab)
                mouse_move_token_presence_loss = _family_token_presence_bce_loss(
                    torch,
                    payload.get("mouse_move_token_presence_logits"),
                    mouse_move_presence_targets,
                    event_offset_mask,
                    pos_weight=float(config.get("mouse_move_token_presence_pos_weight", config.get("token_presence_mouse_move_pos_weight", 2.0))),
                    negative_weight=float(config.get("mouse_move_token_presence_negative_weight", 0.05)),
                )
                if mouse_move_token_presence_rank_weight > 0.0:
                    mouse_move_token_presence_rank_loss = _family_token_presence_rank_loss(
                        torch,
                        payload.get("mouse_move_token_presence_logits"),
                        mouse_move_presence_targets,
                        event_offset_mask,
                        margin=float(config.get("mouse_move_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
                        top_negatives=int(config.get("mouse_move_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
                    )
            if token_presence_auxiliary and token_presence_aux_weight > 0.0:
                token_presence_targets = _token_presence_targets(
                    torch,
                    target_ids,
                    vocab,
                    include_noop=bool(config.get("token_presence_include_noop", False)),
                )
                token_presence_loss = _token_presence_bce_loss(
                    torch,
                    payload.get("token_presence_logits"),
                    token_presence_targets,
                    event_offset_mask,
                    token_presence_pos_weights,
                )
            loss = (
                action_loss
                + video_reconstruction_aux_weight * video_loss
                + key_event_aux_weight * key_event_loss
                + button_event_aux_weight * button_event_loss
                + button_class_aux_weight * button_class_loss
                + key_class_aux_weight * key_class_loss
                + mouse_axis_class_aux_weight * mouse_axis_class_loss
                + key_token_presence_aux_weight * key_token_presence_loss
                + key_token_presence_rank_weight * key_token_presence_rank_loss
                + button_token_presence_aux_weight * button_token_presence_loss
                + button_token_presence_rank_weight * button_token_presence_rank_loss
                + mouse_move_token_presence_aux_weight * mouse_move_token_presence_loss
                + mouse_move_token_presence_rank_weight * mouse_move_token_presence_rank_loss
                + token_presence_aux_weight * token_presence_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            count = int(loss_mask.sum().detach().cpu())
            batch = int(features.shape[0])
            total_loss += float(loss.detach().cpu()) * max(1, count)
            total_action += float(action_loss.detach().cpu()) * max(1, count)
            total_video += float(video_loss.detach().cpu()) * batch
            total_key_event += float(key_event_loss.detach().cpu()) * batch
            total_button_event += float(button_event_loss.detach().cpu()) * batch
            total_button_class += float(button_class_loss.detach().cpu()) * batch
            total_key_class += float(key_class_loss.detach().cpu()) * batch
            total_mouse_axis_class += float(mouse_axis_class_loss.detach().cpu()) * batch
            total_key_token_presence += float(key_token_presence_loss.detach().cpu()) * batch
            total_key_token_presence_rank += float(key_token_presence_rank_loss.detach().cpu()) * batch
            total_button_token_presence += float(button_token_presence_loss.detach().cpu()) * batch
            total_button_token_presence_rank += float(button_token_presence_rank_loss.detach().cpu()) * batch
            total_mouse_move_token_presence += float(mouse_move_token_presence_loss.detach().cpu()) * batch
            total_mouse_move_token_presence_rank += float(mouse_move_token_presence_rank_loss.detach().cpu()) * batch
            total_token_presence += float(token_presence_loss.detach().cpu()) * batch
            total_targets += count
            total_examples += batch
            if batch_index == 1 or batch_index % progress_every == 0 or (total_batches and batch_index == total_batches):
                _write_rank_progress(
                    config,
                    output_dir=output_dir,
                    payload={
                        "phase": "action_denoising",
                        "epoch": epoch + 1,
                        "epochs": int(config.get("epochs", 1)),
                        "batch": batch_index,
                        "batches": total_batches,
                        "examples": total_examples,
                        "masked_targets": total_targets,
                        "loss": total_loss / max(1, total_targets),
                        "action_loss": total_action / max(1, total_targets),
                        "video_reconstruction_loss": total_video / max(1, total_examples),
                        "key_event_loss": total_key_event / max(1, total_examples),
                        "button_event_loss": total_button_event / max(1, total_examples),
                        "button_class_loss": total_button_class / max(1, total_examples),
                        "key_class_loss": total_key_class / max(1, total_examples),
                        "mouse_axis_class_loss": total_mouse_axis_class / max(1, total_examples),
                        "key_token_presence_loss": total_key_token_presence / max(1, total_examples),
                        "key_token_presence_rank_loss": total_key_token_presence_rank / max(1, total_examples),
                        "button_token_presence_loss": total_button_token_presence / max(1, total_examples),
                        "button_token_presence_rank_loss": total_button_token_presence_rank / max(1, total_examples),
                        "mouse_move_token_presence_loss": total_mouse_move_token_presence / max(1, total_examples),
                        "mouse_move_token_presence_rank_loss": total_mouse_move_token_presence_rank / max(1, total_examples),
                        "token_presence_loss": total_token_presence / max(1, total_examples),
                    },
                )
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(1, total_targets),
            "action_loss": total_action / max(1, total_targets),
            "video_reconstruction_loss": total_video / max(1, len(dataset)),
            "key_event_loss": total_key_event / max(1, total_examples),
            "button_event_loss": total_button_event / max(1, total_examples),
            "button_class_loss": total_button_class / max(1, total_examples),
            "key_class_loss": total_key_class / max(1, total_examples),
            "mouse_axis_class_loss": total_mouse_axis_class / max(1, total_examples),
            "key_token_presence_loss": total_key_token_presence / max(1, total_examples),
            "key_token_presence_rank_loss": total_key_token_presence_rank / max(1, total_examples),
            "button_token_presence_loss": total_button_token_presence / max(1, total_examples),
            "button_token_presence_rank_loss": total_button_token_presence_rank / max(1, total_examples),
            "mouse_move_token_presence_loss": total_mouse_move_token_presence / max(1, total_examples),
            "mouse_move_token_presence_rank_loss": total_mouse_move_token_presence_rank / max(1, total_examples),
            "token_presence_loss": total_token_presence / max(1, total_examples),
            "masked_targets": total_targets,
        })
    train_history_path = Path(output_dir) / "train_history.json"
    write_json(
        train_history_path,
        {
            "schema": "temporal_masked_diffusion_idm_train_history.v1",
            "model_name": str(config.get("model_name", "temporal_masked_diffusion_idm")),
            "history": history,
        },
    )
    checkpoint_path = Path(output_dir) / "checkpoint.pt"
    torch.save({
        "schema":"temporal_masked_diffusion_idm_checkpoint.v1",
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config,
        "max_slots": max_slots,
        "feature_dim": feature_dim,
        "temporal_offsets": offsets,
    }, checkpoint_path)
    retrieval_index = _build_temporal_retrieval_prior_index(
        model,
        torch,
        fit_rows,
        fit_features,
        config={**config, "max_slots": max_slots},
        vocab=vocab,
        device=device,
    )
    retrieval_summary = {key: value for key, value in retrieval_index.items() if key not in {"embeddings", "tokens"}}
    candidate_token_prior_weights, candidate_token_prior_summary = _candidate_token_prior_weights(
        fit_rows,
        vocab=vocab,
        max_slots=max_slots,
        preserve_pad_slots=preserve_pad_slots,
        config=config,
    )
    non_noop_budget = {"status": "skipped", "reason": "disabled"}
    family_non_noop_budget = {"status": "skipped", "reason": "disabled"}
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
            config={**config, "max_slots": max_slots},
            vocab=vocab,
            device=device,
            retrieval_index=retrieval_index,
            token_prior_weights=candidate_token_prior_weights,
        )
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
            config={**config, "max_slots": max_slots},
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
                config={**config, "max_slots": max_slots},
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
        model_name=str(config.get("model_name", "temporal_masked_diffusion_idm")),
        max_rows=len(target_rows),
    )
    summary = {
        "schema":"temporal_masked_diffusion_idm_train_summary.v1",
        "status":"pass",
        "model_name":str(config.get("model_name", "temporal_masked_diffusion_idm")),
        "recipe_alignment":"public FDM-1-shaped noncausal masked-diffusion IDM over temporal action-token sequences conditioned on all D2E frame-window tokens in the local window.",
        "train_rows":len(train_rows),
        "fit_rows":len(fit_rows),
        "calibration_rows":len(calibration_rows),
        "target_rows":len(target_rows),
        "vocab_size":len(vocab),
        "max_slots":max_slots,
        "action_mouse_tokenization":action_mouse_tokenization,
        "preserve_pad_action_slots":preserve_pad_slots,
        "temporal_offsets":offsets,
        "temporal_window":len(offsets),
        "video_feature_source":feature_source,
        "video_feature_dim":feature_dim,
        "precompute_features_as_tensor":bool(config.get("precompute_features_as_tensor", config.get("tensorize_precomputed_features", False))),
        "precompute_feature_tensor_dtype":str(config.get("precompute_feature_tensor_dtype", "float16")),
        "dataloader_num_workers":dataloader_workers,
        "video_encoder_arch":str(config.get("video_encoder_arch", "flat_mlp")),
        "video_tokens_per_offset":int(getattr(model, "video_tokens_per_offset", 1) or 1),
        "raw_video_frame_offsets":_raw_video_frame_offsets(config)
        if feature_source in {"raw_frames", "raw_video_frames", "frame_provider", "video_idm_cache", "raw_video_cache"}
        else None,
        "raw_video_image_size":int(config.get("raw_video_image_size", config.get("video_image_size", 96)))
        if feature_source in {"raw_frames", "raw_video_frames", "frame_provider", "video_idm_cache", "raw_video_cache"}
        else None,
        "video_cache_dir":str(config.get("video_cache_dir")) if feature_source in {"video_idm_cache", "raw_video_cache"} else None,
        "video_cache_stats_path":str(config.get("video_cache_stats_path", config.get("stats_path"))) if feature_source in {"video_idm_cache", "raw_video_cache"} else None,
        "video_encoder_pretrain_history":video_pretrain_history,
        "history":history,
        "non_noop_budget":non_noop_budget,
        "family_non_noop_budget":family_non_noop_budget,
        "candidate_family_diagnostics":candidate_family_diagnostics,
        "retrieval_action_prior":retrieval_summary,
        "candidate_token_prior":candidate_token_prior_summary,
        "loss_weights":{
            "noop_loss_weight":float(config.get("noop_loss_weight", 1.0)),
            "pad_loss_weight":float(config.get("pad_loss_weight", 0.0)),
            "preserve_pad_action_slots":preserve_pad_slots,
            "action_mouse_tokenization":action_mouse_tokenization,
            "action_loss_weight":float(config.get("action_loss_weight", 1.0)),
            "keyboard_loss_weight":float(config.get("keyboard_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_button_loss_weight":float(config.get("mouse_button_loss_weight", config.get("action_loss_weight", 1.0))),
            "mouse_move_loss_weight":float(config.get("mouse_move_loss_weight", config.get("action_loss_weight", 1.0))),
            "video_reconstruction_aux_weight":video_reconstruction_aux_weight,
            "temporal_event_auxiliary":event_auxiliary,
            "key_event_aux_weight":key_event_aux_weight,
            "button_event_aux_weight":button_event_aux_weight,
            "temporal_button_class_auxiliary":button_class_auxiliary,
            "button_class_aux_weight":button_class_aux_weight,
            "button_class_vocab_size":len(button_vocab),
            "button_class_no_button_weight":float(config.get("button_class_no_button_weight", 0.05)),
            "button_class_button_weight":float(config.get("button_class_button_weight", config.get("button_event_pos_weight", 16.0))),
            "button_class_focal_gamma":float(config.get("button_class_focal_gamma", 0.0) or 0.0),
            "button_class_candidate_score_blend":float(config.get("button_class_candidate_score_blend", 0.0)),
            "button_class_no_button_gate_power":float(config.get("button_class_no_button_gate_power", 0.0) or 0.0),
            "button_class_no_button_gate_floor":float(config.get("button_class_no_button_gate_floor", 0.0) or 0.0),
            "temporal_key_class_auxiliary":key_class_auxiliary,
            "key_class_aux_weight":key_class_aux_weight,
            "key_class_vocab_size":len(key_vocab),
            "key_class_no_key_weight":float(config.get("key_class_no_key_weight", 0.05)),
            "key_class_key_weight":float(config.get("key_class_key_weight", config.get("key_event_pos_weight", 8.0))),
            "key_class_focal_gamma":float(config.get("key_class_focal_gamma", 0.0) or 0.0),
            "key_class_candidate_score_blend":float(config.get("key_class_candidate_score_blend", 0.0)),
            "direct_auxiliary_key_class_blend":float(config.get("direct_auxiliary_key_class_blend", 0.0) or 0.0),
            "temporal_key_token_presence_auxiliary":key_token_presence_auxiliary,
            "key_token_presence_aux_weight":key_token_presence_aux_weight,
            "key_token_presence_rank_weight":key_token_presence_rank_weight,
            "key_token_presence_rank_margin":float(config.get("key_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
            "key_token_presence_rank_top_negatives":int(config.get("key_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
            "key_token_presence_vocab_size":len(key_vocab),
            "key_token_presence_pos_weight":float(config.get("key_token_presence_pos_weight", config.get("key_event_pos_weight", 8.0))),
            "key_token_presence_negative_weight":float(config.get("key_token_presence_negative_weight", 0.05)),
            "key_token_presence_candidate_score_blend":float(config.get("key_token_presence_candidate_score_blend", 0.0)),
            "temporal_button_token_presence_auxiliary":button_token_presence_auxiliary,
            "button_token_presence_aux_weight":button_token_presence_aux_weight,
            "button_token_presence_rank_weight":button_token_presence_rank_weight,
            "button_token_presence_rank_margin":float(config.get("button_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
            "button_token_presence_rank_top_negatives":int(config.get("button_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
            "button_token_presence_vocab_size":len(button_vocab),
            "button_token_presence_pos_weight":float(config.get("button_token_presence_pos_weight", config.get("button_event_pos_weight", 16.0))),
            "button_token_presence_negative_weight":float(config.get("button_token_presence_negative_weight", 0.05)),
            "button_token_presence_candidate_score_blend":float(config.get("button_token_presence_candidate_score_blend", 0.0)),
            "temporal_mouse_move_token_presence_auxiliary":mouse_move_token_presence_auxiliary,
            "mouse_move_token_presence_aux_weight":mouse_move_token_presence_aux_weight,
            "mouse_move_token_presence_rank_weight":mouse_move_token_presence_rank_weight,
            "mouse_move_token_presence_rank_margin":float(config.get("mouse_move_token_presence_rank_margin", config.get("token_presence_rank_margin", 1.0))),
            "mouse_move_token_presence_rank_top_negatives":int(config.get("mouse_move_token_presence_rank_top_negatives", config.get("token_presence_rank_top_negatives", 1))),
            "mouse_move_token_presence_vocab_size":len(mouse_move_vocab),
            "mouse_move_token_presence_pos_weight":float(config.get("mouse_move_token_presence_pos_weight", config.get("token_presence_mouse_move_pos_weight", 2.0))),
            "mouse_move_token_presence_negative_weight":float(config.get("mouse_move_token_presence_negative_weight", 0.05)),
            "mouse_move_token_presence_candidate_score_blend":float(config.get("mouse_move_token_presence_candidate_score_blend", 0.0)),
            "temporal_mouse_axis_class_auxiliary":mouse_axis_class_auxiliary,
            "mouse_axis_class_aux_weight":mouse_axis_class_aux_weight,
            "mouse_dx_class_vocab_size":len(mouse_dx_vocab),
            "mouse_dy_class_vocab_size":len(mouse_dy_vocab),
            "mouse_axis_class_no_axis_weight":float(config.get("mouse_axis_class_no_axis_weight", 0.05)),
            "mouse_axis_class_axis_weight":float(config.get("mouse_axis_class_axis_weight", config.get("token_presence_mouse_move_pos_weight", 2.0))),
            "mouse_axis_class_focal_gamma":float(config.get("mouse_axis_class_focal_gamma", 0.0) or 0.0),
            "mouse_axis_class_candidate_score_blend":float(config.get("mouse_axis_class_candidate_score_blend", 0.0)),
            "direct_auxiliary_mouse_axis_class_blend":float(config.get("direct_auxiliary_mouse_axis_class_blend", 0.0) or 0.0),
            "mouse_move_axis_constrained_budget":bool(config.get("mouse_move_axis_constrained_budget", False)),
            "key_event_pos_weight":float(config.get("key_event_pos_weight", 8.0)),
            "button_event_pos_weight":float(config.get("button_event_pos_weight", 16.0)),
            "event_auxiliary_candidate_score_blend":float(config.get("event_auxiliary_candidate_score_blend", 0.5)),
            "event_auxiliary_candidate_gate_power":float(config.get("event_auxiliary_candidate_gate_power", 0.0) or 0.0),
            "event_auxiliary_candidate_gate_floor":float(config.get("event_auxiliary_candidate_gate_floor", 0.0) or 0.0),
            "event_auxiliary_candidate_gate_families":list(config.get("event_auxiliary_candidate_gate_families", ["keyboard", "mouse_button"]))
            if isinstance(config.get("event_auxiliary_candidate_gate_families", ["keyboard", "mouse_button"]), list)
            else ["keyboard", "mouse_button"],
            "temporal_token_presence_auxiliary":token_presence_auxiliary,
            "token_presence_aux_weight":token_presence_aux_weight,
            "token_presence_include_noop":bool(config.get("token_presence_include_noop", False)),
            "token_presence_candidate_score_blend":float(config.get("token_presence_candidate_score_blend", 0.0)),
            "token_presence_keyboard_pos_weight":float(config.get("token_presence_keyboard_pos_weight", config.get("key_event_pos_weight", 8.0))),
            "token_presence_mouse_button_pos_weight":float(config.get("token_presence_mouse_button_pos_weight", config.get("button_event_pos_weight", 16.0))),
            "token_presence_mouse_move_pos_weight":float(config.get("token_presence_mouse_move_pos_weight", 2.0)),
            "retrieval_action_prior_blend":float(config.get("retrieval_action_prior_blend", 0.35)),
            "candidate_token_prior_correction":bool(config.get("candidate_token_prior_correction", False)),
            "candidate_token_prior_strength":float(config.get("candidate_token_prior_strength", 0.5)),
            "candidate_token_prior_smoothing":float(config.get("candidate_token_prior_smoothing", 8.0)),
            "candidate_token_prior_min_weight":float(config.get("candidate_token_prior_min_weight", 0.25)),
            "candidate_token_prior_max_weight":float(config.get("candidate_token_prior_max_weight", 8.0)),
            "candidate_token_prior_unseen_weight":float(config.get("candidate_token_prior_unseen_weight", 1.0)),
            "candidate_token_prior_families":list(config.get("candidate_token_prior_families", ["keyboard", "mouse_button"]))
            if isinstance(config.get("candidate_token_prior_families", ["keyboard", "mouse_button"]), list)
            else ["keyboard", "mouse_button"],
            "non_noop_budget_candidates_per_row":int(config.get("non_noop_budget_candidates_per_row", max_slots * 8)),
            "non_noop_budget_min_candidates_per_family":int(
                config.get("non_noop_budget_min_candidates_per_family", config.get("candidate_min_candidates_per_family", 0)) or 0
            ),
            "candidate_diagnostics_target_max_rows":int(config.get("candidate_diagnostics_target_max_rows", 0) or 0),
            "direct_auxiliary_candidate_families":list(config.get("direct_auxiliary_candidate_families", []))
            if isinstance(config.get("direct_auxiliary_candidate_families", []), list)
            else [],
            "direct_auxiliary_candidate_min_score":float(config.get("direct_auxiliary_candidate_min_score", 0.0) or 0.0),
            "direct_auxiliary_button_class_blend":float(config.get("direct_auxiliary_button_class_blend", 0.5) or 0.0),
            "direct_auxiliary_button_no_button_gate_power":float(config.get("direct_auxiliary_button_no_button_gate_power", 0.0) or 0.0),
            "token_loss_type":str(config.get("token_loss_type", "cross_entropy")),
            "token_focal_gamma":float(config.get("token_focal_gamma", config.get("focal_gamma", 2.0))),
            "full_action_mask_probability":float(
                config.get(
                    "full_action_mask_probability",
                    config.get("all_action_mask_probability", config.get("all_mask_probability", 0.0)),
                )
                or 0.0
            ),
        },
        "device":str(device),
        "checkpoint_path":str(checkpoint_path),
        "train_history_path":str(train_history_path),
        "predictions_path":str(predictions_path),
        "metrics_path":str(metrics_path),
        "wall_clock_seconds":time.time()-start,
        "claim_boundary":"Temporal prefix trainer scaffold; not G005 completion evidence without full-corpus 4xH200 run, recipe-alignment audit, paper/G-IDM target win, and split statistics.",
    }
    summary_path = Path(config.get("summary_out", Path(output_dir) / "summary.json"))
    write_json(summary_path, summary)
    write_json(Path(output_dir) / "resolved_config.json", config)
    return summary
