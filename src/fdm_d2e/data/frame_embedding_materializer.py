from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.io_utils import ensure_dir, sha256_file, stable_hash_json, write_json
from fdm_d2e.training.neural_idm import record_features
from fdm_d2e.training.video_idm import _FramePairProvider

try:  # pragma: no cover - exercised on cluster images when present.
    import orjson  # type: ignore
except Exception:  # pragma: no cover - fallback is covered.
    orjson = None


_JSONL_BUFFER = 1024 * 1024
_JSONL_SEPARATORS = (",", ":")


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _dumps(row: dict[str, Any]) -> str:
    if orjson is not None:
        return orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=_JSONL_SEPARATORS)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", buffering=_JSONL_BUFFER) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield _loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc


def parse_offsets(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("frame offsets must not be empty")
        offsets = tuple(int(part) for part in parts)
    else:
        offsets = tuple(int(item) for item in value)
        if not offsets:
            raise ValueError("frame offsets must not be empty")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"frame offsets must be unique: {offsets}")
    return offsets


def _source_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _frame_stats_embedding(frame: bytes) -> list[float]:
    if not frame:
        return [0.0 for _ in range(12)]
    values = [float(byte) / 255.0 for byte in frame]
    count = float(len(values))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    std = math.sqrt(max(0.0, variance))
    minimum = min(values)
    maximum = max(values)
    energy = sum(value * value for value in values) / count
    # Lightweight spatial moments over square grayscale frames.  These are not a
    # competitive embedding; they make the materializer fully testable without
    # optional train/HF dependencies and serve as a deterministic smoke backend.
    side = int(math.sqrt(len(values)))
    if side * side == len(values) and side > 0:
        row_means = [sum(values[row * side : (row + 1) * side]) / side for row in range(side)]
        col_means = [sum(values[col::side]) / side for col in range(side)]
        top = sum(row_means[: max(1, side // 4)]) / max(1, len(row_means[: max(1, side // 4)]))
        bottom = sum(row_means[-max(1, side // 4) :]) / max(1, len(row_means[-max(1, side // 4) :]))
        left = sum(col_means[: max(1, side // 4)]) / max(1, len(col_means[: max(1, side // 4)]))
        right = sum(col_means[-max(1, side // 4) :]) / max(1, len(col_means[-max(1, side // 4) :]))
        center_values: list[float] = []
        lo = side // 4
        hi = side - lo
        for y in range(lo, hi):
            center_values.extend(values[y * side + lo : y * side + hi])
        center = sum(center_values) / max(1, len(center_values))
        vertical_delta = bottom - top
        horizontal_delta = right - left
        center_delta = center - mean
    else:
        top = bottom = left = right = center = mean
        vertical_delta = horizontal_delta = center_delta = 0.0
    return [
        mean,
        std,
        minimum,
        maximum,
        energy,
        top,
        bottom,
        left,
        right,
        center,
        vertical_delta,
        horizontal_delta + center_delta,
    ]


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def _round_values(values: Sequence[float], digits: int | None) -> list[float]:
    if digits is None or digits < 0:
        return [float(value) for value in values]
    return [round(float(value), int(digits)) for value in values]


def _embedding_deltas(embeddings: Sequence[Sequence[float]]) -> list[float]:
    if not embeddings:
        return []
    base = list(embeddings[0])
    out: list[float] = []
    for embedding in embeddings[1:]:
        out.extend(float(value) - float(base[idx]) for idx, value in enumerate(embedding))
    return out


def _metadata_fingerprint(metadata: dict[str, Any]) -> str:
    return stable_hash_json(metadata)


@dataclass(frozen=True)
class FrameEmbeddingMaterializerConfig:
    input_path: Path
    output_path: Path
    summary_out: Path
    backend: str = "dummy-stat"
    model_id: str = "facebook/dinov2-small"
    frame_offsets: tuple[int, ...] = (0, 2)
    frame_source: str = "video"
    image_size: int = 224
    frame_fps: int = 20
    missing_frame_policy: str = "zero"
    batch_size: int = 16
    device: str = "auto"
    embedding_pooling: str = "cls"
    hf_preprocess: str = "manual-imagenet"
    normalize_embeddings: bool = True
    include_embedding_deltas: bool = True
    include_summary_features: bool = True
    summary_feature_mode: str = "summary_compact_luma16_pair_shift_time_state_duration_prior_action"
    max_rows: int | None = None
    round_digits: int | None = 6
    trust_remote_code: bool = False
    path_remaps: tuple[tuple[str, str], ...] = ()
    progress_output: Path | None = None
    progress_rows: int = 50_000
    source_label: str = "g005_frozen_frame_embedding_materialization"


class _DummyStatEmbedder:
    backend = "dummy-stat"

    def __init__(self, *, normalize_embeddings: bool) -> None:
        self.normalize_embeddings = bool(normalize_embeddings)
        self.embedding_dim = 12

    def embed_frames(self, frames: Sequence[bytes]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for frame in frames:
            values = _frame_stats_embedding(frame)
            embeddings.append(_normalize(values) if self.normalize_embeddings else values)
        return embeddings


class _HfVisionEmbedder:
    backend = "hf-vision"

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        pooling: str,
        hf_preprocess: str,
        image_size: int,
        normalize_embeddings: bool,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover - depends on optional train extra.
            raise RuntimeError("hf-vision backend requires `uv sync --extra train` plus Pillow/transformers/torch") from exc
        self.torch = torch
        self.Image = Image
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pooling = pooling
        self.hf_preprocess = hf_preprocess.replace("_", "-").lower()
        self.image_size = int(image_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.processor = None
        if self.hf_preprocess == "auto":
            try:
                from transformers import AutoImageProcessor
            except Exception as exc:  # pragma: no cover - depends on optional processor deps.
                raise RuntimeError("hf_preprocess=auto requires AutoImageProcessor dependencies") from exc
            self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        elif self.hf_preprocess not in {"manual-imagenet", "manual"}:
            raise ValueError("hf_preprocess must be one of: manual-imagenet, auto")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim: int | None = None

    def _pil_images(self, frames: Sequence[bytes]) -> list[Any]:
        side = int(math.sqrt(len(frames[0]))) if frames else 0
        images: list[Any] = []
        for frame in frames:
            frame_side = int(math.sqrt(len(frame)))
            if frame_side * frame_side != len(frame):
                raise ValueError(f"expected square grayscale frame bytes, got {len(frame)} bytes")
            images.append(self.Image.frombytes("L", (frame_side, frame_side), bytes(frame)).convert("RGB"))
        if side <= 0:
            raise ValueError("no frames to embed")
        return images

    def _manual_pixel_values(self, images: Sequence[Any]) -> Any:
        torch = self.torch
        resample = getattr(getattr(self.Image, "Resampling", self.Image), "BICUBIC", 3)
        tensors = []
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        for image in images:
            resized = image.resize((self.image_size, self.image_size), resample=resample).convert("RGB")
            pixels = list(resized.getdata())
            tensor = torch.tensor(pixels, dtype=torch.float32).view(self.image_size, self.image_size, 3)
            tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0)

    def _select_embedding(self, outputs: Any) -> Any:
        torch = self.torch
        if self.pooling == "pooler":
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                return pooler
        if self.pooling == "image":
            image_embeds = getattr(outputs, "image_embeds", None)
            if image_embeds is not None:
                return image_embeds
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            if isinstance(outputs, (tuple, list)) and outputs:
                hidden = outputs[0]
            else:
                raise ValueError("HF vision model output lacks pooler_output/image_embeds/last_hidden_state")
        if self.pooling in {"pooler", "cls", "image"}:
            return hidden[:, 0, :]
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        raise ValueError("embedding_pooling must be one of: cls, mean, pooler, image")

    def embed_frames(self, frames: Sequence[bytes]) -> list[list[float]]:
        if not frames:
            return []
        torch = self.torch
        images = self._pil_images(frames)
        if self.processor is None:
            encoded = {"pixel_values": self._manual_pixel_values(images)}
        else:
            encoded = self.processor(images=images, return_tensors="pt")
        encoded = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            embedding = self._select_embedding(outputs).detach()
            if self.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
        embedding = embedding.cpu().float()
        if self.embedding_dim is None:
            self.embedding_dim = int(embedding.shape[-1])
        return [[float(value) for value in row] for row in embedding.tolist()]


class _TorchHubDinov2Embedder:
    backend = "dinov2-torchhub"

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        image_size: int,
        normalize_embeddings: bool,
    ) -> None:
        try:
            import torch
            from PIL import Image
        except Exception as exc:  # pragma: no cover - depends on cluster image.
            raise RuntimeError("dinov2-torchhub backend requires torch and Pillow") from exc
        self.torch = torch
        self.Image = Image
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.image_size = int(image_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        aliases = {
            "facebook/dinov2-small": "dinov2_vits14",
            "facebook/dinov2-base": "dinov2_vitb14",
            "facebook/dinov2-large": "dinov2_vitl14",
            "facebook/dinov2-giant": "dinov2_vitg14",
        }
        self.model_name = aliases.get(model_id, model_id)
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
        self.model.eval()
        self.model.to(self.device)
        self.embedding_dim: int | None = None

    def _pil_images(self, frames: Sequence[bytes]) -> list[Any]:
        images: list[Any] = []
        for frame in frames:
            frame_side = int(math.sqrt(len(frame)))
            if frame_side * frame_side != len(frame):
                raise ValueError(f"expected square grayscale frame bytes, got {len(frame)} bytes")
            images.append(self.Image.frombytes("L", (frame_side, frame_side), bytes(frame)).convert("RGB"))
        return images

    def _manual_pixel_values(self, images: Sequence[Any]) -> Any:
        torch = self.torch
        resample = getattr(getattr(self.Image, "Resampling", self.Image), "BICUBIC", 3)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        tensors = []
        for image in images:
            resized = image.resize((self.image_size, self.image_size), resample=resample).convert("RGB")
            pixels = list(resized.getdata())
            tensor = torch.tensor(pixels, dtype=torch.float32).view(self.image_size, self.image_size, 3)
            tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0)

    def _pixel_values_from_gray_bytes(self, frames: Sequence[bytes]) -> Any:
        torch = self.torch
        tensors = []
        side: int | None = None
        for frame in frames:
            frame_side = int(math.sqrt(len(frame)))
            if frame_side * frame_side != len(frame):
                raise ValueError(f"expected square grayscale frame bytes, got {len(frame)} bytes")
            side = frame_side
            try:
                tensor = torch.frombuffer(frame, dtype=torch.uint8)
            except Exception:  # pragma: no cover - older torch fallback.
                tensor = torch.tensor(list(frame), dtype=torch.uint8)
            tensors.append(tensor.float().view(1, frame_side, frame_side) / 255.0)
        if not tensors:
            raise ValueError("no frames to embed")
        batch = torch.stack(tensors, dim=0)
        if side != self.image_size:
            batch = torch.nn.functional.interpolate(batch, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        batch = batch.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return (batch - mean) / std

    def embed_frames(self, frames: Sequence[bytes]) -> list[list[float]]:
        if not frames:
            return []
        torch = self.torch
        pixel_values = self._pixel_values_from_gray_bytes(frames).to(self.device)
        with torch.no_grad():
            output = self.model(pixel_values).detach()
            if self.normalize_embeddings:
                output = torch.nn.functional.normalize(output, dim=-1)
        output = output.cpu().float()
        if self.embedding_dim is None:
            self.embedding_dim = int(output.shape[-1])
        return [[float(value) for value in row] for row in output.tolist()]


def _build_embedder(config: FrameEmbeddingMaterializerConfig) -> Any:
    backend = config.backend.replace("_", "-").lower()
    if backend == "dummy-stat":
        return _DummyStatEmbedder(normalize_embeddings=config.normalize_embeddings)
    if backend == "hf-vision":
        return _HfVisionEmbedder(
            model_id=config.model_id,
            device=config.device,
            pooling=config.embedding_pooling,
            hf_preprocess=config.hf_preprocess,
            image_size=config.image_size,
            normalize_embeddings=config.normalize_embeddings,
            trust_remote_code=config.trust_remote_code,
        )
    if backend == "dinov2-torchhub":
        return _TorchHubDinov2Embedder(
            model_id=config.model_id,
            device=config.device,
            image_size=config.image_size,
            normalize_embeddings=config.normalize_embeddings,
        )
    raise ValueError(f"unsupported frame embedding backend: {config.backend}")


def parse_path_remaps(values: Sequence[str] | None) -> tuple[tuple[str, str], ...]:
    remaps: list[tuple[str, str]] = []
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"path remap must be FROM=TO, got {raw!r}")
        source, target = raw.split("=", 1)
        source = source.rstrip("/")
        target = target.rstrip("/")
        if not source or not target:
            raise ValueError(f"path remap must have non-empty FROM and TO: {raw!r}")
        remaps.append((source, target))
    # Longest source prefix first makes nested remaps deterministic.
    return tuple(sorted(remaps, key=lambda item: len(item[0]), reverse=True))


def _row_for_frame_lookup(row: dict[str, Any], path_remaps: Sequence[tuple[str, str]]) -> dict[str, Any]:
    if not path_remaps:
        return row
    frame = row.get("frame")
    if not isinstance(frame, dict):
        return row
    raw_path = frame.get("path")
    if not isinstance(raw_path, str):
        return row
    mapped_path = raw_path
    for source, target in path_remaps:
        if mapped_path == source or mapped_path.startswith(source + "/"):
            mapped_path = target + mapped_path[len(source) :]
            break
    if mapped_path == raw_path:
        return row
    mapped_frame = dict(frame)
    mapped_frame["path"] = mapped_path
    mapped_row = dict(row)
    mapped_row["frame"] = mapped_frame
    return mapped_row


def _float_to_byte(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 1.0:
        numeric *= 255.0
    return max(0, min(255, int(round(numeric))))


def _luma_values_for_offset(row: dict[str, Any], offset: int) -> list[Any] | None:
    frame = row.get("frame")
    if int(offset) <= 0 and isinstance(frame, dict) and isinstance(frame.get("luma16"), list):
        return list(frame.get("luma16") or [])
    if int(offset) > 0 and isinstance(row.get("next_frame_luma16"), list):
        return list(row.get("next_frame_luma16") or [])
    return None


def _compact_luma_frame(
    row: dict[str, Any],
    *,
    offset: int,
    image_size: int,
    missing_frame_policy: str,
) -> tuple[bytes, bool]:
    values = _luma_values_for_offset(row, int(offset))
    if values is None:
        if missing_frame_policy == "zero":
            return bytes(int(image_size) * int(image_size)), True
        raise FileNotFoundError("missing compact luma frame fields")
    side = int(math.sqrt(len(values)))
    if side * side != len(values) or side <= 0:
        if missing_frame_policy == "zero":
            return bytes(int(image_size) * int(image_size)), True
        raise ValueError(f"expected square compact luma field, got {len(values)} values")
    image_size = int(image_size)
    out = bytearray(image_size * image_size)
    for y in range(image_size):
        src_y = min(side - 1, y * side // image_size)
        for x in range(image_size):
            src_x = min(side - 1, x * side // image_size)
            out[y * image_size + x] = _float_to_byte(values[src_y * side + src_x])
    return bytes(out), False


def _frames_for_row(
    row: dict[str, Any],
    *,
    provider: _FramePairProvider,
    config: FrameEmbeddingMaterializerConfig,
) -> tuple[list[bytes], int]:
    source = config.frame_source.replace("_", "-").lower()
    if source == "video":
        return provider.frames(_row_for_frame_lookup(row, config.path_remaps), offsets=config.frame_offsets), 0
    if source in {"compact-luma", "luma16"}:
        missing_before = provider.missing_frames
        frames: list[bytes] = []
        for offset in config.frame_offsets:
            try:
                frame, missing = _compact_luma_frame(
                    row,
                    offset=int(offset),
                    image_size=int(config.image_size),
                    missing_frame_policy=str(config.missing_frame_policy),
                )
            except FileNotFoundError:
                provider.missing_frames += 1
                raise
            frames.append(frame)
            if missing:
                provider.missing_frames += 1
        return frames, provider.missing_frames - missing_before
    raise ValueError("frame_source must be one of: video, compact-luma")


def _flush_batch(
    rows: Sequence[dict[str, Any]],
    frames_by_row: Sequence[Sequence[bytes]],
    *,
    embedder: Any,
    config: FrameEmbeddingMaterializerConfig,
) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        return [], 0
    flat_frames = [frame for frames in frames_by_row for frame in frames]
    flat_embeddings = embedder.embed_frames(flat_frames)
    per_row = len(config.frame_offsets)
    if len(flat_embeddings) != len(rows) * per_row:
        raise ValueError(
            f"embedding count mismatch: got {len(flat_embeddings)} for rows={len(rows)} offsets={per_row}"
        )
    output_rows: list[dict[str, Any]] = []
    embedding_dim = 0
    for row_idx, row in enumerate(rows):
        start = row_idx * per_row
        embeddings = flat_embeddings[start : start + per_row]
        embedding_dim = len(embeddings[0]) if embeddings else 0
        features: list[float] = []
        for embedding in embeddings:
            features.extend(float(value) for value in embedding)
        if config.include_embedding_deltas:
            features.extend(_embedding_deltas(embeddings))
        summary_feature_count = 0
        if config.include_summary_features:
            summary_features = record_features(row, feature_mode=config.summary_feature_mode)
            summary_feature_count = len(summary_features)
            features.extend(float(value) for value in summary_features)
        rounded = _round_values(features, config.round_digits)
        out_row = dict(row)
        metadata = {
            "schema": "g005_frozen_frame_embedding_features.row.v1",
            "backend": config.backend,
            "model_id": config.model_id if config.backend != "dummy-stat" else "deterministic-frame-statistics",
            "frame_offsets": list(config.frame_offsets),
            "embedding_pooling": config.embedding_pooling if config.backend != "dummy-stat" else "stats",
            "hf_preprocess": config.hf_preprocess if config.backend == "hf-vision" else None,
            "embedding_dim_per_frame": embedding_dim,
            "include_embedding_deltas": bool(config.include_embedding_deltas),
            "include_summary_features": bool(config.include_summary_features),
            "summary_feature_mode": config.summary_feature_mode if config.include_summary_features else None,
            "summary_feature_count": summary_feature_count,
            "feature_dim": len(rounded),
        }
        out_row["__streaming_idm_features"] = rounded
        out_row["frame_embedding_feature_metadata"] = metadata
        out_row["frame_embedding_feature_fingerprint"] = _metadata_fingerprint(metadata)
        output_rows.append(out_row)
    return output_rows, embedding_dim


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        write_json(path, payload)


def materialize_frame_embedding_features(config: FrameEmbeddingMaterializerConfig) -> dict[str, Any]:
    started_at = time.time()
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    ensure_dir(output_path.parent)
    ensure_dir(Path(config.summary_out).parent)
    if config.progress_output is not None:
        ensure_dir(config.progress_output.parent)
    embedder = _build_embedder(config)
    provider = _FramePairProvider(
        root=Path(".").resolve(),
        image_size=int(config.image_size),
        fps=int(config.frame_fps),
        next_frame_offset=1,
        missing_frame_policy=str(config.missing_frame_policy),
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows_seen = 0
    rows_written = 0
    feature_override_rows = 0
    feature_lengths: list[int] = []
    embedding_dim_per_frame: int | None = None
    dataset_fingerprint = hashlib.sha256()
    batch_rows: list[dict[str, Any]] = []
    batch_frames: list[list[bytes]] = []
    progress_base = {
        "schema": "g005_frozen_frame_embedding_materializer_progress.v1",
        "status": "running",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "backend": config.backend,
        "model_id": config.model_id,
        "frame_offsets": list(config.frame_offsets),
        "frame_source": config.frame_source,
        "path_remaps": list(config.path_remaps),
        "source_label": config.source_label,
    }
    try:
        with tmp_path.open("w", encoding="utf-8", buffering=_JSONL_BUFFER) as out:
            for row in _iter_jsonl(input_path):
                if config.max_rows is not None and rows_seen >= int(config.max_rows):
                    break
                rows_seen += 1
                dataset_fingerprint.update(
                    json.dumps(
                        {
                            "sequence_id": row.get("sequence_id"),
                            "recording_id": row.get("recording_id"),
                            "timestamp_ns": row.get("timestamp_ns"),
                            "ground_truth_tokens": row.get("ground_truth_tokens", []),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=_JSONL_SEPARATORS,
                    ).encode("utf-8")
                )
                dataset_fingerprint.update(b"\n")
                batch_rows.append(row)
                frames, _missing_delta = _frames_for_row(row, provider=provider, config=config)
                batch_frames.append(frames)
                if len(batch_rows) >= max(1, int(config.batch_size)):
                    output_rows, emb_dim = _flush_batch(batch_rows, batch_frames, embedder=embedder, config=config)
                    embedding_dim_per_frame = emb_dim
                    for out_row in output_rows:
                        out.write(_dumps(out_row) + "\n")
                        rows_written += 1
                        feature_override_rows += int("__streaming_idm_features" in out_row)
                        feature_lengths.append(len(out_row.get("__streaming_idm_features", [])))
                    batch_rows = []
                    batch_frames = []
                if config.progress_rows > 0 and rows_seen % int(config.progress_rows) == 0:
                    _write_progress(
                        config.progress_output,
                        {
                            **progress_base,
                            "rows_seen": rows_seen,
                            "rows_written": rows_written,
                            "feature_override_rows": feature_override_rows,
                            "missing_frames": int(provider.missing_frames),
                            "video_restarts": int(provider.video_restarts),
                            "elapsed_seconds": time.time() - started_at,
                        },
                    )
            if batch_rows:
                output_rows, emb_dim = _flush_batch(batch_rows, batch_frames, embedder=embedder, config=config)
                embedding_dim_per_frame = emb_dim
                for out_row in output_rows:
                    out.write(_dumps(out_row) + "\n")
                    rows_written += 1
                    feature_override_rows += int("__streaming_idm_features" in out_row)
                    feature_lengths.append(len(out_row.get("__streaming_idm_features", [])))
    finally:
        provider.close()
    tmp_path.replace(output_path)
    unique_feature_lengths = sorted(set(feature_lengths))
    status = "pass"
    errors: list[str] = []
    if rows_seen != rows_written:
        status = "fail"
        errors.append(f"rows_seen ({rows_seen}) != rows_written ({rows_written})")
    if feature_override_rows != rows_written:
        status = "fail"
        errors.append(f"feature_override_rows ({feature_override_rows}) != rows_written ({rows_written})")
    if len(unique_feature_lengths) != 1:
        status = "fail"
        errors.append(f"inconsistent feature lengths: {unique_feature_lengths[:8]}")
    summary = {
        "schema": "g005_frozen_frame_embedding_materializer.v1",
        "status": status,
        "error_count": len(errors),
        "errors": errors,
        "source_label": config.source_label,
        "input": _source_metadata(input_path),
        "output": _source_metadata(output_path),
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "feature_override_rows": feature_override_rows,
        "feature_dim": unique_feature_lengths[0] if unique_feature_lengths else 0,
        "unique_feature_lengths": unique_feature_lengths,
        "embedding_dim_per_frame": int(embedding_dim_per_frame or getattr(embedder, "embedding_dim", 0) or 0),
        "backend": config.backend,
        "model_id": config.model_id if config.backend != "dummy-stat" else "deterministic-frame-statistics",
        "frame_offsets": list(config.frame_offsets),
        "frame_source": config.frame_source,
        "image_size": int(config.image_size),
        "frame_fps": int(config.frame_fps),
        "missing_frame_policy": str(config.missing_frame_policy),
        "path_remaps": [{"from": source, "to": target} for source, target in config.path_remaps],
        "missing_frames": int(provider.missing_frames),
        "video_restarts": int(provider.video_restarts),
        "batch_size": int(config.batch_size),
        "embedding_pooling": config.embedding_pooling,
        "hf_preprocess": config.hf_preprocess if config.backend == "hf-vision" else None,
        "normalize_embeddings": bool(config.normalize_embeddings),
        "include_embedding_deltas": bool(config.include_embedding_deltas),
        "include_summary_features": bool(config.include_summary_features),
        "summary_feature_mode": config.summary_feature_mode if config.include_summary_features else None,
        "max_rows": config.max_rows,
        "round_digits": config.round_digits,
        "dataset_fingerprint": dataset_fingerprint.hexdigest(),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "claim_boundary": (
            "Frozen frame-embedding materialization is a prefix-gated G005 diagnostic/preparation artifact. "
            "It is not trained-model evidence until a downstream IDM run beats the paper-target metrics."
        ),
    }
    write_json(config.summary_out, summary)
    _write_progress(config.progress_output, {**progress_base, "status": status, "rows_seen": rows_seen, "rows_written": rows_written})
    return summary
