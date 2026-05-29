from __future__ import annotations

import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.config import load_config
from fdm_d2e.io_utils import ensure_dir, read_json, stable_hash_json, write_json
from fdm_d2e.schema import validate_named
from fdm_d2e.training.neural_idm import target_mouse_delta
from fdm_d2e.training.streaming_idm import (
    StreamingActionMetrics,
    _aggregate_epoch_stats,
    _barrier,
    _group_keys,
    _merge_named_metric_states,
    _merge_nested_metric_states,
    _metric_payloads,
    _nested_metric_state_map,
    _record_paths_from_config,
    _streaming_statistical_comparison,
    _training_cache_assignment_plan,
    _training_cache_manifest_byte_count,
    _training_cache_manifest_row_count,
    _training_cache_rank_assignment,
    iter_jsonl,
)
from fdm_d2e.training.torch_idm import (
    MOUSE_AXIS_CLASSES,
    _axis_suffix_from_delta,
    _button_class_metadata,
    _button_target_indices,
    _categorical_loss,
    _prediction_from_output,
    button_softmax_classes,
    require_torch,
)


_FRAME_REF_RE = re.compile(r"^(?P<source>.+)#frame=(?P<index>\d+)$")
_PPM_FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>\.ppm)$")
_JSONL_COMPACT_SEPARATORS = (",", ":")
_CHECKPOINT_CALIBRATION_KEYS = {
    "button_softmax_threshold",
    "category_thresholds",
    "keyboard_softmax_threshold",
    "mouse_output_gain",
}


def _record_paths_from_value(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _chunk_sequence(items: Sequence[Path], chunks: int) -> list[list[Path]]:
    chunks = max(1, min(int(chunks), len(items)))
    base = len(items) // chunks
    extra = len(items) % chunks
    out: list[list[Path]] = []
    start = 0
    for idx in range(chunks):
        size = base + (1 if idx < extra else 0)
        out.append(list(items[start : start + size]))
        start += size
    return [chunk for chunk in out if chunk]


def _glob_record_paths(pattern: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if pattern is None:
        return []
    patterns = [pattern] if isinstance(pattern, (str, Path)) else list(pattern)
    paths: list[Path] = []
    for item in patterns:
        paths.extend(Path(match) for match in sorted(glob.glob(str(item))))
    return paths


def _video_frame_offsets(config: dict[str, Any]) -> tuple[int, ...]:
    value = config.get("video_frame_offsets")
    if value is None:
        return (0, int(config.get("next_frame_offset", 1)))
    offsets = tuple(int(item) for item in value)
    if not offsets:
        raise ValueError("video_frame_offsets must not be empty")
    return offsets


def _is_category_token(token: str) -> bool:
    return token.startswith("KEY_") or (
        token.startswith("MOUSE_")
        and not token.startswith("MOUSE_DX_")
        and not token.startswith("MOUSE_DY_")
    )


def _is_keyboard_token(token: str) -> bool:
    return token.startswith("KEY_")


def _is_mouse_button_token(token: str) -> bool:
    return token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))


def _tokens(row: dict[str, Any]) -> list[str]:
    return [str(token) for token in row.get("ground_truth_tokens", [])]


def _jsonl_source_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_source_identity(
    path: str | Path,
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    identity = {
        "schema": "video_idm_cache_identity.v1",
        "source": _jsonl_source_metadata(path),
        "split_name": str(split_name),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "image_size": int(config.get("video_image_size", 112)),
        "frame_fps": int(config.get("video_frame_fps", 20)),
        "next_frame_offset": int(config.get("next_frame_offset", 1)),
        "category_vocab": list(stats.get("category_vocab", [])),
        "button_head_mode": str(config.get("button_head_mode", "softmax")),
        "button_classes": list(stats.get("button_classes", [])),
        "mouse_head_mode": str(config.get("mouse_head_mode", "regression")),
        "mouse_axis_classes": list(config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)),
        "mouse_target_mode": str(config.get("mouse_target_mode", "sum")),
        "game_vocab": list(stats.get("game_vocab", [])),
        "cache_version": 1,
    }
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    if keyboard_head_mode == "softmax":
        identity["keyboard_head_mode"] = keyboard_head_mode
        identity["keyboard_classes"] = list(stats.get("keyboard_classes", []))
    if "video_frame_offsets" in config:
        identity["frame_offsets"] = list(_video_frame_offsets(config))
    return identity


def _video_cache_manifest_path(
    cache_dir: str | Path,
    path: str | Path,
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
) -> Path:
    identity = _cache_source_identity(path, stats=stats, config=config, split_name=split_name)
    key = stable_hash_json(identity)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(path).stem)[:64] or "records"
    return Path(cache_dir) / str(split_name) / f"{safe_stem}-{key[:20]}.manifest.json"


def _read_ppm_tokens(payload: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    idx = 0
    while len(tokens) < 4:
        while idx < len(payload) and payload[idx] in b" \t\r\n":
            idx += 1
        if idx < len(payload) and payload[idx] == ord("#"):
            while idx < len(payload) and payload[idx] not in b"\r\n":
                idx += 1
            continue
        start = idx
        while idx < len(payload) and payload[idx] not in b" \t\r\n":
            idx += 1
        if start == idx:
            raise ValueError("invalid PPM header")
        tokens.append(payload[start:idx])
    while idx < len(payload) and payload[idx] in b" \t\r\n":
        idx += 1
    return tokens, idx


def _resize_gray_nearest(gray: bytes | bytearray, *, width: int, height: int, output_size: int) -> bytes:
    if width == output_size and height == output_size:
        return bytes(gray)
    out = bytearray(output_size * output_size)
    for y in range(output_size):
        src_y = min(height - 1, y * height // output_size)
        row_base = src_y * width
        out_base = y * output_size
        for x in range(output_size):
            src_x = min(width - 1, x * width // output_size)
            out[out_base + x] = gray[row_base + src_x]
    return bytes(out)


def _ppm_gray(path: Path, *, output_size: int) -> bytes:
    payload = path.read_bytes()
    tokens, offset = _read_ppm_tokens(payload)
    magic = tokens[0]
    width, height, max_value = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if width <= 0 or height <= 0 or max_value <= 0 or max_value > 255:
        raise ValueError(f"unsupported PPM header: {path}")
    if magic == b"P5":
        expected = width * height
        pixels = payload[offset : offset + expected]
        if len(pixels) != expected:
            raise ValueError(f"truncated P5 PPM payload: {path}")
        return _resize_gray_nearest(pixels, width=width, height=height, output_size=output_size)
    if magic != b"P6":
        raise ValueError(f"expected P5/P6 PPM frame: {path}")
    expected = width * height * 3
    pixels = payload[offset : offset + expected]
    if len(pixels) != expected:
        raise ValueError(f"truncated P6 PPM payload: {path}")
    gray = bytearray(width * height)
    for idx in range(width * height):
        base = idx * 3
        gray[idx] = (77 * pixels[base] + 150 * pixels[base + 1] + 29 * pixels[base + 2]) >> 8
    return _resize_gray_nearest(gray, width=width, height=height, output_size=output_size)


class _VideoFrameStream:
    def __init__(self, source: str, *, image_size: int, fps: int) -> None:
        self.source = source
        self.image_size = int(image_size)
        self.fps = int(fps)
        self.frame_size = self.image_size * self.image_size
        self.proc: subprocess.Popen[bytes] | None = None
        self.cap: Any | None = None
        self.backend = ""
        self.current_index = 0
        self.last_frame: bytes | None = None
        self.cache: dict[int, bytes] = {}
        self.cv2_seek_gap = 8

    def _open(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self._open_cv2()
            return
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            self.source,
            "-vf",
            f"fps={self.fps},scale={self.image_size}:{self.image_size}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.cap = None
        self.backend = "ffmpeg"
        self.current_index = 0
        self.last_frame = None
        self.cache = {}

    def _open_cv2(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional cluster dependency.
            raise FileNotFoundError(
                "ffmpeg is not installed and cv2 is unavailable for video frame decoding"
            ) from exc
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"unable to open video source without ffmpeg: {self.source}")
        self.proc = None
        self.cap = cap
        self.backend = "cv2"
        self.current_index = 0
        self.last_frame = None
        self.cache = {}

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.proc = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.backend = ""
        self.current_index = 0
        self.last_frame = None
        self.cache = {}

    def _read_next_cv2(self) -> bytes | None:
        if self.cap is None:
            self._open_cv2()
        assert self.cap is not None
        import cv2  # type: ignore

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] != self.image_size or gray.shape[1] != self.image_size:
            gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        payload = gray.tobytes()
        frame_index = self.current_index
        self.current_index += 1
        self.last_frame = payload
        self.cache[frame_index] = payload
        min_keep = self.current_index - 8
        for old_index in [idx for idx in self.cache if idx < min_keep]:
            self.cache.pop(old_index, None)
        return payload

    def _seek_cv2(self, index: int) -> None:
        if self.cap is None:
            self._open_cv2()
        assert self.cap is not None
        import cv2  # type: ignore

        target_msec = 1000.0 * float(index) / max(1.0, float(self.fps))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        self.current_index = int(index)
        self.last_frame = None
        self.cache = {}

    def _read_next(self) -> bytes | None:
        if self.proc is None and self.cap is None:
            self._open()
        if self.backend == "cv2":
            return self._read_next_cv2()
        assert self.proc is not None
        assert self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = self.frame_size
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else b""
            code = self.proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, ["ffmpeg", self.source], stderr=stderr)
            return None
        frame_index = self.current_index
        frame = b"".join(chunks)
        self.current_index += 1
        self.last_frame = frame
        self.cache[frame_index] = frame
        min_keep = self.current_index - 8
        for old_index in [idx for idx in self.cache if idx < min_keep]:
            self.cache.pop(old_index, None)
        return frame

    def get(self, index: int) -> bytes | None:
        if index in self.cache:
            return self.cache[index]
        if index < self.current_index:
            self.close()
        if self.proc is None and self.cap is None:
            self._open()
        if self.backend == "cv2" and index > self.current_index + self.cv2_seek_gap:
            self._seek_cv2(index)
        while self.current_index <= index:
            frame = self._read_next()
            if frame is None:
                return self.last_frame
        return self.last_frame


class _FramePairProvider:
    def __init__(self, *, root: Path, image_size: int, fps: int, next_frame_offset: int, missing_frame_policy: str) -> None:
        self.root = root
        self.image_size = int(image_size)
        self.fps = int(fps)
        self.next_frame_offset = int(next_frame_offset)
        self.missing_frame_policy = str(missing_frame_policy)
        self.max_open_streams = 2
        self.streams: dict[str, _VideoFrameStream] = {}
        self.missing_frames = 0
        self.video_restarts = 0

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()
        self.streams = {}

    def _zero(self) -> bytes:
        return bytes(self.image_size * self.image_size)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _resolve_source(self, value: str) -> str:
        if value.startswith(("http://", "https://", "s3://", "gs://")):
            return value
        path = Path(value)
        return str(path if path.is_absolute() else self.root / path)

    def _stream_frame(self, source: str, index: int) -> bytes | None:
        stream = self.streams.get(source)
        if stream is None:
            if len(self.streams) >= self.max_open_streams:
                for old_source, old_stream in list(self.streams.items()):
                    old_stream.close()
                    self.streams.pop(old_source, None)
            stream = _VideoFrameStream(source, image_size=self.image_size, fps=self.fps)
            self.streams[source] = stream
            self.video_restarts += 1
        return stream.get(index)

    def _frame_from_ppm(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        return _ppm_gray(path, output_size=self.image_size)

    def _next_ppm_path(self, path: Path) -> Path | None:
        return self._ppm_path_with_offset(path, self.next_frame_offset)

    def _ppm_path_with_offset(self, path: Path, offset: int) -> Path | None:
        match = _PPM_FRAME_RE.match(path.name)
        if not match:
            return None
        number = int(match.group("number")) + int(offset)
        width = len(match.group("number"))
        if number < 0:
            return None
        return path.with_name(f"{match.group('prefix')}{number:0{width}d}{match.group('suffix')}")

    def frames(self, row: dict[str, Any], *, offsets: Sequence[int]) -> list[bytes]:
        frame = row.get("frame", {}) if isinstance(row.get("frame"), dict) else {}
        raw_path = str(frame.get("path") or "")
        if not raw_path:
            return [self._missing_frame() for _ in offsets]
        match = _FRAME_REF_RE.match(raw_path)
        if match:
            source = self._resolve_source(match.group("source"))
            index = int(match.group("index"))
            return [self._resolve_optional_frame(self._stream_frame(source, index + int(offset))) for offset in offsets]
        path = self._resolve(raw_path)
        if path.suffix.lower() == ".ppm":
            frames: list[bytes] = []
            for offset in offsets:
                offset_path = self._ppm_path_with_offset(path, int(offset))
                frames.append(self._resolve_optional_frame(self._frame_from_ppm(offset_path) if offset_path is not None else None))
            return frames
        index = int(frame.get("index", row.get("bin_index", 0)) or 0)
        source = str(path)
        return [self._resolve_optional_frame(self._stream_frame(source, index + int(offset))) for offset in offsets]

    def pair(self, row: dict[str, Any]) -> bytes:
        frame = row.get("frame", {}) if isinstance(row.get("frame"), dict) else {}
        raw_path = str(frame.get("path") or "")
        if not raw_path:
            return self._missing_pair()
        match = _FRAME_REF_RE.match(raw_path)
        if match:
            source = self._resolve_source(match.group("source"))
            index = int(match.group("index"))
            current = self._stream_frame(source, index)
            nxt = self._stream_frame(source, index + self.next_frame_offset)
            return self._join_or_missing(current, nxt)
        path = self._resolve(raw_path)
        if path.suffix.lower() == ".ppm":
            current = self._frame_from_ppm(path)
            next_path = self._next_ppm_path(path)
            nxt = self._frame_from_ppm(next_path) if next_path is not None else current
            return self._join_or_missing(current, nxt)
        index = int(frame.get("index", row.get("bin_index", 0)) or 0)
        source = str(path)
        current = self._stream_frame(source, index)
        nxt = self._stream_frame(source, index + self.next_frame_offset)
        return self._join_or_missing(current, nxt)

    def _missing_frame(self) -> bytes:
        self.missing_frames += 1
        if self.missing_frame_policy == "zero":
            return self._zero()
        raise FileNotFoundError("missing video IDM frame reference")

    def _missing_pair(self) -> bytes:
        self.missing_frames += 2
        if self.missing_frame_policy == "zero":
            zero = self._zero()
            return zero + zero
        raise FileNotFoundError("missing video IDM frame reference")

    def _resolve_optional_frame(self, frame: bytes | None) -> bytes:
        if frame is None:
            return self._missing_frame()
        return frame

    def _join_or_missing(self, current: bytes | None, nxt: bytes | None) -> bytes:
        if current is None:
            self.missing_frames += 1
            if self.missing_frame_policy != "zero":
                raise FileNotFoundError("missing current video IDM frame")
            current = self._zero()
        if nxt is None:
            self.missing_frames += 1
            nxt = current if self.missing_frame_policy != "zero" else self._zero()
        return current + nxt


def _temporal_basis(row: dict[str, Any]) -> list[float]:
    bin_index = float(row.get("bin_index", 0))
    values = [bin_index / 10_000.0]
    for period in (2.0, 3.0, 4.0, 5.0, 8.0, 16.0):
        phase = 2.0 * math.pi * bin_index / period
        values.extend([math.sin(phase), math.cos(phase)])
    return values


def _aux_features(row: dict[str, Any], *, game_vocab: Sequence[str]) -> list[float]:
    game = str(row.get("game", ""))
    return _temporal_basis(row) + [1.0 if game == item else 0.0 for item in game_vocab]


def _axis_indices(row: dict[str, Any], axis_index: dict[str, int], *, mouse_target_mode: str) -> tuple[int, int]:
    dx, dy = target_mouse_delta(row, mode=mouse_target_mode)
    return (
        axis_index[_axis_suffix_from_delta(dx, "MOUSE_DX_")],
        axis_index[_axis_suffix_from_delta(dy, "MOUSE_DY_")],
    )


def _button_label(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({token for token in _tokens(row) if _is_mouse_button_token(token)}))


def _keyboard_label(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({token for token in _tokens(row) if _is_keyboard_token(token)}))


def _button_class_counts(record_paths: Sequence[str | Path], classes: Sequence[Sequence[str]]) -> dict[str, int]:
    class_index = {tuple(str(token) for token in row): idx for idx, row in enumerate(classes)}
    counts = {str(idx): 0 for idx in range(len(classes))}
    for row in _iter_records(record_paths):
        idx = class_index.get(_button_label(row), 0)
        counts[str(idx)] = counts.get(str(idx), 0) + 1
    return counts


def _iter_records(record_paths: str | Path | Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    for record_path in _record_paths_from_value(record_paths):
        yield from iter_jsonl(record_path)


def _scan_video_idm_stats_part(path: str | Path) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    keyboard_counter: Counter[tuple[str, ...]] = Counter({(): 0})
    button_counter: Counter[tuple[str, ...]] = Counter({(): 0})
    game_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    global_tokens: Counter[str] = Counter()
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps(_jsonl_source_metadata(path), sort_keys=True).encode("utf-8"))
    fingerprint.update(b"\n")
    examples = 0
    for row in iter_jsonl(path):
        examples += 1
        tokens = _tokens(row)
        global_tokens.update(tokens or ["NOOP"])
        for token in tokens:
            if _is_category_token(token):
                category_counts[token] += 1
        keyboard_counter[_keyboard_label(row)] += 1
        button_counter[_button_label(row)] += 1
        game_counts[str(row.get("game", ""))] += 1
        if row.get("source_id") is not None:
            source_ids.add(str(row["source_id"]))
        if row.get("resolution_tier") is not None:
            resolution_tiers.add(str(row["resolution_tier"]))
        if row.get("split") is not None:
            split_names.add(str(row["split"]))
        for tag in row.get("eval_split_tags", []) or []:
            eval_split_tags.add(str(tag))
        fingerprint.update(
            json.dumps(
                {
                    "sequence_id": row.get("sequence_id"),
                    "recording_id": row.get("recording_id"),
                    "game": row.get("game"),
                    "tokens": tokens,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        fingerprint.update(b"\n")
    return {
        "path": str(path),
        "examples": int(examples),
        "category_counts": category_counts,
        "keyboard_counter": keyboard_counter,
        "button_counter": button_counter,
        "game_counts": game_counts,
        "source_ids": source_ids,
        "resolution_tiers": resolution_tiers,
        "split_names": split_names,
        "eval_split_tags": eval_split_tags,
        "global_tokens": global_tokens,
        "fingerprint": fingerprint.hexdigest(),
    }


def scan_video_idm_stats(
    record_paths: str | Path | Sequence[str | Path],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    category_min_count = int(config.get("categorical_min_count", 1))
    keyboard_head_mode = str(config.get("keyboard_head_mode", "multilabel"))
    button_head_mode = str(config.get("button_head_mode", "softmax"))
    category_counts: Counter[str] = Counter()
    keyboard_counter: Counter[tuple[str, ...]] = Counter({(): 0})
    button_counter: Counter[tuple[str, ...]] = Counter({(): 0})
    game_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    resolution_tiers: set[str] = set()
    split_names: set[str] = set()
    eval_split_tags: set[str] = set()
    fingerprint = hashlib.sha256()
    examples = 0
    global_tokens: Counter[str] = Counter()
    paths = _record_paths_from_value(record_paths)
    workers = max(1, int(config.get("video_stats_num_workers", config.get("stats_num_workers", 1))))
    if len(paths) > 1 and workers > 1:
        parts_by_path: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            futures = {pool.submit(_scan_video_idm_stats_part, path): str(path) for path in paths}
            for future in as_completed(futures):
                parts_by_path[futures[future]] = future.result()
        parts = [parts_by_path[str(path)] for path in paths]
    else:
        parts = [_scan_video_idm_stats_part(path) for path in paths]
    for part in parts:
        examples += int(part["examples"])
        category_counts.update(part["category_counts"])
        keyboard_counter.update(part["keyboard_counter"])
        button_counter.update(part["button_counter"])
        game_counts.update(part["game_counts"])
        source_ids.update(part["source_ids"])
        resolution_tiers.update(part["resolution_tiers"])
        split_names.update(part["split_names"])
        eval_split_tags.update(part["eval_split_tags"])
        global_tokens.update(part["global_tokens"])
        fingerprint.update(str(part["fingerprint"]).encode("utf-8"))
        fingerprint.update(b"\n")
    if keyboard_head_mode == "softmax" or button_head_mode == "softmax":
        category_vocab = sorted(
            token
            for token, count in category_counts.items()
            if count >= category_min_count
            and not (keyboard_head_mode == "softmax" and _is_keyboard_token(token))
            and not (button_head_mode == "softmax" and _is_mouse_button_token(token))
        )
    else:
        category_vocab = sorted(token for token, count in category_counts.items() if count >= category_min_count)
    keyboard_classes = [()]
    if keyboard_head_mode == "softmax":
        min_count = int(config.get("keyboard_softmax_min_count", 1))
        for label in sorted(label for label, count in keyboard_counter.items() if label and count >= min_count):
            keyboard_classes.append(label)
    button_classes = [()]
    for label in sorted(label for label, count in button_counter.items() if label and count >= int(config.get("button_softmax_min_count", 1))):
        button_classes.append(label)
    game_vocab = sorted(game for game, count in game_counts.items() if game and count >= int(config.get("game_min_count", 1)))
    stats = {
        "schema": "video_idm_stats.v1",
        "num_examples": int(examples),
        "dataset_fingerprint": fingerprint.hexdigest(),
        "image_size": int(config.get("video_image_size", 112)),
        "frame_fps": int(config.get("video_frame_fps", 20)),
        "frame_offsets": list(_video_frame_offsets(config)),
        "stats_num_workers": int(workers),
        "category_vocab": category_vocab,
        "category_counts": {token: int(category_counts.get(token, 0)) for token in category_vocab},
        "keyboard_head_mode": keyboard_head_mode,
        "keyboard_classes": [list(row) for row in keyboard_classes],
        "keyboard_class_counts": {str(idx): int(keyboard_counter.get(tuple(tokens), 0)) for idx, tokens in enumerate(keyboard_classes)},
        "button_head_mode": button_head_mode,
        "button_classes": [list(row) for row in button_classes],
        "button_class_counts": {str(idx): int(button_counter.get(tuple(tokens), 0)) for idx, tokens in enumerate(button_classes)},
        "mouse_head_mode": str(config.get("mouse_head_mode", "regression")),
        "mouse_axis_classes": [str(value) for value in config.get("mouse_axis_classes", MOUSE_AXIS_CLASSES)],
        "mouse_target_mode": str(config.get("mouse_target_mode", "sum")),
        "game_vocab": game_vocab,
        "game_counts": {game: int(game_counts[game]) for game in game_vocab},
        "aux_dim": 13 + len(game_vocab),
        "source_ids": sorted(source_ids),
        "resolution_tiers": sorted(resolution_tiers),
        "split_names": sorted(split_names),
        "eval_split_tags": sorted(eval_split_tags),
        "global_majority_tokens": [token for token, _count in global_tokens.most_common(max(1, int(config.get("global_majority_token_count", 3))))],
    }
    return stats


def _flush_video_cache_chunk(
    torch,
    *,
    chunk_path: Path,
    frame_pairs: list[bytes],
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    image_size = int(stats["image_size"])
    frame_count = len(_video_frame_offsets(config))
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    category_index = {token: idx for idx, token in enumerate(category_vocab)}
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in stats.get("keyboard_classes", [])]
    keyboard_class_index = {label: idx for idx, label in enumerate(keyboard_classes)}
    button_head_mode = str(config.get("button_head_mode", stats.get("button_head_mode", "softmax")))
    button_classes = [tuple(str(token) for token in row) for row in stats.get("button_classes", [])]
    button_class_index = {label: idx for idx, label in enumerate(button_classes)}
    axis_classes = [str(value) for value in config.get("mouse_axis_classes", stats.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
    axis_index = {label: idx for idx, label in enumerate(axis_classes)}
    mouse_head_mode = str(config.get("mouse_head_mode", stats.get("mouse_head_mode", "regression")))
    mouse_target_mode = str(config.get("mouse_target_mode", stats.get("mouse_target_mode", "sum")))
    game_vocab = [str(game) for game in stats.get("game_vocab", [])]
    frame_tensors = [
        torch.frombuffer(bytearray(pair), dtype=torch.uint8).reshape(frame_count, image_size, image_size)
        for pair in frame_pairs
    ]
    cat_y = torch.zeros((len(rows), len(category_vocab)), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for token in set(_tokens(row)):
            idx = category_index.get(token)
            if idx is not None:
                cat_y[row_idx, idx] = 1.0
    payload: dict[str, Any] = {
        "schema": "video_idm_cache_chunk.v1",
        "rows": int(len(rows)),
        "frames": torch.stack(frame_tensors, dim=0),
        "aux": torch.tensor([_aux_features(row, game_vocab=game_vocab) for row in rows], dtype=torch.float32),
        "mouse_y": torch.tensor([target_mouse_delta(row, mode=mouse_target_mode) for row in rows], dtype=torch.float32),
        "cat_y": cat_y,
    }
    if button_head_mode == "softmax":
        if keyboard_head_mode == "softmax":
            payload["keyboard_y"] = torch.tensor(
                [keyboard_class_index.get(_keyboard_label(row), 0) for row in rows],
                dtype=torch.long,
            )
        payload["button_y"] = torch.tensor(
            [button_class_index.get(_button_label(row), 0) for row in rows],
            dtype=torch.long,
        )
    elif keyboard_head_mode == "softmax":
        payload["keyboard_y"] = torch.tensor(
            [keyboard_class_index.get(_keyboard_label(row), 0) for row in rows],
            dtype=torch.long,
        )
    if mouse_head_mode == "axis_softmax":
        axis = [_axis_indices(row, axis_index, mouse_target_mode=mouse_target_mode) for row in rows]
        payload["dx_y"] = torch.tensor([item[0] for item in axis], dtype=torch.long)
        payload["dy_y"] = torch.tensor([item[1] for item in axis], dtype=torch.long)
    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(chunk_path)
    return {"path": str(chunk_path), "rows": int(len(rows)), "bytes": int(chunk_path.stat().st_size)}


def _build_video_cache_for_path(
    path: str | Path,
    *,
    manifest_path: str | Path,
    identity: dict[str, Any],
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not force_rebuild:
        manifest = read_json(manifest_path)
        chunks = manifest.get("chunks", [])
        if manifest.get("identity") == identity and chunks and all(Path(row["path"]).exists() for row in chunks):
            return manifest
    torch = require_torch()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = ensure_dir(manifest_path.with_suffix(""))
    for old_chunk in chunk_dir.glob("chunk_*.pt"):
        old_chunk.unlink()
    root = Path(config.get("root", ".")).resolve()
    chunk_size = int(config.get("video_cache_chunk_size", config.get("batch_size", 512) * 2))
    provider = _FramePairProvider(
        root=root,
        image_size=int(config.get("video_image_size", stats.get("image_size", 112))),
        fps=int(config.get("video_frame_fps", stats.get("frame_fps", 20))),
        next_frame_offset=int(config.get("next_frame_offset", 1)),
        missing_frame_policy=str(config.get("missing_frame_policy", "error")),
    )
    frame_offsets = _video_frame_offsets(config)
    rows: list[dict[str, Any]] = []
    frame_pairs: list[bytes] = []
    chunks: list[dict[str, Any]] = []
    count = 0
    started = time.time()
    try:
        for row in iter_jsonl(path):
            rows.append(row)
            frame_pairs.append(b"".join(provider.frames(row, offsets=frame_offsets)))
            count += 1
            if len(rows) >= chunk_size:
                chunks.append(
                    _flush_video_cache_chunk(
                        torch,
                        chunk_path=chunk_dir / f"chunk_{len(chunks):06d}.pt",
                        frame_pairs=frame_pairs,
                        rows=rows,
                        stats=stats,
                        config=config,
                    )
                )
                rows = []
                frame_pairs = []
        if rows:
            chunks.append(
                _flush_video_cache_chunk(
                    torch,
                    chunk_path=chunk_dir / f"chunk_{len(chunks):06d}.pt",
                    frame_pairs=frame_pairs,
                    rows=rows,
                    stats=stats,
                    config=config,
                )
            )
    finally:
        provider.close()
    manifest = {
        "schema": "video_idm_cache_manifest.v1",
        "identity": identity,
        "split_name": str(split_name),
        "source_path": str(path),
        "manifest_path": str(manifest_path),
        "chunk_size": int(chunk_size),
        "rows": int(count),
        "bytes": int(sum(int(chunk.get("bytes", 0)) for chunk in chunks)),
        "chunks": chunks,
        "provider_summary": {
            "missing_frames": int(provider.missing_frames),
            "video_restarts": int(provider.video_restarts),
            "wall_clock_seconds": time.time() - started,
        },
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    write_json(tmp_manifest, manifest)
    tmp_manifest.replace(manifest_path)
    return manifest


def build_video_idm_cache_manifests(
    record_paths: Sequence[str | Path],
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    cache_dir = config.get("video_cache_dir")
    if not cache_dir:
        raise ValueError("video IDM requires video_cache_dir")
    force_rebuild = bool(config.get("force_rebuild_video_cache", False))
    tasks = []
    for path in record_paths:
        identity = _cache_source_identity(path, stats=stats, config=config, split_name=split_name)
        manifest_path = _video_cache_manifest_path(cache_dir, path, stats=stats, config=config, split_name=split_name)
        tasks.append((path, manifest_path, identity))
    workers = max(1, int(config.get("video_cache_num_workers", 1)))
    if len(tasks) > 1 and workers > 1:
        manifests: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {
                pool.submit(
                    _build_video_cache_for_path,
                    path,
                    manifest_path=manifest_path,
                    identity=identity,
                    stats=stats,
                    config=config,
                    split_name=split_name,
                    force_rebuild=force_rebuild,
                ): manifest_path
                for path, manifest_path, identity in tasks
            }
            for future in as_completed(futures):
                manifests.append(future.result())
        return sorted(manifests, key=lambda row: str(row["source_path"]))
    return [
        _build_video_cache_for_path(
            path,
            manifest_path=manifest_path,
            identity=identity,
            stats=stats,
            config=config,
            split_name=split_name,
            force_rebuild=force_rebuild,
        )
        for path, manifest_path, identity in tasks
    ]


def load_video_idm_cache_manifests(
    record_paths: Sequence[str | Path],
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    cache_dir = config.get("video_cache_dir")
    if not cache_dir:
        raise ValueError("video IDM requires video_cache_dir")
    manifests: list[dict[str, Any]] = []
    for path in record_paths:
        identity = _cache_source_identity(path, stats=stats, config=config, split_name=split_name)
        manifest_path = _video_cache_manifest_path(cache_dir, path, stats=stats, config=config, split_name=split_name)
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing video IDM cache manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale video IDM cache manifest: {manifest_path}")
        manifests.append(manifest)
    return manifests


def _video_cache_precompute_splits(config: dict[str, Any]) -> set[str]:
    raw_value = config.get("video_cache_precompute_splits", config.get("precompute_splits", ["train", "target"]))
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",")]
    else:
        parts = [str(part).strip() for part in raw_value]
    splits = {part for part in parts if part}
    unknown = splits - {"train", "target"}
    if unknown:
        raise ValueError(f"unsupported video cache precompute split(s): {sorted(unknown)}")
    if not splits:
        raise ValueError("video_cache_precompute_splits must include train, target, or both")
    return splits


def _load_existing_video_cache_manifests(
    record_paths: Sequence[str | Path],
    *,
    stats: dict[str, Any],
    config: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    try:
        return load_video_idm_cache_manifests(record_paths, stats=stats, config=config, split_name=split_name)
    except FileNotFoundError:
        return []


def precompute_video_idm_cache(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config.get("output_dir", "outputs/idm_video_pair"))
    train_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    target_paths = _record_paths_from_config(
        config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    stats_path = Path(config.get("stats_path", Path(output_dir) / "video_idm_stats.json"))
    if stats_path.exists() and not bool(config.get("force_rebuild_video_stats", False)):
        stats = read_json(stats_path)
    else:
        stats = scan_video_idm_stats(train_paths, config=config)
        write_json(stats_path, stats)
    requested_splits = _video_cache_precompute_splits(config)
    train_manifests = (
        build_video_idm_cache_manifests(train_paths, stats=stats, config=config, split_name="train")
        if "train" in requested_splits
        else _load_existing_video_cache_manifests(train_paths, stats=stats, config=config, split_name="train")
    )
    target_manifests = (
        build_video_idm_cache_manifests(target_paths, stats=stats, config=config, split_name="target")
        if "target" in requested_splits
        else _load_existing_video_cache_manifests(target_paths, stats=stats, config=config, split_name="target")
    )
    summary = {
        "schema": "video_idm_cache_precompute_summary.v1",
        "status": "pass",
        "stats_path": str(stats_path),
        "requested_splits": sorted(requested_splits),
        "train_record_paths": [str(path) for path in train_paths],
        "target_record_paths": [str(path) for path in target_paths],
        "train_cache": {
            "precomputed": "train" in requested_splits,
            "manifest_paths": [str(row["manifest_path"]) for row in train_manifests],
            "rows": sum(_training_cache_manifest_row_count(row) for row in train_manifests),
            "bytes": sum(_training_cache_manifest_byte_count(row) for row in train_manifests),
        },
        "target_cache": {
            "precomputed": "target" in requested_splits,
            "manifest_paths": [str(row["manifest_path"]) for row in target_manifests],
            "rows": sum(_training_cache_manifest_row_count(row) for row in target_manifests),
            "bytes": sum(_training_cache_manifest_byte_count(row) for row in target_manifests),
        },
    }
    summary_out = config.get("cache_summary_out")
    if summary_out:
        write_json(summary_out, summary)
    return summary


def _soft_pos_weight(torch, category_counts: dict[str, int], vocab: list[str], total: int, *, cap: float, device: str):
    if not vocab:
        return None
    values = []
    for token in vocab:
        pos = max(1, int(category_counts.get(token, 0)))
        neg = max(1, total - pos)
        values.append(min(float(cap), neg / pos))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _class_weight(torch, counts: dict[str, int], *, class_count: int, cap: float, device: str):
    if class_count <= 0:
        return None
    raw = [max(1, int(counts.get(str(idx), 0))) for idx in range(class_count)]
    total = float(sum(raw) or 1)
    values = [min(float(cap), total / (count * max(1, class_count))) for count in raw]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _video_input_channels(mode: str, *, frame_count: int) -> int:
    if mode in {"pair", "stack"}:
        return int(frame_count)
    if mode == "pair_delta":
        return 3
    if mode == "pair_delta_abs":
        return 4
    if mode == "stack_delta":
        return int(frame_count) + max(0, int(frame_count) - 1)
    if mode == "stack_delta_abs":
        return int(frame_count) + (2 * max(0, int(frame_count) - 1))
    raise ValueError(f"unsupported video_input_mode: {mode}")


def _augment_video_inputs(torch, frames, *, mode: str):
    if mode in {"pair", "stack"}:
        return frames
    if mode.startswith("pair") and frames.shape[1] != 2:
        raise ValueError(f"video_input_mode={mode} requires two cached frame channels")
    if mode == "pair_delta":
        delta = frames[:, 1:2] - frames[:, 0:1]
        return torch.cat([frames, delta], dim=1)
    if mode == "pair_delta_abs":
        delta = frames[:, 1:2] - frames[:, 0:1]
        return torch.cat([frames, delta, delta.abs()], dim=1)
    if mode in {"stack_delta", "stack_delta_abs"}:
        if frames.shape[1] < 2:
            raise ValueError(f"video_input_mode={mode} requires at least two cached frame channels")
        deltas = frames[:, 1:] - frames[:, :-1]
        if mode == "stack_delta":
            return torch.cat([frames, deltas], dim=1)
        return torch.cat([frames, deltas, deltas.abs()], dim=1)
    raise ValueError(f"unsupported video_input_mode: {mode}")


def _build_video_pair_model(torch, *, output_dim: int, aux_dim: int, config: dict[str, Any]):
    channels = [int(value) for value in config.get("video_conv_channels", [32, 64, 128, 256])]
    if not channels:
        raise ValueError("video_conv_channels must not be empty")
    dropout = float(config.get("dropout", 0.05))
    hidden_dim = int(config.get("hidden_dim", 1024))
    depth = int(config.get("depth", 2))
    video_input_mode = str(config.get("video_input_mode", "pair"))
    input_channels = _video_input_channels(video_input_mode, frame_count=len(_video_frame_offsets(config)))

    class ResidualBlock(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(dim),
                torch.nn.GELU(),
                torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(dim),
            )
            self.act = torch.nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class VideoPairIDM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = input_channels
            for out_channels in channels:
                layers.extend(
                    [
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.GELU(),
                        ResidualBlock(out_channels),
                    ]
                )
                in_channels = out_channels
            self.encoder = torch.nn.Sequential(*layers, torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten())
            encoded_dim = channels[-1] * 4 * 4
            head_layers = []
            dim = encoded_dim + int(aux_dim)
            for _ in range(max(0, depth)):
                head_layers.extend([torch.nn.Linear(dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout)])
                dim = hidden_dim
            head_layers.append(torch.nn.Linear(dim, output_dim))
            self.head = torch.nn.Sequential(*head_layers)

        def forward(self, frames, aux):
            encoded = self.encoder(_augment_video_inputs(torch, frames, mode=video_input_mode))
            return self.head(torch.cat([encoded, aux], dim=1))

    return VideoPairIDM()


def _video_model_signature(config: dict[str, Any], *, output_dim: int, aux_dim: int, stats: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "schema": "video_idm_model_signature.v1",
        "output_dim": int(output_dim),
        "aux_dim": int(aux_dim),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "video_input_mode": str(config.get("video_input_mode", "pair")),
        "video_conv_channels": [int(value) for value in config.get("video_conv_channels", [32, 64, 128, 256])],
        "hidden_dim": int(config.get("hidden_dim", 1024)),
        "depth": int(config.get("depth", 2)),
        "button_head_mode": str(config.get("button_head_mode", stats.get("button_head_mode", "softmax"))),
        "mouse_head_mode": str(config.get("mouse_head_mode", stats.get("mouse_head_mode", "regression"))),
        "category_vocab_sha256": hashlib.sha256(
            json.dumps(list(stats.get("category_vocab", [])), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "button_classes_sha256": hashlib.sha256(
            json.dumps(list(stats.get("button_classes", [])), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if "video_frame_offsets" in config:
        signature["video_frame_offsets"] = list(_video_frame_offsets(config))
        signature["video_input_channels"] = _video_input_channels(
            str(config.get("video_input_mode", "pair")),
            frame_count=len(_video_frame_offsets(config)),
        )
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    if keyboard_head_mode == "softmax":
        signature["keyboard_head_mode"] = keyboard_head_mode
        signature["keyboard_classes_sha256"] = hashlib.sha256(
            json.dumps(list(stats.get("keyboard_classes", [])), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return signature


def _load_video_train_state(torch, path: Path, *, device: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != "video_idm_train_state.v1":
        raise ValueError(f"invalid video IDM train state: {path}")
    return payload


def _save_video_train_state(
    torch,
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    signature: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "video_idm_train_state.v1",
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "signature": signature,
            "runtime_overrides": config.get("runtime_overrides", {}),
            "saved_at_epoch": time.time(),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _iter_video_cache_batches(
    torch,
    cache_manifests: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    max_examples: int | None,
    rank: int,
    world_size: int,
    shard_by_path: bool,
    shard_assignment: str,
) -> Iterable[tuple[Any, Any, Any, Any, Any | None, Any | None, Any | None, Any | None, int]]:
    seen = 0
    source_batch_idx = 0
    assigned_indices = (
        _training_cache_rank_assignment(cache_manifests, rank=rank, world_size=world_size, mode=shard_assignment)
        if shard_by_path and world_size > 1
        else None
    )
    for path_idx, manifest in enumerate(cache_manifests):
        if assigned_indices is not None and path_idx not in assigned_indices:
            continue
        for chunk in manifest.get("chunks", []):
            payload = _load_cache_chunk(torch, chunk["path"])
            rows = int(payload["rows"])
            for start in range(0, rows, batch_size):
                if max_examples is not None and seen >= max_examples:
                    break
                end = min(rows, start + batch_size)
                if max_examples is not None:
                    end = min(end, start + (max_examples - seen))
                batch_rows = int(end - start)
                current_source_batch_idx = source_batch_idx
                source_batch_idx += 1
                if not shard_by_path and world_size > 1 and (current_source_batch_idx % world_size) != rank:
                    continue
                frames = payload["frames"][start:end].to(device=device, dtype=torch.float32).div_(255.0)
                aux = payload["aux"][start:end].to(device=device, dtype=torch.float32)
                mouse_y = payload["mouse_y"][start:end].to(device)
                cat_y = payload["cat_y"][start:end].to(device)
                keyboard_y = payload.get("keyboard_y")
                button_y = payload.get("button_y")
                dx_y = payload.get("dx_y")
                dy_y = payload.get("dy_y")
                if keyboard_y is not None:
                    keyboard_y = keyboard_y[start:end].to(device)
                if button_y is not None:
                    button_y = button_y[start:end].to(device)
                if dx_y is not None and dy_y is not None:
                    dx_y = dx_y[start:end].to(device)
                    dy_y = dy_y[start:end].to(device)
                seen += batch_rows
                yield frames, aux, mouse_y, cat_y, keyboard_y, button_y, dx_y, dy_y, batch_rows
            if max_examples is not None and seen >= max_examples:
                break
        if max_examples is not None and seen >= max_examples:
            break


def _train_one_epoch(
    torch,
    model,
    opt,
    *,
    train_manifests: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    cat_pos_weight,
    keyboard_class_weight,
    button_class_weight,
    axis_class_weight,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in stats.get("keyboard_classes", [])]
    button_head_mode = str(config.get("button_head_mode", stats.get("button_head_mode", "softmax")))
    button_classes = [tuple(str(token) for token in row) for row in stats.get("button_classes", [])]
    mouse_head_mode = str(config.get("mouse_head_mode", stats.get("mouse_head_mode", "regression")))
    mouse_axis_classes = [str(value) for value in config.get("mouse_axis_classes", stats.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
    keyboard_output_dim = len(keyboard_classes) if keyboard_head_mode == "softmax" else 0
    button_output_dim = len(button_classes) if button_head_mode == "softmax" else 0
    batch_size = int(config.get("batch_size", 512))
    shard_by_path = bool(config.get("video_cache_shard_by_path", len(train_manifests) > 1))
    shard_assignment = str(config.get("video_cache_shard_assignment", "greedy_rows"))
    losses: list[float] = []
    loss_sum = 0.0
    batches = 0
    examples = 0
    progress_interval = int(config.get("training_progress_interval_batches", 0) or 0)
    progress_dir = ensure_dir(Path(config.get("output_dir", "outputs/idm_video_pair")) / "rank_progress")
    heartbeat_path = progress_dir / f"train_rank{rank}.json"
    for batch_idx, (frames, aux, mouse_y, cat_y, keyboard_y, button_y, dx_y, dy_y, batch_rows) in enumerate(
        _iter_video_cache_batches(
            torch,
            train_manifests,
            batch_size=batch_size,
            device=device,
            max_examples=config.get("max_train_examples"),
            rank=rank,
            world_size=world_size,
            shard_by_path=shard_by_path,
            shard_assignment=shard_assignment,
        )
    ):
        pred = model(frames, aux)
        mouse_loss = torch.nn.functional.smooth_l1_loss(pred[:, :2], mouse_y)
        category_end = 2 + len(category_vocab)
        keyboard_end = category_end + keyboard_output_dim
        button_end = keyboard_end + button_output_dim
        cat_loss = (
            _categorical_loss(torch, pred[:, 2:category_end], cat_y, cat_pos_weight, config)
            if category_vocab
            else torch.tensor(0.0, device=device)
        )
        if keyboard_head_mode == "softmax" and keyboard_y is not None and len(keyboard_classes) > 1:
            keyboard_loss = torch.nn.functional.cross_entropy(
                pred[:, category_end:keyboard_end],
                keyboard_y,
                weight=keyboard_class_weight,
            )
        else:
            keyboard_loss = torch.tensor(0.0, device=device)
        if button_head_mode == "softmax" and button_y is not None and len(button_classes) > 1:
            button_loss = torch.nn.functional.cross_entropy(
                pred[:, keyboard_end:button_end],
                button_y,
                weight=button_class_weight,
            )
        else:
            button_loss = torch.tensor(0.0, device=device)
        if mouse_head_mode == "axis_softmax" and dx_y is not None and dy_y is not None:
            axis_count = len(mouse_axis_classes)
            dx_logits = pred[:, button_end : button_end + axis_count]
            dy_logits = pred[:, button_end + axis_count : button_end + (2 * axis_count)]
            axis_loss = 0.5 * (
                torch.nn.functional.cross_entropy(dx_logits, dx_y, weight=axis_class_weight)
                + torch.nn.functional.cross_entropy(dy_logits, dy_y, weight=axis_class_weight)
            )
        else:
            axis_loss = torch.tensor(0.0, device=device)
        loss = (
            float(config.get("mouse_regression_loss_weight", 1.0)) * mouse_loss
            + float(config.get("categorical_loss_weight", 1.0)) * cat_loss
            + float(config.get("keyboard_softmax_loss_weight", 1.0)) * keyboard_loss
            + float(config.get("button_softmax_loss_weight", 1.0)) * button_loss
            + float(config.get("mouse_axis_loss_weight", 1.0 if mouse_head_mode == "axis_softmax" else 0.0)) * axis_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
        opt.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        loss_sum += loss_value * batch_rows
        batches += 1
        examples += batch_rows
        if progress_interval > 0 and (batches == 1 or batches % progress_interval == 0):
            write_json(
                heartbeat_path,
                {
                    "schema": "video_idm_train_rank_progress.v1",
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "batches": int(batches),
                    "examples": int(examples),
                    "last_local_batch_index": int(batch_idx),
                    "loss": loss_value,
                    "updated_at_epoch": time.time(),
                    "video_cache": True,
                    "video_cache_shard_by_path": shard_by_path,
                },
            )
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "loss_sum": loss_sum,
        "batches": batches,
        "examples": examples,
    }


def _metric_state(metric: StreamingActionMetrics) -> dict[str, int | float]:
    fields = (
        "matched",
        "keyboard_total",
        "keyboard_correct",
        "button_total",
        "button_correct",
        "button_predicted_total",
        "button_exact_tp",
        "button_fp",
        "button_fn",
        "button_no_gt",
        "button_no_gt_fp",
        "mouse_n",
        "sum_pred",
        "sum_gt",
        "sum_pred_sq",
        "sum_gt_sq",
        "sum_cross",
        "sum_abs_pred",
        "sum_abs_gt",
        "failure_count",
    )
    return {field: getattr(metric, field) for field in fields}


def _button_fbeta_key(counts: dict[str, int], *, threshold: float, beta: float) -> tuple[float, float, float, float]:
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    fn = float(counts.get("fn", 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta2 = beta * beta
    score = (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall) if precision or recall else 0.0
    return score, precision, recall, float(threshold)


def _select_button_softmax_threshold(
    button_counts: dict[float, dict[str, int]],
    *,
    default_threshold: float,
    beta: float,
    max_no_button_fpr: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Select a train-only button threshold with an optional no-button FPR cap."""

    best_threshold = float(default_threshold)
    best_key = (-1.0, -1.0, -1.0, best_threshold)
    best_counts: dict[str, int] = {}
    eligible: list[tuple[float, dict[str, int]]] = []
    for threshold, counts in sorted(button_counts.items()):
        no_gt = float(counts.get("no_button_examples", 0))
        no_gt_fp = float(counts.get("no_button_false_positive_examples", 0))
        no_button_fpr = no_gt_fp / no_gt if no_gt else 0.0
        if max_no_button_fpr is None or no_button_fpr <= float(max_no_button_fpr):
            eligible.append((float(threshold), counts))
    if not eligible and max_no_button_fpr is not None:
        # Fail closed: if the requested cap is unreachable on the train-calibration
        # sample, choose the least-spammy threshold rather than maximizing recall.
        eligible = sorted(
            ((float(threshold), counts) for threshold, counts in button_counts.items()),
            key=lambda item: (
                float(item[1].get("no_button_false_positive_examples", 0))
                / max(1.0, float(item[1].get("no_button_examples", 0))),
                -item[0],
            ),
        )[:1]
    for threshold, counts in eligible:
        key = _button_fbeta_key(counts, threshold=threshold, beta=beta)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_counts = dict(counts)
    no_button_examples = float(best_counts.get("no_button_examples", 0))
    no_button_fp = float(best_counts.get("no_button_false_positive_examples", 0))
    score, precision, recall, _threshold = best_key
    return best_threshold, {
        **best_counts,
        "score": score if score >= 0.0 else None,
        "precision": precision if precision >= 0.0 else None,
        "recall": recall if recall >= 0.0 else None,
        "no_button_false_positive_rate": no_button_fp / no_button_examples if no_button_examples else 0.0,
        "max_no_button_false_positive_rate": max_no_button_fpr,
        "constraint_satisfied": (
            True
            if max_no_button_fpr is None
            else (no_button_fp / no_button_examples if no_button_examples else 0.0) <= float(max_no_button_fpr)
        ),
        "eligible_thresholds": len(eligible),
    }


def _ensure_metric(metrics: dict[str, StreamingActionMetrics], key: str) -> StreamingActionMetrics:
    if key not in metrics:
        metrics[key] = StreamingActionMetrics()
    return metrics[key]


def _baseline_tokens(name: str, row: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    if name == "noop":
        return ["NOOP"]
    if name == "global_majority":
        return [str(token) for token in stats.get("global_majority_tokens", ["NOOP"])]
    raise ValueError(f"unsupported video IDM baseline: {name}")


def _observe_metrics(
    *,
    row: dict[str, Any],
    tokens: list[str],
    stats: dict[str, Any],
    model_name: str,
    baseline_names: list[str],
    metrics_by_model: dict[str, StreamingActionMetrics],
    group_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]],
    cluster_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]],
) -> None:
    model_tokens = {model_name: tokens}
    for baseline_name in baseline_names:
        model_tokens[baseline_name] = _baseline_tokens(baseline_name, row, stats)
    cluster = str(row.get("recording_id") or row.get("cross_resolution_key") or row.get("sequence_id"))
    for name, pred_tokens in model_tokens.items():
        metrics_by_model[name].update(pred_tokens, row)
        _ensure_metric(cluster_metrics_by_model[name], cluster).update(pred_tokens, row)
        for group_key in _group_keys(row):
            _ensure_metric(group_metrics_by_model[name], group_key).update(pred_tokens, row)


def _predicted_tokens_from_output(output: list[float], *, stats: dict[str, Any], config: dict[str, Any]) -> list[str]:
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    category_threshold = float(config.get("category_threshold", 0.35))
    configured_thresholds = config.get("category_thresholds", {})
    category_thresholds = (
        {token: float(configured_thresholds.get(token, category_threshold)) for token in category_vocab}
        if isinstance(configured_thresholds, dict)
        else {token: category_threshold for token in category_vocab}
    )
    button_classes = [tuple(str(token) for token in row) for row in stats.get("button_classes", [])]
    keyboard_classes = [tuple(str(token) for token in row) for row in stats.get("keyboard_classes", [])]
    _dx, _dy, tokens = _prediction_from_output(
        output,
        base_dx=0.0,
        base_dy=0.0,
        residual_mouse=False,
        category_vocab=category_vocab,
        category_thresholds=category_thresholds,
        category_threshold=category_threshold,
        keyboard_head_mode=str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel"))),
        keyboard_classes=keyboard_classes,
        keyboard_softmax_threshold=float(config.get("keyboard_softmax_threshold", 0.5)),
        button_head_mode=str(config.get("button_head_mode", stats.get("button_head_mode", "softmax"))),
        button_classes=button_classes,
        button_softmax_threshold=float(config.get("button_softmax_threshold", 0.5)),
        mouse_head_mode=str(config.get("mouse_head_mode", stats.get("mouse_head_mode", "regression"))),
        mouse_axis_classes=[str(value) for value in config.get("mouse_axis_classes", stats.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))],
        mouse_axis_decode_mode=str(config.get("mouse_axis_decode_mode", "expected")),
        mouse_axis_temperature=float(config.get("mouse_axis_temperature", 1.0)),
        mouse_output_gain=float(config.get("mouse_output_gain", 1.0)),
        mouse_emit_mode=str(config.get("mouse_emit_mode", "decompose")),
        mouse_max_tokens_per_axis=int(config.get("mouse_max_tokens_per_axis", 32)),
    )
    return tokens


def _load_cache_chunk(torch, path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    source_path = payload.get("payload_source_path")
    if source_path:
        try:
            source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        except TypeError:
            source_payload = torch.load(source_path, map_location="cpu")
        if int(source_payload["rows"]) != int(payload["rows"]):
            raise ValueError(f"sidecar cache row mismatch: {path} -> {source_path}")
        merged = dict(source_payload)
        merged.update(payload)
        return merged
    return payload


def _iter_payload_record_batches(
    torch,
    manifest: dict[str, Any],
    record_path: str | Path,
    *,
    batch_size: int,
    device: str,
    max_examples: int | None,
) -> Iterable[tuple[Any, Any, list[dict[str, Any]]]]:
    emitted = 0
    record_iter = iter_jsonl(record_path)
    for chunk in manifest.get("chunks", []):
        payload = _load_cache_chunk(torch, chunk["path"])
        rows = int(payload["rows"])
        chunk_records = [next(record_iter) for _ in range(rows)]
        for start in range(0, rows, batch_size):
            if max_examples is not None and emitted >= max_examples:
                return
            end = min(rows, start + batch_size)
            if max_examples is not None:
                end = min(end, start + (max_examples - emitted))
            batch_rows = chunk_records[start:end]
            emitted += len(batch_rows)
            yield (
                payload["frames"][start:end].to(device=device, dtype=torch.float32).div_(255.0),
                payload["aux"][start:end].to(device=device, dtype=torch.float32),
                batch_rows,
            )


def _predict_from_cache(
    torch,
    model,
    *,
    target_record_paths: Sequence[str | Path],
    target_manifests: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
    output_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    model_name = str(config.get("model_name", "video_pair_idm"))
    baseline_names = [str(name) for name in config.get("baseline_names", ["noop", "global_majority"])]
    all_model_names = [model_name, *baseline_names]
    metrics_by_model = {name: StreamingActionMetrics() for name in all_model_names}
    group_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    cluster_metrics_by_model: dict[str, dict[str, StreamingActionMetrics]] = {name: {} for name in all_model_names}
    pseudo_path = output_dir / "pseudolabels.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_fingerprint = hashlib.sha256()
    target_count = 0
    target_source_ids: set[str] = set()
    target_resolution_tiers: set[str] = set()
    target_eval_split_tags: set[str] = set()
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 512)))
    max_target_examples = config.get("max_target_examples")
    model.eval()
    with pseudo_path.open("w", encoding="utf-8") as pseudo_f, predictions_path.open("w", encoding="utf-8") as pred_f, torch.no_grad():
        for manifest, record_path in zip(target_manifests, target_record_paths):
            for frames, aux, rows in _iter_payload_record_batches(
                torch,
                manifest,
                record_path,
                batch_size=batch_size,
                device=device,
                max_examples=max_target_examples,
            ):
                outputs = model(frames, aux).detach().cpu().tolist()
                for row, output in zip(rows, outputs):
                    if row.get("source_id") is not None:
                        target_source_ids.add(str(row["source_id"]))
                    if row.get("resolution_tier") is not None:
                        target_resolution_tiers.add(str(row["resolution_tier"]))
                    for tag in row.get("eval_split_tags", []) or []:
                        target_eval_split_tags.add(str(tag))
                    tokens = _predicted_tokens_from_output(output, stats=stats, config=config)
                    pseudo = {
                        "schema": "idm_pseudolabel.v1",
                        "sequence_id": row["sequence_id"],
                        "timestamp_ns": int(row["timestamp_ns"]),
                        "predicted_tokens": tokens,
                        "label_source": "idm_generated",
                        "confidence": max(0.05, min(0.99, 1.0 / (1.0 + len(tokens)))),
                        "model": model_name,
                        "training_split_hash": str(stats["dataset_fingerprint"]),
                        "input_window": {
                            "frame_ref": row.get("frame", {}).get("path", ""),
                            "frame_index": int(row.get("frame", {}).get("index", 0)),
                        },
                    }
                    validate_named(pseudo, "idm_pseudolabel.schema.json")
                    pred = {
                        "sequence_id": row["sequence_id"],
                        "recording_id": row.get("recording_id"),
                        "cross_resolution_key": row.get("cross_resolution_key"),
                        "game": row.get("game"),
                        "timestamp_ns": row["timestamp_ns"],
                        "predicted_tokens": tokens,
                    }
                    pseudo_f.write(json.dumps(pseudo, ensure_ascii=False, separators=_JSONL_COMPACT_SEPARATORS) + "\n")
                    pred_f.write(json.dumps(pred, ensure_ascii=False, separators=_JSONL_COMPACT_SEPARATORS) + "\n")
                    _observe_metrics(
                        row=row,
                        tokens=tokens,
                        stats=stats,
                        model_name=model_name,
                        baseline_names=baseline_names,
                        metrics_by_model=metrics_by_model,
                        group_metrics_by_model=group_metrics_by_model,
                        cluster_metrics_by_model=cluster_metrics_by_model,
                    )
                    sequence_fingerprint.update(json.dumps({"id": row["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                    sequence_fingerprint.update(b"\n")
                    target_count += 1
                    if max_target_examples is not None and target_count >= int(max_target_examples):
                        break
                if max_target_examples is not None and target_count >= int(max_target_examples):
                    break
            if max_target_examples is not None and target_count >= int(max_target_examples):
                break
    metrics_path = output_dir / "metrics.json"
    metrics_payload = metrics_by_model[model_name].payload()
    write_json(metrics_path, metrics_payload)
    label_quality_report = {
        "schema": "idm_label_quality_report.v1",
        "model": model_name,
        "target_records": target_count,
        "model_metrics": metrics_payload,
        "baseline_metrics": {name: metrics_by_model[name].payload() for name in baseline_names},
        "groups_by_model": {
            name: {key: metric.payload() for key, metric in sorted(group_metrics.items())}
            for name, group_metrics in group_metrics_by_model.items()
        },
        "cluster_count": len(cluster_metrics_by_model[model_name]),
    }
    label_quality_report_path = output_dir / "label_quality_report.json"
    write_json(label_quality_report_path, label_quality_report)
    statistical_comparison = None
    statistical_comparison_path = None
    if config.get("endpoints"):
        statistical_comparison = _streaming_statistical_comparison(
            cluster_metrics_by_model,
            load_config(config["endpoints"]),
        )
        statistical_comparison_path = output_dir / "statistical_comparison.json"
        write_json(statistical_comparison_path, statistical_comparison)
    return {
        "pseudo_label_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
        "label_quality_report_path": str(label_quality_report_path),
        "label_quality_report": label_quality_report,
        "statistical_comparison_path": str(statistical_comparison_path) if statistical_comparison_path else None,
        "statistical_comparison": statistical_comparison,
        "target_records": target_count,
        "prediction_fingerprint": sequence_fingerprint.hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "target_source_ids": sorted(target_source_ids),
        "target_resolution_tiers": sorted(target_resolution_tiers),
        "target_eval_split_tags": sorted(target_eval_split_tags),
        "metrics_state": {name: _metric_state(metric) for name, metric in metrics_by_model.items()},
        "group_metrics_state": _nested_metric_state_map(group_metrics_by_model),
        "cluster_metrics_state": _nested_metric_state_map(cluster_metrics_by_model),
    }


def _predict_video_for_parallel_part(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_visible_devices"])
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    torch = require_torch()
    force_cpu = bool(payload.get("force_cpu", False))
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    checkpoint_path = Path(payload["checkpoint_path"])
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prediction_config = dict(payload["prediction_config"])
    stats = dict(checkpoint["stats"])
    model = _build_video_pair_model(
        torch,
        output_dim=int(checkpoint["output_dim"]),
        aux_dim=int(checkpoint.get("aux_dim", stats.get("aux_dim", 13))),
        config=prediction_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    record_paths = [Path(path) for path in payload["record_paths"]]
    target_manifests = load_video_idm_cache_manifests(
        record_paths,
        stats=stats,
        config=prediction_config,
        split_name="target",
    )
    output_dir = ensure_dir(payload["output_dir"])
    prediction = _predict_from_cache(
        torch,
        model,
        target_record_paths=record_paths,
        target_manifests=target_manifests,
        stats=stats,
        config=prediction_config,
        device=device,
        output_dir=Path(output_dir),
        checkpoint_path=checkpoint_path,
    )
    return {
        "part_index": int(payload["part_index"]),
        "record_paths": [str(path) for path in record_paths],
        "output_dir": str(output_dir),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "predictions_path": prediction["predictions_path"],
        "records": int(prediction["target_records"]),
        "target_source_ids": prediction["target_source_ids"],
        "target_resolution_tiers": prediction["target_resolution_tiers"],
        "target_eval_split_tags": prediction["target_eval_split_tags"],
        "metrics_state": prediction["metrics_state"],
        "group_metrics_state": prediction["group_metrics_state"],
        "cluster_metrics_state": prediction["cluster_metrics_state"],
    }


def _concatenate_video_prediction_parts(parts: list[dict[str, Any]], *, output_dir: Path) -> dict[str, Any]:
    pseudo_path = output_dir / "pseudolabels.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    sequence_fingerprint = hashlib.sha256()
    rows = 0
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    with pseudo_path.open("w", encoding="utf-8") as pseudo_out, predictions_path.open("w", encoding="utf-8") as pred_out:
        for part in sorted(parts, key=lambda item: int(item["part_index"])):
            with Path(part["pseudo_label_path"]).open(encoding="utf-8") as pseudo_in, Path(part["predictions_path"]).open(encoding="utf-8") as pred_in:
                for pseudo_line, pred_line in zip(pseudo_in, pred_in):
                    if not pseudo_line.strip() and not pred_line.strip():
                        continue
                    if not pseudo_line.strip() or not pred_line.strip():
                        raise ValueError(f"mismatched blank prediction lines in video part {part['part_index']}")
                    pseudo = json.loads(pseudo_line)
                    pred = json.loads(pred_line)
                    if str(pseudo.get("sequence_id")) != str(pred.get("sequence_id")):
                        raise ValueError(
                            f"sequence_id mismatch while merging video part {part['part_index']}: "
                            f"{pseudo.get('sequence_id')} != {pred.get('sequence_id')}"
                        )
                    tokens = [str(token) for token in pred.get("predicted_tokens", [])]
                    if [str(token) for token in pseudo.get("predicted_tokens", [])] != tokens:
                        raise ValueError(f"token mismatch while merging video part {part['part_index']}: {pred.get('sequence_id')}")
                    pseudo_out.write(pseudo_line if pseudo_line.endswith("\n") else pseudo_line + "\n")
                    pred_out.write(pred_line if pred_line.endswith("\n") else pred_line + "\n")
                    sequence_fingerprint.update(json.dumps({"id": pred["sequence_id"], "tokens": tokens}, sort_keys=True).encode("utf-8"))
                    sequence_fingerprint.update(b"\n")
                    rows += 1
                remaining_pseudo = [line for line in pseudo_in if line.strip()]
                remaining_pred = [line for line in pred_in if line.strip()]
                if remaining_pseudo or remaining_pred:
                    raise ValueError(f"mismatched prediction row counts while merging video part {part['part_index']}")
    return {
        "pseudo_label_path": str(pseudo_path),
        "predictions_path": str(predictions_path),
        "target_records": rows,
        "prediction_fingerprint": sequence_fingerprint.hexdigest(),
    }


def _predict_video_checkpoint_parallel(
    config: dict[str, Any],
    *,
    checkpoint_path: Path,
    target_record_paths: Sequence[Path],
    prediction_config_base: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    workers = int(config.get("prediction_workers", 1))
    if workers <= 1 or len(target_record_paths) <= 1:
        raise ValueError("parallel video prediction requested without multiple workers/record paths")
    if prediction_config_base.get("max_target_examples") is not None:
        raise ValueError("parallel video prediction does not support max_target_examples; use serial prediction for probes")
    chunks = _chunk_sequence(list(target_record_paths), workers)
    parts_root = ensure_dir(config.get("prediction_parts_dir", output_dir / "prediction_parts"))
    cuda_devices = config.get("prediction_cuda_devices")
    if isinstance(cuda_devices, str):
        cuda_devices = [item.strip() for item in cuda_devices.split(",") if item.strip()]
    if cuda_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            cuda_devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        else:
            cuda_devices = list(range(len(chunks)))
    cuda_devices = list(cuda_devices)
    if not cuda_devices and not bool(config.get("force_cpu", False)):
        raise ValueError("prediction_cuda_devices resolved to an empty list")
    prediction_config = dict(prediction_config_base)
    prediction_config["resume_predictions"] = False
    payloads = []
    for part_index, record_paths in enumerate(chunks):
        payloads.append(
            {
                "part_index": part_index,
                "record_paths": [str(path) for path in record_paths],
                "output_dir": str(Path(parts_root) / f"part_{part_index:03d}"),
                "checkpoint_path": str(checkpoint_path),
                "prediction_config": prediction_config,
                "force_cpu": bool(config.get("force_cpu", False)),
                "cuda_visible_devices": None if bool(config.get("force_cpu", False)) else cuda_devices[part_index % len(cuda_devices)],
            }
        )
    parts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_predict_video_for_parallel_part, payload) for payload in payloads]
        for future in as_completed(futures):
            parts.append(future.result())
    parts = sorted(parts, key=lambda item: int(item["part_index"]))
    merged_paths = _concatenate_video_prediction_parts(parts, output_dir=output_dir)
    metrics_by_model = _merge_named_metric_states(part["metrics_state"] for part in parts)
    group_metrics_by_model = _merge_nested_metric_states(part["group_metrics_state"] for part in parts)
    cluster_metrics_by_model = _merge_nested_metric_states(part["cluster_metrics_state"] for part in parts)
    model_name = str(prediction_config.get("model_name", "video_pair_idm"))
    baseline_names = [str(name) for name in prediction_config.get("baseline_names", ["noop", "global_majority"])]
    metrics_path = output_dir / "metrics.json"
    metrics_payload = metrics_by_model[model_name].payload()
    write_json(metrics_path, metrics_payload)
    label_quality_report = {
        "schema": "idm_label_quality_report.v1",
        "model": model_name,
        "target_records": int(merged_paths["target_records"]),
        "model_metrics": metrics_payload,
        "baseline_metrics": {name: metrics_by_model[name].payload() for name in baseline_names if name in metrics_by_model},
        "groups_by_model": {
            name: _metric_payloads(group_metrics)
            for name, group_metrics in group_metrics_by_model.items()
        },
        "cluster_count": len(cluster_metrics_by_model.get(model_name, {})),
    }
    label_quality_report_path = output_dir / "label_quality_report.json"
    write_json(label_quality_report_path, label_quality_report)
    statistical_comparison = None
    statistical_comparison_path = None
    if prediction_config.get("endpoints"):
        statistical_comparison = _streaming_statistical_comparison(
            cluster_metrics_by_model,
            load_config(prediction_config["endpoints"]),
        )
        statistical_comparison_path = output_dir / "statistical_comparison.json"
        write_json(statistical_comparison_path, statistical_comparison)
    return {
        **merged_paths,
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
        "label_quality_report_path": str(label_quality_report_path),
        "label_quality_report": label_quality_report,
        "statistical_comparison_path": str(statistical_comparison_path) if statistical_comparison_path else None,
        "statistical_comparison": statistical_comparison,
        "target_source_ids": sorted({source for part in parts for source in part.get("target_source_ids", [])}),
        "target_resolution_tiers": sorted({tier for part in parts for tier in part.get("target_resolution_tiers", [])}),
        "target_eval_split_tags": sorted({tag for part in parts for tag in part.get("target_eval_split_tags", [])}),
        "metrics_state": {name: _metric_state(metric) for name, metric in metrics_by_model.items()},
        "group_metrics_state": _nested_metric_state_map(group_metrics_by_model),
        "cluster_metrics_state": _nested_metric_state_map(cluster_metrics_by_model),
        "prediction_parallel": {
            "enabled": True,
            "workers": len(parts),
            "parts_dir": str(parts_root),
            "parts": [
                {
                    "part_index": part["part_index"],
                    "records": part["records"],
                    "record_paths": part["record_paths"],
                    "pseudo_label_path": part["pseudo_label_path"],
                    "predictions_path": part["predictions_path"],
                }
                for part in parts
            ],
        },
    }


def _calibrate_from_cache(
    torch,
    model,
    *,
    train_manifests: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    category_threshold = float(config.get("category_threshold", 0.35))
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in stats.get("keyboard_classes", [])]
    button_head_mode = str(config.get("button_head_mode", stats.get("button_head_mode", "softmax")))
    button_classes = [tuple(str(token) for token in row) for row in stats.get("button_classes", [])]
    grid = [float(value) for value in config.get("category_calibration_grid", [x / 100.0 for x in range(5, 96, 5)])]
    beta = float(config.get("category_calibration_beta", 1.0))
    batch_size = int(config.get("category_calibration_batch_size", config.get("eval_batch_size", config.get("batch_size", 512))))
    max_examples = config.get("category_calibration_max_examples")
    thresholds = {token: category_threshold for token in category_vocab}
    group_indices = {
        "keyboard": [idx for idx, token in enumerate(category_vocab) if token.startswith("KEY_")],
        "other": [idx for idx, token in enumerate(category_vocab) if not token.startswith("KEY_")],
    }
    group_indices = {name: indices for name, indices in group_indices.items() if indices}
    group_counts = {
        group: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in grid}
        for group in group_indices
    }
    keyboard_counts = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "predicted_positive": 0,
            "no_key_examples": 0,
            "no_key_false_positive_examples": 0,
        }
        for threshold in grid
    }
    button_counts = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "predicted_positive": 0,
            "no_button_examples": 0,
            "no_button_false_positive_examples": 0,
        }
        for threshold in grid
    }
    predicted_abs_sum = 0.0
    target_abs_sum = 0.0
    value_count = 0
    observed = 0
    model.eval()
    with torch.no_grad():
        for frames, aux, mouse_y, cat_y, keyboard_y, button_y, _dx_y, _dy_y, batch_rows in _iter_video_cache_batches(
            torch,
            train_manifests,
            batch_size=batch_size,
            device=device,
            max_examples=max_examples,
            rank=0,
            world_size=1,
            shard_by_path=False,
            shard_assignment=str(config.get("video_cache_shard_assignment", "greedy_rows")),
        ):
            outputs = model(frames, aux)
            category_end = 2 + len(category_vocab)
            probs = torch.sigmoid(outputs[:, 2:category_end]).detach()
            labels = cat_y.bool()
            observed += int(batch_rows)
            for threshold in grid:
                pred = probs >= float(threshold)
                tp_vec = (pred & labels).sum(dim=0).detach().cpu().tolist()
                fp_vec = (pred & ~labels).sum(dim=0).detach().cpu().tolist()
                fn_vec = (~pred & labels).sum(dim=0).detach().cpu().tolist()
                for group, indices in group_indices.items():
                    counts = group_counts[group][threshold]
                    counts["tp"] += sum(int(tp_vec[idx]) for idx in indices)
                    counts["fp"] += sum(int(fp_vec[idx]) for idx in indices)
                    counts["fn"] += sum(int(fn_vec[idx]) for idx in indices)
            raw_outputs = outputs.detach().cpu().tolist()
            targets = mouse_y.detach().cpu().tolist()
            for output, target in zip(raw_outputs, targets):
                predicted_abs_sum += abs(float(output[0])) + abs(float(output[1]))
                target_abs_sum += abs(float(target[0])) + abs(float(target[1]))
                value_count += 2
            if keyboard_head_mode == "softmax" and keyboard_y is not None and len(keyboard_classes) > 1:
                logits = outputs[:, category_end : category_end + len(keyboard_classes)]
                key_probs = torch.softmax(logits, dim=1).detach()
                key_labels_idx = keyboard_y.detach()
                for threshold in grid:
                    top_probs, top_idx = torch.max(key_probs, dim=1)
                    pred_idx = torch.where((top_idx != 0) & (top_probs >= float(threshold)), top_idx, torch.zeros_like(top_idx))
                    counts = keyboard_counts[threshold]
                    counts["tp"] += int(((pred_idx == key_labels_idx) & (key_labels_idx != 0)).sum().item())
                    counts["fp"] += int(((pred_idx != 0) & (pred_idx != key_labels_idx)).sum().item())
                    counts["fn"] += int(((key_labels_idx != 0) & (pred_idx != key_labels_idx)).sum().item())
                    counts["predicted_positive"] += int((pred_idx != 0).sum().item())
                    counts["no_key_examples"] += int((key_labels_idx == 0).sum().item())
                    counts["no_key_false_positive_examples"] += int(((key_labels_idx == 0) & (pred_idx != 0)).sum().item())
            if button_head_mode == "softmax" and button_y is not None and len(button_classes) > 1:
                button_start = category_end + (len(keyboard_classes) if keyboard_head_mode == "softmax" else 0)
                logits = outputs[:, button_start : button_start + len(button_classes)]
                button_probs = torch.softmax(logits, dim=1).detach()
                labels_idx = button_y.detach()
                for threshold in grid:
                    top_probs, top_idx = torch.max(button_probs, dim=1)
                    pred_idx = torch.where((top_idx != 0) & (top_probs >= float(threshold)), top_idx, torch.zeros_like(top_idx))
                    counts = button_counts[threshold]
                    counts["tp"] += int(((pred_idx == labels_idx) & (labels_idx != 0)).sum().item())
                    counts["fp"] += int(((pred_idx != 0) & (pred_idx != labels_idx)).sum().item())
                    counts["fn"] += int(((labels_idx != 0) & (pred_idx != labels_idx)).sum().item())
                    counts["predicted_positive"] += int((pred_idx != 0).sum().item())
                    counts["no_button_examples"] += int((labels_idx == 0).sum().item())
                    counts["no_button_false_positive_examples"] += int(((labels_idx == 0) & (pred_idx != 0)).sum().item())
    diagnostics: dict[str, Any] = {
        "schema": "video_idm_calibration.v1",
        "observed_examples": int(observed),
        "grid": grid,
        "beta": beta,
    }

    def fbeta_key(counts: dict[str, int], threshold: float) -> tuple[float, float, float, float]:
        tp = float(counts.get("tp", 0))
        fp = float(counts.get("fp", 0))
        fn = float(counts.get("fn", 0))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        beta2 = beta * beta
        score = (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall) if precision or recall else 0.0
        return score, precision, recall, float(threshold)

    per_group = {}
    for group, indices in group_indices.items():
        best_threshold = category_threshold
        best_counts: dict[str, int] = {}
        best_key = (-1.0, -1.0, -1.0, category_threshold)
        for threshold in grid:
            counts = group_counts[group][threshold]
            key = fbeta_key(counts, float(threshold))
            if key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_counts = dict(counts)
        for idx in indices:
            thresholds[category_vocab[idx]] = best_threshold
        per_group[group] = {"threshold": best_threshold, "token_count": len(indices), **best_counts}
    diagnostics["category_thresholds"] = thresholds
    diagnostics["per_group"] = per_group
    keyboard_threshold = float(config.get("keyboard_softmax_threshold", 0.5))
    if keyboard_head_mode == "softmax" and len(keyboard_classes) > 1:
        max_no_key_fpr = config.get("keyboard_softmax_calibration_max_no_key_fpr")
        max_no_key_fpr = float(max_no_key_fpr) if max_no_key_fpr is not None else None
        converted_counts = {
            threshold: {
                "tp": counts.get("tp", 0),
                "fp": counts.get("fp", 0),
                "fn": counts.get("fn", 0),
                "predicted_positive": counts.get("predicted_positive", 0),
                "no_button_examples": counts.get("no_key_examples", 0),
                "no_button_false_positive_examples": counts.get("no_key_false_positive_examples", 0),
            }
            for threshold, counts in keyboard_counts.items()
        }
        keyboard_threshold, key_diag = _select_button_softmax_threshold(
            converted_counts,
            default_threshold=keyboard_threshold,
            beta=beta,
            max_no_button_fpr=max_no_key_fpr,
        )
        diagnostics["keyboard_softmax_threshold"] = keyboard_threshold
        diagnostics["keyboard_softmax_threshold_diagnostics"] = {
            **key_diag,
            "no_key_examples": key_diag.get("no_button_examples", 0),
            "no_key_false_positive_examples": key_diag.get("no_button_false_positive_examples", 0),
            "no_key_false_positive_rate": key_diag.get("no_button_false_positive_rate", 0.0),
            "max_no_key_false_positive_rate": max_no_key_fpr,
        }
    button_threshold = float(config.get("button_softmax_threshold", 0.5))
    if button_head_mode == "softmax" and len(button_classes) > 1:
        max_no_button_fpr = config.get("button_softmax_calibration_max_no_button_fpr")
        max_no_button_fpr = float(max_no_button_fpr) if max_no_button_fpr is not None else None
        button_threshold, best_counts = _select_button_softmax_threshold(
            button_counts,
            default_threshold=button_threshold,
            beta=beta,
            max_no_button_fpr=max_no_button_fpr,
        )
        diagnostics["button_softmax_threshold"] = button_threshold
        diagnostics["button_softmax_threshold_diagnostics"] = best_counts
    configured_gain = float(config.get("mouse_output_gain", 1.0))
    if str(config.get("mouse_output_gain_mode", "fixed")) == "train_abs_ratio" and predicted_abs_sum > 0.0 and target_abs_sum > 0.0:
        min_gain = float(config.get("mouse_output_gain_min", 0.1))
        max_gain = float(config.get("mouse_output_gain_max", 32.0))
        gain = configured_gain * (target_abs_sum / max(predicted_abs_sum, 1e-9))
        diagnostics["mouse_output_gain"] = min(max_gain, max(min_gain, gain))
        diagnostics["mouse_output_gain_diagnostics"] = {
            "predicted_abs_mean": predicted_abs_sum / max(1, value_count),
            "target_abs_mean": target_abs_sum / max(1, value_count),
            "raw_ratio": target_abs_sum / max(predicted_abs_sum, 1e-9),
            "value_count": value_count,
        }
    else:
        diagnostics["mouse_output_gain"] = configured_gain
    return diagnostics


def _distributed_runtime(torch, config: dict[str, Any]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    force_cpu = bool(config.get("force_cpu", False))
    enabled = world_size > 1
    backend = None
    if enabled:
        backend = str(config.get("distributed_backend") or ("nccl" if torch.cuda.is_available() and not force_cpu else "gloo"))
        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            init_kwargs: dict[str, Any] = {"backend": backend}
            timeout_seconds = config.get("distributed_timeout_seconds")
            if timeout_seconds is not None:
                init_kwargs["timeout"] = timedelta(seconds=float(timeout_seconds))
            torch.distributed.init_process_group(**init_kwargs)
    if torch.cuda.is_available() and not force_cpu:
        device = f"cuda:{local_rank}" if enabled else "cuda"
    else:
        device = "cpu"
    return {
        "enabled": enabled,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_rank0": rank == 0,
        "backend": backend,
        "device": device,
    }


def train_video_idm(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    if int(config.get("seed", 0)):
        torch.manual_seed(int(config["seed"]))
    dist = _distributed_runtime(torch, config)
    device = str(dist["device"])
    out_dir = ensure_dir(config.get("output_dir", "outputs/idm_video_pair"))
    train_paths = _record_paths_from_config(
        config,
        primary_key="train_records",
        paths_key="train_record_paths",
        glob_key="train_records_glob",
    )
    target_paths = _record_paths_from_config(
        config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    stats_path = Path(config.get("stats_path", Path(out_dir) / "video_idm_stats.json"))
    if dist["is_rank0"]:
        if stats_path.exists() and not bool(config.get("force_rebuild_video_stats", False)):
            stats = read_json(stats_path)
        else:
            stats = scan_video_idm_stats(train_paths, config=config)
            write_json(stats_path, stats)
    if dist["enabled"]:
        _barrier(torch, dist)
    stats = read_json(stats_path)
    skip_prediction_requested = bool(config.get("skip_prediction", False))
    if bool(config.get("require_precomputed_video_cache", False)):
        train_manifests = load_video_idm_cache_manifests(train_paths, stats=stats, config=config, split_name="train")
        target_manifests = (
            []
            if skip_prediction_requested
            else load_video_idm_cache_manifests(target_paths, stats=stats, config=config, split_name="target")
        )
    elif dist["enabled"] and not dist["is_rank0"]:
        _barrier(torch, dist)
        train_manifests = load_video_idm_cache_manifests(train_paths, stats=stats, config=config, split_name="train")
        target_manifests = (
            []
            if skip_prediction_requested
            else load_video_idm_cache_manifests(target_paths, stats=stats, config=config, split_name="target")
        )
    else:
        train_manifests = build_video_idm_cache_manifests(train_paths, stats=stats, config=config, split_name="train")
        target_manifests = [] if skip_prediction_requested else build_video_idm_cache_manifests(
            target_paths,
            stats=stats,
            config=config,
            split_name="target",
        )
        if dist["enabled"]:
            _barrier(torch, dist)
    category_vocab = [str(token) for token in stats.get("category_vocab", [])]
    keyboard_head_mode = str(config.get("keyboard_head_mode", stats.get("keyboard_head_mode", "multilabel")))
    keyboard_classes = [tuple(str(token) for token in row) for row in stats.get("keyboard_classes", [])]
    button_head_mode = str(config.get("button_head_mode", stats.get("button_head_mode", "softmax")))
    button_classes = [tuple(str(token) for token in row) for row in stats.get("button_classes", [])]
    mouse_head_mode = str(config.get("mouse_head_mode", stats.get("mouse_head_mode", "regression")))
    mouse_axis_classes = [str(value) for value in config.get("mouse_axis_classes", stats.get("mouse_axis_classes", MOUSE_AXIS_CLASSES))]
    keyboard_output_dim = len(keyboard_classes) if keyboard_head_mode == "softmax" else 0
    button_output_dim = len(button_classes) if button_head_mode == "softmax" else 0
    axis_output_dim = (2 * len(mouse_axis_classes)) if mouse_head_mode == "axis_softmax" else 0
    output_dim = 2 + len(category_vocab) + keyboard_output_dim + button_output_dim + axis_output_dim
    aux_dim = int(stats.get("aux_dim", 13))
    signature = _video_model_signature(config, output_dim=output_dim, aux_dim=aux_dim, stats=stats)
    model = _build_video_pair_model(torch, output_dim=output_dim, aux_dim=aux_dim, config=config).to(device)
    train_state_path = Path(config.get("train_state_path", Path(out_dir) / "train_state.pt"))
    resume_state = (
        _load_video_train_state(torch, train_state_path, device=device)
        if bool(config.get("resume_train_state", True))
        else None
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_state is not None:
        if resume_state.get("signature") != signature:
            raise ValueError(
                f"video IDM train state signature mismatch for {train_state_path}; "
                "set resume_train_state=false or use a matching config"
            )
        model.load_state_dict(resume_state["model_state"])
        start_epoch = int(resume_state.get("epoch", 0))
        history = [dict(row) for row in resume_state.get("history", [])]
    train_model = model
    if dist["enabled"]:
        ddp_kwargs = {"device_ids": [int(dist["local_rank"])]} if str(device).startswith("cuda") else {}
        train_model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    opt = torch.optim.AdamW(
        train_model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer_state"])
    cat_pos_weight = _soft_pos_weight(
        torch,
        {str(k): int(v) for k, v in stats.get("category_counts", {}).items()},
        category_vocab,
        int(stats["num_examples"]),
        cap=float(config.get("categorical_pos_weight_cap", 80.0)),
        device=device,
    )
    button_class_weight = _class_weight(
        torch,
        {str(k): int(v) for k, v in stats.get("button_class_counts", {}).items()},
        class_count=len(button_classes),
        cap=float(config.get("button_softmax_class_weight_cap", 20.0)),
        device=device,
    )
    keyboard_class_weight = _class_weight(
        torch,
        {str(k): int(v) for k, v in stats.get("keyboard_class_counts", {}).items()},
        class_count=len(keyboard_classes),
        cap=float(config.get("keyboard_softmax_class_weight_cap", 20.0)),
        device=device,
    )
    axis_class_weight = _class_weight(
        torch,
        {str(idx): 1 for idx, _label in enumerate(mouse_axis_classes)},
        class_count=len(mouse_axis_classes),
        cap=float(config.get("mouse_axis_class_weight_cap", 20.0)),
        device=device,
    )
    for epoch in range(start_epoch, int(config.get("epochs", 3))):
        join_context = train_model.join() if dist["enabled"] else nullcontext()
        with join_context:
            epoch_stats = _train_one_epoch(
                torch,
                train_model,
                opt,
                train_manifests=train_manifests,
                stats=stats,
                config=config,
                device=device,
                cat_pos_weight=cat_pos_weight,
                keyboard_class_weight=keyboard_class_weight,
                button_class_weight=button_class_weight,
                axis_class_weight=axis_class_weight,
                rank=int(dist["rank"]),
                world_size=int(dist["world_size"]),
            )
        epoch_stats = _aggregate_epoch_stats(torch, epoch_stats, device=device, dist=dist)
        if dist["is_rank0"]:
            row = {"epoch": epoch + 1, **epoch_stats}
            history.append(row)
            write_json(out_dir / "train_history.json", {"schema": "video_idm_train_history.v1", "history": history})
            write_json(
                out_dir / "convergence_report.json",
                {
                    "schema": "streaming_convergence_report.v1",
                    "score_mode": str(config.get("convergence_score", "train_loss")),
                    "direction": "lower",
                    "eval_interval_epochs": 0,
                    "history": [{"epoch": item["epoch"], "train_loss": item.get("loss")} for item in history],
                    "report_path": str(out_dir / "convergence_report.json"),
                },
            )
            _save_video_train_state(
                torch,
                train_state_path,
                model=model,
                optimizer=opt,
                epoch=epoch + 1,
                history=history,
                signature=signature,
                config=config,
            )
        _barrier(torch, dist)
    if dist["enabled"] and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not dist["is_rank0"]:
        return {
            "schema": "video_idm_worker_summary.v1",
            "rank": int(dist["rank"]),
            "world_size": int(dist["world_size"]),
            "status": "worker_complete",
        }
    calibration = _calibrate_from_cache(
        torch,
        model,
        train_manifests=train_manifests,
        stats=stats,
        config=config,
        device=device,
    )
    calibrated_config = dict(config)
    if "category_thresholds" in calibration:
        calibrated_config["category_thresholds"] = calibration["category_thresholds"]
    if "keyboard_softmax_threshold" in calibration:
        calibrated_config["keyboard_softmax_threshold"] = calibration["keyboard_softmax_threshold"]
    if "button_softmax_threshold" in calibration:
        calibrated_config["button_softmax_threshold"] = calibration["button_softmax_threshold"]
    if "mouse_output_gain" in calibration:
        calibrated_config["mouse_output_gain"] = calibration["mouse_output_gain"]
    checkpoint_path = Path(out_dir) / "checkpoint.pt"
    torch.save(
        {
            "schema": "video_idm_checkpoint.v1",
            "model_state": model.state_dict(),
            "model_config": calibrated_config,
            "stats": stats,
            "output_dim": int(output_dim),
            "aux_dim": int(stats.get("aux_dim", 13)),
            "history": history,
            "calibration": calibration,
        },
        checkpoint_path,
    )
    if bool(config.get("skip_prediction", False)):
        prediction = {
            "pseudo_label_path": str(Path(out_dir) / "pseudolabels.jsonl"),
            "predictions_path": str(Path(out_dir) / "predictions.jsonl"),
            "metrics_path": str(Path(out_dir) / "metrics.json"),
            "metrics": None,
            "label_quality_report_path": str(Path(out_dir) / "label_quality_report.json"),
            "label_quality_report": None,
            "statistical_comparison_path": None,
            "statistical_comparison": None,
            "target_records": 0,
            "prediction_fingerprint": None,
            "checkpoint_path": str(checkpoint_path),
            "target_source_ids": [],
            "target_resolution_tiers": [],
            "target_eval_split_tags": [],
            "skipped": True,
        }
    else:
        prediction = _predict_from_cache(
            torch,
            model,
            target_record_paths=target_paths,
            target_manifests=target_manifests,
            stats=stats,
            config=calibrated_config,
            device=device,
            output_dir=Path(out_dir),
            checkpoint_path=checkpoint_path,
        )
    metadata = {
        "schema": "idm_checkpoint_metadata.v1",
        "model": str(config.get("model_name", "video_pair_idm")),
        "dataset_fingerprint": str(stats["dataset_fingerprint"]),
        "train_records": int(stats["num_examples"]),
        "target_records": int(prediction["target_records"]),
        "pseudo_label_path": prediction["pseudo_label_path"],
        "filtered_pseudo_label_path": prediction["pseudo_label_path"],
        "checkpoint_path": str(checkpoint_path),
        "train_state_path": str(train_state_path),
        "metrics_path": prediction["metrics_path"],
        "calibration": calibration,
        "distributed": {key: value for key, value in dist.items() if key != "device"},
    }
    validate_named(metadata, "idm_checkpoint_metadata.schema.json")
    write_json(Path(out_dir) / "checkpoint_metadata.json", metadata)
    summary = {
        "schema": "video_idm_train_summary.v1",
        "model_name": str(config.get("model_name", "video_pair_idm")),
        "device": device,
        "distributed": {key: value for key, value in dist.items() if key != "device"},
        "stats_path": str(stats_path),
        "checkpoint_path": str(checkpoint_path),
        "train_state_path": str(train_state_path),
        "resumed_from_train_state": resume_state is not None,
        "start_epoch": int(start_epoch),
        "metadata": metadata,
        "train_history_path": str(Path(out_dir) / "train_history.json"),
        "convergence_report_path": str(Path(out_dir) / "convergence_report.json"),
        "train_cache": {
            "manifest_paths": [str(row["manifest_path"]) for row in train_manifests],
            "rows": sum(_training_cache_manifest_row_count(row) for row in train_manifests),
            "bytes": sum(_training_cache_manifest_byte_count(row) for row in train_manifests),
            "assignment_plan": _training_cache_assignment_plan(
                train_manifests,
                world_size=max(1, int(dist["world_size"])),
                mode=str(config.get("video_cache_shard_assignment", "greedy_rows")),
            ),
        },
        "target_cache": {
            "manifest_paths": [str(row["manifest_path"]) for row in target_manifests],
            "rows": sum(_training_cache_manifest_row_count(row) for row in target_manifests),
            "bytes": sum(_training_cache_manifest_byte_count(row) for row in target_manifests),
        },
        "prediction": prediction,
        "metrics": prediction["metrics"],
        "label_quality_report": prediction["label_quality_report"],
        "statistical_comparison": prediction["statistical_comparison"],
    }
    summary_out = config.get("summary_out")
    if summary_out:
        write_json(summary_out, summary)
    return summary


def predict_video_idm_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    force_cpu = bool(config.get("force_cpu", False))
    checkpoint_value = config.get("checkpoint_path", config.get("checkpoint"))
    if not checkpoint_value:
        raise ValueError("predict_video_idm_checkpoint requires checkpoint_path")
    checkpoint_path = Path(checkpoint_value)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = dict(checkpoint.get("model_config", {}))
    prediction_config = dict(checkpoint_config)
    prediction_config.update({key: value for key, value in config.items() if value is not None})
    if not bool(config.get("override_checkpoint_calibration", False)):
        for key in _CHECKPOINT_CALIBRATION_KEYS:
            if key in checkpoint_config:
                prediction_config[key] = checkpoint_config[key]
    stats = dict(checkpoint["stats"])
    target_paths = _record_paths_from_config(
        prediction_config,
        primary_key="target_records",
        paths_key="target_record_paths",
        glob_key="target_records_glob",
    )
    output_dir = ensure_dir(prediction_config.get("output_dir", checkpoint_path.parent))
    prediction_workers = int(prediction_config.get("prediction_workers", 1) or 1)
    parallel_prediction = (
        prediction_workers > 1
        and len(target_paths) > 1
        and prediction_config.get("max_target_examples") is None
    )
    recalibration = None
    if bool(prediction_config.get("recalibrate_from_train_cache", False)):
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        model = _build_video_pair_model(
            torch,
            output_dim=int(checkpoint["output_dim"]),
            aux_dim=int(checkpoint.get("aux_dim", stats.get("aux_dim", 13))),
            config=prediction_config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        train_paths = _record_paths_from_config(
            prediction_config,
            primary_key="train_records",
            paths_key="train_record_paths",
            glob_key="train_records_glob",
        )
        train_manifests = load_video_idm_cache_manifests(train_paths, stats=stats, config=prediction_config, split_name="train")
        recalibration = _calibrate_from_cache(
            torch,
            model,
            train_manifests=train_manifests,
            stats=stats,
            config=prediction_config,
            device=device,
        )
        if "category_thresholds" in recalibration:
            prediction_config["category_thresholds"] = recalibration["category_thresholds"]
        if "keyboard_softmax_threshold" in recalibration:
            prediction_config["keyboard_softmax_threshold"] = recalibration["keyboard_softmax_threshold"]
        if "button_softmax_threshold" in recalibration:
            prediction_config["button_softmax_threshold"] = recalibration["button_softmax_threshold"]
        if "mouse_output_gain" in recalibration:
            prediction_config["mouse_output_gain"] = recalibration["mouse_output_gain"]
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if parallel_prediction:
        device = "parallel"
        prediction = _predict_video_checkpoint_parallel(
            prediction_config,
            checkpoint_path=checkpoint_path,
            target_record_paths=target_paths,
            prediction_config_base=prediction_config,
            output_dir=Path(output_dir),
        )
    else:
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        model = _build_video_pair_model(
            torch,
            output_dim=int(checkpoint["output_dim"]),
            aux_dim=int(checkpoint.get("aux_dim", stats.get("aux_dim", 13))),
            config=prediction_config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        target_manifests = load_video_idm_cache_manifests(target_paths, stats=stats, config=prediction_config, split_name="target")
        prediction = _predict_from_cache(
            torch,
            model,
            target_record_paths=target_paths,
            target_manifests=target_manifests,
            stats=stats,
            config=prediction_config,
            device=device,
            output_dir=Path(output_dir),
            checkpoint_path=checkpoint_path,
        )
    summary = {
        "schema": "video_idm_prediction_summary.v1",
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "output_dir": str(output_dir),
        "target_records": int(prediction["target_records"]),
        "prediction_workers": prediction_workers,
        "recalibration": recalibration,
        "prediction": prediction,
        "resolved_prediction_config": {
            key: prediction_config.get(key)
            for key in sorted(_CHECKPOINT_CALIBRATION_KEYS)
            if key in prediction_config
        },
    }
    summary_out = prediction_config.get("prediction_summary_out") or prediction_config.get("summary_out")
    if summary_out:
        write_json(summary_out, summary)
    return summary
