from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.io_utils import ensure_dir, sha256_file, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named
from fdm_d2e.tokenization.actions import add_tokens


HF_DATASET_API = "https://huggingface.co/api/datasets/{repo_id}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"

RAW_MOUSE_BUTTON_FLAGS = {
    0x0001: ("left", "press"),
    0x0002: ("left", "release"),
    0x0004: ("right", "press"),
    0x0008: ("right", "release"),
    0x0010: ("middle", "press"),
    0x0020: ("middle", "release"),
    0x0040: ("x1", "press"),
    0x0080: ("x1", "release"),
}
RAW_MOUSE_WHEEL_FLAG = 0x0400
RAW_MOUSE_HWHEEL_FLAG = 0x0800


@dataclass(frozen=True)
class D2ERecordingRef:
    repo_id: str
    revision: str
    game: str
    recording_id: str
    video_path: str
    mcap_path: str
    video_url: str
    mcap_url: str

    @property
    def pair_id(self) -> str:
        return f"{self.game}/{self.recording_id}"

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "schema": "recording_ref.v1",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "game": self.game,
            "recording_id": self.recording_id,
            "pair_id": self.pair_id,
            "video_path": self.video_path,
            "mcap_path": self.mcap_path,
            "video_url": self.video_url,
            "mcap_url": self.mcap_url,
        }


def _quote_repo_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part) for part in path.split("/"))


def hf_resolve_url(repo_id: str, path: str, revision: str = "main") -> str:
    return HF_RESOLVE.format(
        repo_id=urllib.parse.quote(repo_id, safe="/"),
        revision=urllib.parse.quote(revision, safe=""),
        path=_quote_repo_path(path),
    )


def list_hf_dataset_files(repo_id: str, revision: str = "main", token: str | None = None) -> list[str]:
    """List Hugging Face dataset files without requiring huggingface_hub.

    If huggingface_hub is installed, use it. Otherwise use the public HF REST
    API. This keeps local contract tests lightweight while allowing G2 to pin
    the richer dependency stack for full D2E execution.
    """

    try:
        from huggingface_hub import HfApi  # type: ignore

        return sorted(HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))
    except Exception:
        pass

    url = HF_DATASET_API.format(repo_id=urllib.parse.quote(repo_id, safe="/"))
    if revision and revision != "main":
        url += f"/revision/{urllib.parse.quote(revision, safe='')}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.load(response)
    return sorted(
        sibling["rfilename"]
        for sibling in payload.get("siblings", [])
        if isinstance(sibling, dict) and sibling.get("rfilename")
    )


def build_recording_refs(repo_id: str, files: Iterable[str], revision: str = "main") -> list[D2ERecordingRef]:
    by_stem: dict[str, set[str]] = {}
    paths: dict[tuple[str, str], str] = {}
    for path in files:
        if "/" not in path or "." not in path:
            continue
        stem, ext = path.rsplit(".", 1)
        if ext not in {"mkv", "mcap"}:
            continue
        by_stem.setdefault(stem, set()).add(ext)
        paths[(stem, ext)] = path

    refs: list[D2ERecordingRef] = []
    for stem in sorted(by_stem):
        if not {"mkv", "mcap"} <= by_stem[stem]:
            continue
        game, recording_id = stem.rsplit("/", 1)
        video_path = paths[(stem, "mkv")]
        mcap_path = paths[(stem, "mcap")]
        refs.append(
            D2ERecordingRef(
                repo_id=repo_id,
                revision=revision,
                game=game,
                recording_id=recording_id,
                video_path=video_path,
                mcap_path=mcap_path,
                video_url=hf_resolve_url(repo_id, video_path, revision),
                mcap_url=hf_resolve_url(repo_id, mcap_path, revision),
            )
        )
    return refs


def select_recording_refs(
    refs: list[D2ERecordingRef],
    *,
    max_recordings: int | None = None,
    games: list[str] | None = None,
    recording_ids: list[str] | None = None,
) -> list[D2ERecordingRef]:
    selected = refs
    if games:
        allowed = set(games)
        selected = [ref for ref in selected if ref.game in allowed]
    if recording_ids:
        allowed_ids = set(recording_ids)
        selected = [ref for ref in selected if ref.recording_id in allowed_ids or ref.pair_id in allowed_ids]
    if max_recordings is not None:
        selected = selected[: int(max_recordings)]
    return selected


def split_recordings(
    refs: list[D2ERecordingRef],
    train_fraction: float = 0.8,
    *,
    min_heldout: int = 1,
) -> dict[str, list[D2ERecordingRef]]:
    if not refs:
        return {"train": [], "heldout": []}
    if len(refs) == 1:
        return {"train": refs, "heldout": []}
    train_count = max(1, int(round(len(refs) * train_fraction)))
    train_count = min(train_count, max(1, len(refs) - min_heldout))
    return {"train": refs[:train_count], "heldout": refs[train_count:]}


def _field(decoded: Any, key: str, default: Any = None) -> Any:
    if isinstance(decoded, dict):
        return decoded.get(key, default)
    return getattr(decoded, key, default)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return int(enum_value)
    return default


def _wheel_units(button_data: Any) -> float:
    value = _to_int(button_data, 0)
    if value > 32767:
        value -= 65536
    return value / 120.0 if value else 0.0


def normalize_owa_events(topic: str, decoded: Any, timestamp_ns: int) -> list[dict[str, Any]]:
    """Normalize one decoded OWA message into zero or more training events.

    The function intentionally accepts a generic decoded object/dict so tests can
    cover the contract without importing OWA packages. Full MCAP decoding is
    handled by `decode_mcap_events` when optional dependencies are installed.
    Raw mouse packets may contain both movement and button/scroll state, so this
    plural API preserves all action events at the same timestamp.
    """

    if topic == "keyboard":
        key = _field(decoded, "key") or _field(decoded, "vk") or _field(decoded, "vk_code") or "UNKNOWN"
        event_type = str(_field(decoded, "event_type", "")).lower()
        action = "release" if "release" in event_type or "up" in event_type else "press"
        return [{"type": "keyboard", "event_type": action, "key": str(key), "vk": _field(decoded, "vk", _field(decoded, "vk_code", None)), "timestamp_ns": int(timestamp_ns)}]

    if topic == "mouse/raw":
        button_flags = _to_int(_field(decoded, "button_flags", 0), 0)
        dx = _to_int(_field(decoded, "dx", _field(decoded, "last_x", 0)), 0)
        dy = _to_int(_field(decoded, "dy", _field(decoded, "last_y", 0)), 0)
        rows: list[dict[str, Any]] = []
        if dx or dy or not button_flags:
            rows.append({"type": "mouse_move", "dx": dx, "dy": dy, "timestamp_ns": int(timestamp_ns)})
        for flag, (button, action) in RAW_MOUSE_BUTTON_FLAGS.items():
            if button_flags & flag:
                rows.append({"type": "mouse_button", "button": button, "event_type": action, "timestamp_ns": int(timestamp_ns)})
        if button_flags & RAW_MOUSE_WHEEL_FLAG:
            rows.append({"type": "scroll", "dx": 0.0, "dy": _wheel_units(_field(decoded, "button_data", 0)), "timestamp_ns": int(timestamp_ns)})
        if button_flags & RAW_MOUSE_HWHEEL_FLAG:
            rows.append({"type": "scroll", "dx": _wheel_units(_field(decoded, "button_data", 0)), "dy": 0.0, "timestamp_ns": int(timestamp_ns)})
        return rows

    if topic == "mouse":
        event_type = str(_field(decoded, "event_type", "")).lower()
        if event_type == "scroll":
            return [{"type": "scroll", "dx": float(_field(decoded, "dx", 0) or 0), "dy": float(_field(decoded, "dy", 0) or 0), "timestamp_ns": int(timestamp_ns)}]
        button = _field(decoded, "button", None)
        pressed = _field(decoded, "pressed", None)
        if button and pressed is not None:
            return [{"type": "mouse_button", "button": str(button), "event_type": "press" if pressed else "release", "timestamp_ns": int(timestamp_ns)}]
        dx = _field(decoded, "dx", None)
        dy = _field(decoded, "dy", None)
        if dx is not None or dy is not None:
            return [{"type": "mouse_move", "dx": _to_int(dx, 0), "dy": _to_int(dy, 0), "timestamp_ns": int(timestamp_ns)}]
        return []

    if topic == "screen":
        media_ref = _field(decoded, "media_ref", {}) or {}
        pts_ns = media_ref.get("pts_ns") if isinstance(media_ref, dict) else getattr(media_ref, "pts_ns", None)
        uri = media_ref.get("uri") if isinstance(media_ref, dict) else getattr(media_ref, "uri", None)
        return [{"type": "screen", "timestamp_ns": int(timestamp_ns), "pts_ns": int(pts_ns) if pts_ns is not None else None, "media_uri": uri}]

    return []


def normalize_owa_event(topic: str, decoded: Any, timestamp_ns: int) -> dict[str, Any] | None:
    """Backward-compatible single-event normalizer used by contract tests."""

    rows = normalize_owa_events(topic, decoded, timestamp_ns)
    return rows[0] if rows else None


def decode_mcap_events(mcap_path: str | Path, *, topics: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Decode D2E/OWA MCAP events when optional OWA dependencies are present."""

    try:
        from mcap_owa.highlevel import OWAMcapReader  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only with optional deps absent
        raise RuntimeError(
            "mcap_owa is required to decode real D2E MCAP files; install the d2e/real-data dependency stack first"
        ) from exc

    selected_topics = topics or ["screen", "keyboard", "mouse/raw"]
    rows: list[dict[str, Any]] = []
    with OWAMcapReader(str(mcap_path)) as reader:
        for message in reader.iter_messages(topics=selected_topics):
            for row in normalize_owa_events(message.topic, message.decoded, int(message.timestamp)):
                row["topic"] = message.topic
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def download_recording_ref(
    ref: D2ERecordingRef,
    cache_dir: str | Path,
    token: str | None = None,
    *,
    kinds: Iterable[str] = ("video", "mcap"),
    max_attempts: int = 4,
    retry_backoff_s: float = 1.0,
) -> dict[str, str]:
    """Download a paired recording into cache when requested.

    Full D2E downloads should normally happen on MLXP storage, not the local
    repo. This helper is intended for explicit sample downloads and keeps paths
    outside source-controlled directories by default.
    """

    cache = ensure_dir(cache_dir)
    out_dir = ensure_dir(cache / ref.game)
    results: dict[str, str] = {}
    wanted = set(kinds)
    for kind, rel_path, url in [("video", ref.video_path, ref.video_url), ("mcap", ref.mcap_path, ref.mcap_url)]:
        if kind not in wanted:
            continue
        out = out_dir / Path(rel_path).name
        if not out.exists():
            _download_with_retries(
                url,
                out,
                token=token,
                max_attempts=max_attempts,
                retry_backoff_s=retry_backoff_s,
            )
        results[kind] = str(out)
    return results


def _download_with_retries(
    url: str,
    out: Path,
    *,
    token: str | None = None,
    max_attempts: int = 4,
    retry_backoff_s: float = 1.0,
) -> None:
    """Download ``url`` to ``out`` with bounded retries for transient failures."""

    attempts = max(1, int(max_attempts))
    part = out.with_name(f"{out.name}.part")
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            part.unlink(missing_ok=True)
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            part.replace(out)
            return
        except _TRANSIENT_DOWNLOAD_ERRORS as exc:
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt >= attempts or not _is_transient_download_error(exc):
                raise
            time.sleep(min(retry_backoff_s * (2 ** (attempt - 1)), 30.0))
        except Exception:
            part.unlink(missing_ok=True)
            raise
    if last_exc is not None:  # defensive; loop either returns or raises.
        raise last_exc


_TRANSIENT_DOWNLOAD_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    socket.timeout,
)


def _is_transient_download_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, _TRANSIENT_DOWNLOAD_ERRORS)


def _ppm_features(path: str | Path) -> list[float]:
    data = Path(path).read_bytes()
    cursor = 0

    def next_token() -> bytes:
        nonlocal cursor
        while cursor < len(data) and data[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            return next_token()
        start = cursor
        while cursor < len(data) and data[cursor] not in b" \t\r\n":
            cursor += 1
        return data[start:cursor]

    magic = next_token()
    if magic != b"P6":
        raise ValueError(f"{path} is not a binary PPM frame")
    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
        break
    pixels = data[cursor:]
    expected = width * height * 3
    if len(pixels) < expected:
        raise ValueError(f"{path} has truncated PPM pixel data")
    pixels = pixels[:expected]
    count = width * height
    r_sum = sum(pixels[0::3])
    g_sum = sum(pixels[1::3])
    b_sum = sum(pixels[2::3])
    lumas = [(pixels[i] * 0.2126 + pixels[i + 1] * 0.7152 + pixels[i + 2] * 0.0722) / max_value for i in range(0, expected, 3)]
    luma_mean = sum(lumas) / count
    luma_energy = sum(v * v for v in lumas) / count
    return [r_sum / count / max_value, g_sum / count / max_value, b_sum / count / max_value, luma_mean, luma_energy]


def _ppm_header_and_pixels(path: str | Path) -> tuple[int, int, int, bytes]:
    data = Path(path).read_bytes()
    cursor = 0

    def next_token() -> bytes:
        nonlocal cursor
        while cursor < len(data) and data[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            return next_token()
        start = cursor
        while cursor < len(data) and data[cursor] not in b" \t\r\n":
            cursor += 1
        return data[start:cursor]

    magic = next_token()
    if magic != b"P6":
        raise ValueError(f"{path} is not a binary PPM frame")
    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
        break
    expected = width * height * 3
    pixels = data[cursor : cursor + expected]
    if len(pixels) < expected:
        raise ValueError(f"{path} has truncated PPM pixel data")
    return width, height, max_value, pixels


def _ppm_compact_features(path: str | Path, *, grid_size: int = 8, luma_size: int = 16) -> dict[str, list[float]]:
    """Return summary, RGB-grid, and luma-grid features for one PPM frame.

    The NumPy path is critical for full-corpus D2E throughput.  A pure-Python
    fallback remains for minimal local environments.
    """

    width, height, max_value, pixels = _ppm_header_and_pixels(path)
    try:
        import numpy as np  # type: ignore

        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).astype("float32") / float(max_value)
        rgb_mean = arr.mean(axis=(0, 1))
        luma = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
        summary = [
            float(rgb_mean[0]),
            float(rgb_mean[1]),
            float(rgb_mean[2]),
            float(luma.mean()),
            float((luma * luma).mean()),
        ]
        if width % grid_size == 0 and height % grid_size == 0:
            grid = arr.reshape(grid_size, height // grid_size, grid_size, width // grid_size, 3).mean(axis=(1, 3))
            grid_values = grid.reshape(grid_size * grid_size * 3).astype("float32").tolist()
        else:
            grid_values = _ppm_grid_luma_features_python(width, height, max_value, pixels, grid_size=grid_size, luma_size=luma_size)[f"grid{grid_size}"]
        if width % luma_size == 0 and height % luma_size == 0:
            luma_grid = luma.reshape(luma_size, height // luma_size, luma_size, width // luma_size).mean(axis=(1, 3))
            luma_values = luma_grid.reshape(luma_size * luma_size).astype("float32").tolist()
        else:
            luma_values = _ppm_grid_luma_features_python(width, height, max_value, pixels, grid_size=grid_size, luma_size=luma_size)[f"luma{luma_size}"]
        return {"features": summary, f"grid{grid_size}": grid_values, f"luma{luma_size}": luma_values}
    except Exception:
        rgb_luma = _ppm_features(path)
        grid_luma = _ppm_grid_luma_features_python(width, height, max_value, pixels, grid_size=grid_size, luma_size=luma_size)
        return {"features": rgb_luma, **grid_luma}


def _ppm_grid_luma_features_python(
    width: int,
    height: int,
    max_value: int,
    pixels: bytes,
    *,
    grid_size: int = 8,
    luma_size: int = 16,
) -> dict[str, list[float]]:
    expected = width * height * 3
    if len(pixels) < expected:
        raise ValueError("truncated PPM pixel data")
    pixels = pixels[:expected]
    grid_sums = [[0.0, 0.0, 0.0, 0.0] for _ in range(grid_size * grid_size)]
    luma_sums = [[0.0, 0.0] for _ in range(luma_size * luma_size)]
    for y in range(height):
        gy = min(grid_size - 1, y * grid_size // height)
        ly = min(luma_size - 1, y * luma_size // height)
        for x in range(width):
            gx = min(grid_size - 1, x * grid_size // width)
            lx = min(luma_size - 1, x * luma_size // width)
            base = (y * width + x) * 3
            r = pixels[base] / max_value
            g = pixels[base + 1] / max_value
            b = pixels[base + 2] / max_value
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            grid_bucket = grid_sums[gy * grid_size + gx]
            grid_bucket[0] += r
            grid_bucket[1] += g
            grid_bucket[2] += b
            grid_bucket[3] += 1.0
            luma_bucket = luma_sums[ly * luma_size + lx]
            luma_bucket[0] += luma
            luma_bucket[1] += 1.0
    grid: list[float] = []
    for r, g, b, count in grid_sums:
        denom = count or 1.0
        grid.extend([r / denom, g / denom, b / denom])
    luma = [total / (count or 1.0) for total, count in luma_sums]
    return {f"grid{grid_size}": grid, f"luma{luma_size}": luma}


def _ppm_grid_luma_features(path: str | Path, *, grid_size: int = 8, luma_size: int = 16) -> dict[str, list[float]]:
    """Extract compact RGB-grid and luma-grid features from a PPM frame.

    Full-corpus D2E runs cannot keep every decoded 64×64 frame as a long-lived
    training artifact.  These features preserve enough spatial signal for the
    repo-native IDM feature modes while allowing extraction jobs to delete
    transient PPM files after each recording.
    """

    width, height, max_value, pixels = _ppm_header_and_pixels(path)
    return _ppm_grid_luma_features_python(width, height, max_value, pixels, grid_size=grid_size, luma_size=luma_size)


def extract_video_frame_features(
    video_source: str,
    output_dir: str | Path,
    *,
    max_frames: int | None = 16,
    fps: int = 20,
    image_size: int = 64,
    start_seconds: float = 0.0,
    compact_features: bool = False,
    keep_frames: bool = True,
) -> list[dict[str, Any]]:
    """Extract small real-video frame features through ffmpeg without storing raw D2E frames in git."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for real D2E video feature extraction")
    frame_dir = ensure_dir(Path(output_dir) / "frames_ppm")
    for old in frame_dir.glob("frame_*.ppm"):
        old.unlink()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        video_source,
        "-vf",
        f"fps={int(fps)},scale={int(image_size)}:{int(image_size)}",
        "-y",
    ]
    if max_frames is not None and int(max_frames) > 0:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.append(str(frame_dir / "frame_%06d.ppm"))
    subprocess.run(cmd, check=True)
    rows: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(sorted(frame_dir.glob("frame_*.ppm"))):
        compact = _ppm_compact_features(frame_path, grid_size=8, luma_size=16) if compact_features else {"features": _ppm_features(frame_path)}
        rows.append(
            {
                "frame_index": idx,
                "path": str(frame_path) if keep_frames else f"{video_source}#frame={idx}",
                **compact,
                "source_video": video_source,
            }
        )
        if not keep_frames:
            frame_path.unlink(missing_ok=True)
    if not rows:
        raise RuntimeError(f"ffmpeg produced no frames from {video_source}")
    return rows


def build_window_records(
    ref: D2ERecordingRef,
    decoded_events: list[dict[str, Any]],
    *,
    split: str = "sample",
    bin_ms: int = 50,
    max_bins: int | None = None,
    frame_features: list[dict[str, Any]] | None = None,
    start_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Bin decoded MCAP action events into D2E-style 50 ms training records."""

    screen_times = sorted(int(row["timestamp_ns"]) for row in decoded_events if row.get("type") == "screen")
    all_times = sorted(int(row["timestamp_ns"]) for row in decoded_events)
    if not all_times:
        return []
    start_ns = int(start_ns if start_ns is not None else (screen_times[0] if screen_times else all_times[0]))
    bin_ns = int(bin_ms) * 1_000_000
    if frame_features:
        num_bins = len(frame_features)
    else:
        last_ns = max(all_times)
        num_bins = int((last_ns - start_ns) // bin_ns) + 1
    if max_bins is not None:
        num_bins = min(num_bins, int(max_bins))
    action_bins: dict[int, list[dict[str, Any]]] = {}
    for row in decoded_events:
        if row.get("type") == "screen":
            continue
        bin_index = int((int(row["timestamp_ns"]) - start_ns) // bin_ns)
        if bin_index < 0 or bin_index >= num_bins:
            continue
        action_bins.setdefault(bin_index, []).append({k: v for k, v in row.items() if k != "topic"})
    records: list[dict[str, Any]] = []
    for bin_index in range(max(0, num_bins)):
        bin_start = start_ns + bin_index * bin_ns
        events = list(action_bins.get(bin_index, ()))
        frame_row = frame_features[bin_index] if frame_features and bin_index < len(frame_features) else {}
        record = {
            "schema": "d2e_window_record.v1",
            "sequence_id": f"{ref.pair_id}#{bin_index:06d}",
            "recording_id": ref.recording_id,
            "game": ref.game,
            "split": split,
            "timestamp_ns": bin_start,
            "bin_index": bin_index,
            "frame": {
                "path": frame_row.get("path", ref.video_path),
                "index": int(frame_row.get("frame_index", bin_index)),
                "features": list(frame_row.get("features", [])),
                **{key: list(frame_row[key]) for key in ("grid8", "luma16") if key in frame_row},
            },
            "events": events,
            "source": "real_d2e_decoded_mcap_video",
        }
        records.append(record)
    for idx, row in enumerate(records):
        current_features = list(row.get("frame", {}).get("features", []))
        next_features = list(records[idx + 1].get("frame", {}).get("features", [])) if idx + 1 < len(records) else current_features
        dims = min(len(current_features), len(next_features))
        row["next_frame_features"] = next_features
        row["frame_delta_features"] = [next_features[i] - current_features[i] for i in range(dims)]
        for key in ("grid8", "luma16"):
            current_values = list(row.get("frame", {}).get(key, []))
            next_values = list(records[idx + 1].get("frame", {}).get(key, [])) if idx + 1 < len(records) else current_values
            if current_values and next_values:
                row[f"next_frame_{key}"] = next_values
    tokenized = add_tokens(records)
    for row in tokenized:
        validate_named(row, "d2e_window_record.schema.json")
    return tokenized


def choose_action_dense_window_start(decoded_events: list[dict[str, Any]], *, duration_ns: int) -> int | None:
    """Pick a window start that contains real actions instead of a no-op-only prefix."""

    action_times = sorted(int(row["timestamp_ns"]) for row in decoded_events if row.get("type") != "screen")
    if not action_times:
        return None
    best_start = action_times[0]
    best_count = -1
    right = 0
    for left, start in enumerate(action_times):
        while right < len(action_times) and action_times[right] < start + duration_ns:
            right += 1
        count = right - left
        if count > best_count:
            best_start = start
            best_count = count
    return best_start


def build_real_manifests(config: dict[str, Any], *, files: list[str] | None = None) -> dict[str, Any]:
    repo_id = str(config.get("hf_repo_id", "open-world-agents/D2E-480p"))
    revision = str(config.get("revision", "main"))
    token = config.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    all_files = files if files is not None else list_hf_dataset_files(repo_id, revision=revision, token=token)
    refs = build_recording_refs(repo_id, all_files, revision=revision)
    selected = select_recording_refs(
        refs,
        max_recordings=config.get("max_recordings"),
        games=list(config.get("games", [])) or None,
        recording_ids=list(config.get("recording_ids", [])) or None,
    )
    splits = split_recordings(selected, float(config.get("train_fraction", 0.8)), min_heldout=int(config.get("min_heldout", 1)))
    split_rows = {name: [ref.pair_id for ref in values] for name, values in splits.items()}
    manifest = {
        "schema": "data_manifest.v2",
        "dataset": "D2E",
        "source_mode": "real_d2e_hf_manifest",
        "hf_repo_id": repo_id,
        "revision": revision,
        "license": "cc-by-nc-4.0",
        "source_contract": {
            "paired_video_mcap": True,
            "video_ext": ".mkv",
            "mcap_ext": ".mcap",
            "timestamp_unit": "nanoseconds",
            "default_bin_ms": int(config.get("bin_ms", 50)),
            "event_topics": ["screen", "keyboard", "mouse/raw"],
            "official_eval_reference": "worv-ai/D2E evaluate.py",
        },
        "recordings": [ref.to_manifest_row() for ref in selected],
        "splits": {name: len(values) for name, values in split_rows.items()},
        "event_categories": ["keyboard", "mouse_move", "mouse_button", "scroll"],
        "dataset_fingerprint": stable_hash_json([ref.to_manifest_row() for ref in selected]),
    }
    recording_manifest = {
        "schema": "recording_manifest.v1",
        "dataset": "D2E",
        "hf_repo_id": repo_id,
        "revision": revision,
        "num_recordings": len(selected),
        "recordings": [ref.to_manifest_row() for ref in selected],
    }
    split_manifest = {
        "schema": "split_manifest.v1",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "split_policy": {
            "method": "deterministic_ordered_recording_split",
            "train_fraction": float(config.get("train_fraction", 0.8)),
            "min_heldout": int(config.get("min_heldout", 1)),
        },
        "splits": split_rows,
    }
    split_by_pair = {pair_id: split_name for split_name, pair_ids in split_rows.items() for pair_id in pair_ids}
    sequence_pack = {
        "schema": "sequence_pack.v2",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "timebase": {"timestamp_unit": "nanoseconds", "bin_ms": int(config.get("bin_ms", 50))},
        "sequences": [
            {
                "sequence_id": ref.pair_id,
                "recording_id": ref.recording_id,
                "game": ref.game,
                "split": split_by_pair.get(ref.pair_id, "unknown"),
                "frame_source": {"type": "mkv", "path": ref.video_path, "url": ref.video_url},
                "event_source": {"type": "mcap", "path": ref.mcap_path, "url": ref.mcap_url},
                "decoded_events_path": None,
            }
            for ref in selected
        ],
    }
    validate_named(manifest, "data_manifest_v2.schema.json")
    validate_named(recording_manifest, "recording_manifest.schema.json")
    validate_named(split_manifest, "split_manifest.schema.json")
    validate_named(sequence_pack, "sequence_pack_v2.schema.json")
    return {"manifest": manifest, "recording_manifest": recording_manifest, "split_manifest": split_manifest, "sequence_pack": sequence_pack}


def prepare_real_dataset(config: dict[str, Any], *, files: list[str] | None = None) -> dict[str, Any]:
    output_dir = Path(config.get("output_dir", "outputs"))
    data_dir = ensure_dir(output_dir / "data")
    prepared = build_real_manifests(config, files=files)
    write_json(data_dir / "manifest.v2.json", prepared["manifest"])
    write_json(data_dir / "recording_manifest.json", prepared["recording_manifest"])
    write_json(data_dir / "split_manifest.json", prepared["split_manifest"])
    write_json(data_dir / "sample_sequence_pack.v2.json", prepared["sequence_pack"])
    write_jsonl(data_dir / "recordings.jsonl", prepared["recording_manifest"]["recordings"])
    return prepared


def prepare_decoded_sample(config: dict[str, Any], *, files: list[str] | None = None) -> dict[str, Any]:
    """Download/decode one real D2E sample pair into ignored training artifacts.

    This is the G1 bridge between source-only manifests and actual D2E training
    examples: it decodes MCAP actions, extracts video frame features from the
    paired MKV stream, bins actions to the configured timebase, tokenizes them,
    and writes v2 sequence-pack plus train/heldout JSONL artifacts.
    """

    prepared = build_real_manifests(config, files=files)
    refs = [D2ERecordingRef(**{k: row[k] for k in ["repo_id", "revision", "game", "recording_id", "video_path", "mcap_path", "video_url", "mcap_url"]}) for row in prepared["recording_manifest"]["recordings"]]
    if not refs:
        raise ValueError("No paired D2E recording references selected for decoded sample")
    ref = refs[0]
    output_dir = Path(config.get("output_dir", "outputs"))
    sample_root = str(config.get("sample_root", "real_sample"))
    sample_dir = ensure_dir(output_dir / "data" / sample_root / ref.game / ref.recording_id)
    cache_dir = Path(config.get("cache_dir", "/tmp/fdm-d2e-sample-cache"))
    downloaded = download_recording_ref(ref, cache_dir, kinds=("mcap",))
    event_limit = config.get("event_limit")
    decoded_events = decode_mcap_events(downloaded["mcap"], limit=int(event_limit) if event_limit is not None else None)
    bin_ms = int(config.get("bin_ms", 50))
    max_bins = int(config.get("max_bins", int(config.get("max_frames", 32))))
    window_duration_ns = max_bins * bin_ms * 1_000_000
    configured_start = config.get("window_start_ns")
    window_start_ns = int(configured_start) if configured_start is not None else choose_action_dense_window_start(decoded_events, duration_ns=window_duration_ns)
    if window_start_ns is None:
        window_start_ns = min(int(row["timestamp_ns"]) for row in decoded_events) if decoded_events else 0
    video_start_seconds = float(config.get("video_start_seconds", window_start_ns / 1_000_000_000))
    video_source = str(config.get("video_source") or ref.video_url)
    frame_kwargs = {
        "max_frames": int(config.get("max_frames", 32)),
        "fps": int(config.get("frame_fps", max(1, round(1000 / bin_ms)))),
        "image_size": int(config.get("image_size", 64)),
        "start_seconds": video_start_seconds,
    }
    try:
        frame_features = extract_video_frame_features(video_source, sample_dir, **frame_kwargs)
    except subprocess.CalledProcessError:
        if config.get("download_video_on_ffmpeg_failure", True) is False or video_source != ref.video_url:
            raise
        video_download = download_recording_ref(
            ref,
            cache_dir,
            token=config.get("hf_token") or os.environ.get("HF_TOKEN"),
            kinds=("video",),
        )
        frame_features = extract_video_frame_features(video_download["video"], sample_dir, **frame_kwargs)
    records = build_window_records(
        ref,
        decoded_events,
        split="sample",
        bin_ms=bin_ms,
        max_bins=max_bins,
        frame_features=frame_features,
        start_ns=window_start_ns,
    )
    train_count = max(1, int(round(len(records) * float(config.get("train_fraction", 0.75))))) if len(records) > 1 else len(records)
    train_count = min(train_count, max(1, len(records) - int(config.get("min_heldout", 1)))) if len(records) > 1 else train_count
    for idx, row in enumerate(records):
        row["split"] = "train" if idx < train_count else "heldout"
    train = [row for row in records if row["split"] == "train"]
    heldout = [row for row in records if row["split"] == "heldout"]
    dataset_fingerprint = stable_hash_json(
        {
            "ref": ref.to_manifest_row(),
            "mcap_sha256": sha256_file(downloaded["mcap"]),
            "num_decoded_events": len(decoded_events),
            "num_records": len(records),
            "bin_ms": bin_ms,
            "window_start_ns": window_start_ns,
        }
    )
    sequence_pack = {
        "schema": "sequence_pack.v2",
        "dataset_fingerprint": dataset_fingerprint,
        "timebase": {"timestamp_unit": "nanoseconds", "bin_ms": bin_ms, "window_start_ns": window_start_ns},
        "sequences": [
            {
                "sequence_id": row["sequence_id"],
                "recording_id": row["recording_id"],
                "game": row["game"],
                "split": row["split"],
                "timestamp_ns": row["timestamp_ns"],
                "bin_index": row["bin_index"],
                "frame_features": row["frame"]["features"],
                "ground_truth_tokens": row["ground_truth_tokens"],
                "decoded_events_path": str(sample_dir / "decoded_events.jsonl"),
                "frame_source": {"type": "mkv", "url": ref.video_url, "feature_path": row["frame"]["path"]},
                "event_source": {"type": "mcap", "path": downloaded["mcap"], "url": ref.mcap_url},
            }
            for row in records
        ],
    }
    validate_named(sequence_pack, "sequence_pack_v2.schema.json")
    write_jsonl(sample_dir / "decoded_events.jsonl", decoded_events)
    write_jsonl(sample_dir / "frame_features.jsonl", frame_features)
    write_jsonl(sample_dir / "all_records.jsonl", records)
    write_jsonl(sample_dir / "train.jsonl", train)
    write_jsonl(sample_dir / "heldout.jsonl", heldout)
    write_json(sample_dir / "sequence_pack.v2.json", sequence_pack)
    summary = {
        "schema": "d2e_decoded_sample_summary.v1",
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "pair_id": ref.pair_id,
        "mcap_path": downloaded["mcap"],
        "mcap_sha256": sha256_file(downloaded["mcap"]),
        "video_url": ref.video_url,
        "output_dir": str(sample_dir),
        "num_decoded_events": len(decoded_events),
        "num_frame_features": len(frame_features),
        "num_window_records": len(records),
        "window_start_ns": window_start_ns,
        "video_start_seconds": video_start_seconds,
        "splits": {"train": len(train), "heldout": len(heldout)},
        "event_types": sorted({str(row.get("type")) for row in decoded_events}),
        "token_fingerprint": stable_hash_json([row.get("ground_truth_tokens", []) for row in records]),
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(sample_dir / "decode_summary.json", summary)
    return {"summary": summary, "sequence_pack": sequence_pack, "records": records, "decoded_events": decoded_events}
