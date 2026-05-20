from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.io_utils import ensure_dir, stable_hash_json, write_json, write_jsonl
from fdm_d2e.schema import validate_named


HF_DATASET_API = "https://huggingface.co/api/datasets/{repo_id}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


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


def normalize_owa_event(topic: str, decoded: Any, timestamp_ns: int) -> dict[str, Any] | None:
    """Normalize decoded OWA desktop events into this repo's event shape.

    The function intentionally accepts a generic decoded object/dict so tests can
    cover the contract without importing OWA packages. Full MCAP decoding is
    handled by `decode_mcap_events` when optional dependencies are installed.
    """

    if isinstance(decoded, dict):
        get = decoded.get
    else:
        get = lambda key, default=None: getattr(decoded, key, default)

    if topic == "keyboard":
        key = get("key") or get("vk") or get("vk_code") or "UNKNOWN"
        event_type = str(get("event_type", "")).lower()
        action = "release" if "release" in event_type or "up" in event_type else "press"
        return {"type": "keyboard", "event_type": action, "key": str(key), "vk": get("vk", get("vk_code", None)), "timestamp_ns": int(timestamp_ns)}

    if topic == "mouse/raw":
        button_flags = int(get("button_flags", 0) or 0)
        dx = int(get("dx", 0) or 0)
        dy = int(get("dy", 0) or 0)
        if button_flags:
            # Preserve raw flags; later action-token work can map OWA flags to
            # left/right/middle down/up with the owa-msgs enum available.
            return {"type": "mouse_button", "event_type": "raw_flags", "button_flags": button_flags, "dx": dx, "dy": dy, "timestamp_ns": int(timestamp_ns)}
        return {"type": "mouse_move", "dx": dx, "dy": dy, "timestamp_ns": int(timestamp_ns)}

    if topic == "screen":
        media_ref = get("media_ref", {}) or {}
        pts_ns = media_ref.get("pts_ns") if isinstance(media_ref, dict) else getattr(media_ref, "pts_ns", None)
        return {"type": "screen", "timestamp_ns": int(timestamp_ns), "pts_ns": int(pts_ns) if pts_ns is not None else None}

    return None


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
            row = normalize_owa_event(message.topic, message.decoded, int(message.timestamp))
            if row is not None:
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def download_recording_ref(ref: D2ERecordingRef, cache_dir: str | Path, token: str | None = None) -> dict[str, str]:
    """Download a paired recording into cache when requested.

    Full D2E downloads should normally happen on MLXP storage, not the local
    repo. This helper is intended for explicit sample downloads and keeps paths
    outside source-controlled directories by default.
    """

    cache = ensure_dir(cache_dir)
    out_dir = ensure_dir(cache / ref.game)
    results: dict[str, str] = {}
    for kind, rel_path, url in [("video", ref.video_path, ref.video_url), ("mcap", ref.mcap_path, ref.mcap_url)]:
        out = out_dir / Path(rel_path).name
        if not out.exists():
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response, out.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        results[kind] = str(out)
    return results


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
