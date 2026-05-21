from __future__ import annotations

import datetime as _dt
import json
import os
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from fdm_d2e.io_utils import stable_hash_json
from fdm_d2e.schema import validate_named


HF_DATASET_API = "https://huggingface.co/api/datasets/{repo_id}"
HF_DATASET_TREE_API = "https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=true"
TIB = 1024**4


@dataclass(frozen=True)
class DataSourceConfig:
    source_id: str
    repo_id: str
    revision: str
    resolution_tier: str
    required_for_full_success: bool = True


DEFAULT_D2E_SOURCES = [
    DataSourceConfig(
        source_id="d2e_480p",
        repo_id="open-world-agents/D2E-480p",
        revision="main",
        resolution_tier="480p",
    ),
    DataSourceConfig(
        source_id="d2e_original",
        repo_id="open-world-agents/D2E-Original",
        revision="main",
        resolution_tier="original_fhd_qhd",
    ),
]


DEFAULT_AUXILIARY_CANDIDATES = [
    {
        "source_id": "atari_head",
        "name": "Atari-HEAD",
        "source_url": "https://zenodo.org/records/2587121",
        "license": "CC-BY-4.0",
        "selection_status": "candidate_needs_integration_review",
        "game_domain": "Atari",
        "supervision": "frame-level human keystroke action labels with gaze/reward metadata",
        "estimated_size_bytes": 8_130_000_000,
        "notes": "Public human demonstration dataset; action space is Atari/ALE, not desktop keyboard/mouse.",
    },
    {
        "source_id": "atari_assault_hf",
        "name": "p-doom/atari-assault-dataset",
        "repo_id": "p-doom/atari-assault-dataset",
        "source_url": "https://huggingface.co/datasets/p-doom/atari-assault-dataset",
        "license": "cc0-1.0",
        "selection_status": "candidate_needs_integration_review",
        "game_domain": "Atari Assault",
        "supervision": "action-conditioned video prediction frames/actions",
        "estimated_size_bytes": None,
        "notes": "Useful as a small open-license action-conditioned game-video candidate; not D2E-style desktop input.",
    },
    {
        "source_id": "nethack_learning_dataset",
        "name": "NetHack Learning Dataset",
        "source_url": "https://openreview.net/forum?id=zHNNSzo10xN",
        "license": "needs_review",
        "selection_status": "candidate_needs_license_review",
        "game_domain": "NetHack",
        "supervision": "state-action trajectories/key actions",
        "estimated_size_bytes": None,
        "notes": "Large action-labeled game trajectory candidate; terminal/symbolic observations differ from D2E video.",
    },
    {
        "source_id": "minerl",
        "name": "MineRL",
        "source_url": "https://minerl.io/",
        "license": "needs_review",
        "selection_status": "candidate_needs_license_review",
        "game_domain": "Minecraft",
        "supervision": "human gameplay trajectories with actions/rewards",
        "estimated_size_bytes": None,
        "notes": "3D game trajectory candidate; integration depends on current dataset availability and license review.",
    },
]


def _quote(value: str, *, safe: str = "") -> str:
    return urllib.parse.quote(value, safe=safe)


def _read_json_url(url: str, *, token: str | None = None, timeout: int = 60) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def fetch_hf_dataset_info(repo_id: str, *, revision: str = "main", token: str | None = None) -> dict[str, Any]:
    del revision  # The repo info endpoint returns the current resolved sha.
    url = HF_DATASET_API.format(repo_id=_quote(repo_id, safe="/"))
    payload = _read_json_url(url, token=token)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object from {url}")
    return payload


def fetch_hf_dataset_tree(repo_id: str, *, revision: str = "main", token: str | None = None) -> list[dict[str, Any]]:
    url = HF_DATASET_TREE_API.format(repo_id=_quote(repo_id, safe="/"), revision=_quote(revision, safe=""))
    payload = _read_json_url(url, token=token)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list from {url}")
    return [row for row in payload if isinstance(row, dict)]


def _license_from_info(info: dict[str, Any]) -> str:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    license_value = card.get("license")
    if isinstance(license_value, str) and license_value:
        return license_value
    for tag in info.get("tags", []) or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return "unknown"


def _file_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
    return {
        "path": str(entry.get("path", "")),
        "size_bytes": int(entry.get("size") or lfs.get("size") or 0),
        "git_oid": entry.get("oid"),
        "lfs_sha256": lfs.get("oid"),
        "xet_hash": entry.get("xetHash"),
    }


def _pair_files(tree_entries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    by_stem: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ignored: list[dict[str, Any]] = []
    for entry in tree_entries:
        if entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        if "/" not in path or "." not in path:
            ignored.append({"path": path, "reason": "not_recording_file"})
            continue
        stem, ext = path.rsplit(".", 1)
        if ext not in {"mkv", "mcap"}:
            ignored.append({"path": path, "reason": "unsupported_extension"})
            continue
        by_stem[stem][ext] = _file_metadata(entry)
    return by_stem, ignored


def build_d2e_source_inventory(
    source: DataSourceConfig,
    *,
    repo_info: dict[str, Any],
    tree_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    by_stem, ignored_files = _pair_files(tree_entries)
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    license_value = _license_from_info(repo_info)
    total_file_size = sum(int(row.get("size") or 0) for row in tree_entries if row.get("type") == "file")
    for stem in sorted(by_stem):
        files = by_stem[stem]
        if "/" not in stem:
            continue
        game, recording_id = stem.rsplit("/", 1)
        has_pair = {"mkv", "mcap"} <= set(files)
        status = "included" if has_pair else "unsupported"
        reason = "paired_video_mcap_present" if has_pair else "missing_video_or_mcap_pair"
        size_bytes = sum(int(meta.get("size_bytes") or 0) for meta in files.values())
        row = {
            "schema": "data_universe_recording.v1",
            "dataset_family": "D2E",
            "source_id": source.source_id,
            "repo_id": source.repo_id,
            "requested_revision": source.revision,
            "resolved_revision": repo_info.get("sha"),
            "resolution_tier": source.resolution_tier,
            "required_for_full_success": source.required_for_full_success,
            "license": license_value,
            "game": game,
            "recording_id": recording_id,
            "source_recording_key": f"{game}/{recording_id}",
            "cross_resolution_key": f"{game}/{recording_id}",
            "status": status,
            "status_reason": reason,
            "files": {
                "video": files.get("mkv"),
                "mcap": files.get("mcap"),
            },
            "size_bytes": size_bytes,
            "audited_exclusion": None if status == "included" else {"retry_log_path": None, "reason": reason, "impact": "not usable until paired"},
        }
        status_counts[status] += 1
        rows.append(row)
    games = sorted({row["game"] for row in rows})
    included = [row for row in rows if row["status"] == "included"]
    source_summary = {
        "schema": "data_universe_source.v1",
        "source_id": source.source_id,
        "repo_id": source.repo_id,
        "requested_revision": source.revision,
        "resolved_revision": repo_info.get("sha"),
        "last_modified": repo_info.get("lastModified"),
        "license": license_value,
        "resolution_tier": source.resolution_tier,
        "required_for_full_success": source.required_for_full_success,
        "private": bool(repo_info.get("private", False)),
        "gated": bool(repo_info.get("gated", False)),
        "disabled": bool(repo_info.get("disabled", False)),
        "used_storage_bytes": int(repo_info.get("usedStorage") or 0),
        "tree_file_count": sum(1 for row in tree_entries if row.get("type") == "file"),
        "tree_directory_count": sum(1 for row in tree_entries if row.get("type") == "directory"),
        "total_file_size_bytes": total_file_size,
        "recording_variants": len(rows),
        "paired_recordings": len(included),
        "games_count": len(games),
        "games": games,
        "status_counts": dict(status_counts),
        "ignored_files": ignored_files,
    }
    return {"source": source_summary, "recordings": rows}


def _storage_budget_report(
    *,
    d2e_total_bytes: int,
    aux_planned_bytes: int,
    budget_tib: float,
) -> dict[str, Any]:
    budget_bytes = int(budget_tib * TIB)
    planned_bytes = d2e_total_bytes + aux_planned_bytes
    return {
        "schema": "storage_budget.v1",
        "budget_tib": budget_tib,
        "budget_bytes": budget_bytes,
        "d2e_source_total_bytes": d2e_total_bytes,
        "auxiliary_planned_bytes": aux_planned_bytes,
        "total_planned_source_bytes": planned_bytes,
        "total_source_bytes_within_budget": planned_bytes <= budget_bytes,
        "working_set_policy": "Use staged/streaming cache if source total exceeds the 5TiB working-set limit; do not silently drop required D2E sources.",
        "requires_staged_cache_or_extra_storage": planned_bytes > budget_bytes,
    }


def build_data_universe_manifest(
    *,
    sources: list[DataSourceConfig] | None = None,
    repo_infos: dict[str, dict[str, Any]] | None = None,
    repo_trees: dict[str, list[dict[str, Any]]] | None = None,
    auxiliary_candidates: list[dict[str, Any]] | None = None,
    budget_tib: float = 5.0,
    token: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    sources = sources or DEFAULT_D2E_SOURCES
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    d2e_sources: list[dict[str, Any]] = []
    recordings: list[dict[str, Any]] = []
    for source in sources:
        info = (repo_infos or {}).get(source.repo_id) or fetch_hf_dataset_info(source.repo_id, revision=source.revision, token=token)
        tree = (repo_trees or {}).get(source.repo_id) or fetch_hf_dataset_tree(source.repo_id, revision=source.revision, token=token)
        built = build_d2e_source_inventory(source, repo_info=info, tree_entries=tree)
        d2e_sources.append(built["source"])
        recordings.extend(built["recordings"])

    status_counts = Counter(row["status"] for row in recordings)
    cross_resolution: dict[str, set[str]] = defaultdict(set)
    for row in recordings:
        cross_resolution[row["cross_resolution_key"]].add(row["source_id"])
    d2e_total_bytes = sum(int(source.get("total_file_size_bytes") or 0) for source in d2e_sources)
    aux = auxiliary_candidates or DEFAULT_AUXILIARY_CANDIDATES
    aux_planned_bytes = sum(int(row.get("planned_size_bytes") or 0) for row in aux)
    coverage = {
        "schema": "data_universe_coverage.v1",
        "d2e_source_count": len(d2e_sources),
        "recording_variants": len(recordings),
        "unique_cross_resolution_recordings": len(cross_resolution),
        "status_counts": dict(status_counts),
        "games_count": len({row["game"] for row in recordings}),
        "games": sorted({row["game"] for row in recordings}),
        "all_recording_variants_statused": len(recordings) == sum(status_counts.values()) and bool(recordings),
        "included_recording_variants": status_counts.get("included", 0),
        "unsupported_recording_variants": status_counts.get("unsupported", 0),
        "cross_resolution_source_counts": Counter(len(values) for values in cross_resolution.values()),
    }
    # JSON requires string keys.
    coverage["cross_resolution_source_counts"] = {str(k): v for k, v in coverage["cross_resolution_source_counts"].items()}
    manifest = {
        "schema": "data_universe_manifest.v1",
        "generated_at_utc": generated_at_utc or _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "d2e_sources": d2e_sources,
        "recordings": recordings,
        "coverage": coverage,
        "storage_budget": _storage_budget_report(d2e_total_bytes=d2e_total_bytes, aux_planned_bytes=aux_planned_bytes, budget_tib=budget_tib),
        "auxiliary_candidates": aux,
        "decision_gates": {
            "inventory_status_coverage_required": "100%",
            "required_status_values": ["included", "excluded", "unavailable", "corrupt", "unsupported"],
            "excluded_items_require_retry_log_reason_impact": True,
            "full_success_requires_sources": [source.source_id for source in sources if source.required_for_full_success],
            "d2e_only_claims_must_precede_aux_headline_claims": True,
        },
        "external_references": [
            "https://worv-ai.github.io/d2e/",
            "https://huggingface.co/datasets/open-world-agents/D2E-480p",
            "https://huggingface.co/datasets/open-world-agents/D2E-Original",
        ],
    }
    manifest["dataset_fingerprint"] = stable_hash_json(
        {
            "d2e_sources": d2e_sources,
            "recordings": recordings,
            "auxiliary_candidates": aux,
            "storage_budget": manifest["storage_budget"],
            "decision_gates": manifest["decision_gates"],
        }
    )
    validate_data_universe_manifest(manifest)
    return manifest


def validate_data_universe_manifest(manifest: dict[str, Any]) -> None:
    validate_named(manifest, "data_universe_manifest.schema.json")
    allowed = set(manifest.get("decision_gates", {}).get("required_status_values", [])) or {"included", "excluded", "unavailable", "corrupt", "unsupported"}
    bad = [row for row in manifest.get("recordings", []) if row.get("status") not in allowed]
    if bad:
        raise ValueError(f"Found {len(bad)} rows with invalid status")
    for row in manifest.get("recordings", []):
        if row.get("status") != "included" and not row.get("audited_exclusion"):
            raise ValueError(f"Non-included row lacks audited_exclusion: {row.get('source_recording_key')}")
    coverage = manifest.get("coverage", {})
    if not coverage.get("all_recording_variants_statused"):
        raise ValueError("Coverage gate failed: not all recording variants are statused")
