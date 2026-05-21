from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from fdm_d2e.data.d2e_real import D2ERecordingRef, hf_resolve_url
from fdm_d2e.io_utils import write_jsonl


def universe_row_id(row: dict[str, Any]) -> str:
    return f"{row.get('source_id')}:{row.get('cross_resolution_key')}"


def included_universe_rows(
    manifest: dict[str, Any],
    *,
    source_ids: Iterable[str] | None = None,
    resolution_tiers: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_sources = set(source_ids or [])
    allowed_tiers = set(resolution_tiers or [])
    rows = []
    for row in manifest.get("recordings", []):
        if row.get("status") != "included":
            continue
        if allowed_sources and str(row.get("source_id")) not in allowed_sources:
            continue
        if allowed_tiers and str(row.get("resolution_tier")) not in allowed_tiers:
            continue
        rows.append(row)
    return sorted(rows, key=lambda item: (str(item.get("source_id")), str(item.get("cross_resolution_key"))))


def d2e_ref_from_universe_row(row: dict[str, Any]) -> D2ERecordingRef:
    repo_id = str(row["repo_id"])
    revision = str(row.get("resolved_revision") or row.get("requested_revision") or "main")
    video_path = str(row["files"]["video"]["path"])
    mcap_path = str(row["files"]["mcap"]["path"])
    return D2ERecordingRef(
        repo_id=repo_id,
        revision=revision,
        game=str(row["game"]),
        recording_id=str(row["recording_id"]),
        video_path=video_path,
        mcap_path=mcap_path,
        video_url=hf_resolve_url(repo_id, video_path, revision),
        mcap_url=hf_resolve_url(repo_id, mcap_path, revision),
    )


def _temporal_fraction(split_contract: dict[str, Any]) -> float:
    policy = split_contract["manifests"]["temporal"]["split_policy"]
    return float(policy.get("train_fraction", 0.8))


def _temporal_train_count(num_records: int, train_fraction: float) -> int:
    if num_records <= 1:
        return max(0, num_records)
    train_count = max(1, int(round(num_records * train_fraction)))
    return min(train_count, num_records - 1)


def split_membership_for_universe_row(row: dict[str, Any], split_contract: dict[str, Any]) -> dict[str, Any]:
    row_key = universe_row_id(row)
    heldout_recording = set(split_contract["manifests"]["heldout_recording"]["splits"].get("heldout_recording", []))
    heldout_game = set(split_contract["manifests"]["heldout_game"]["splits"].get("heldout_game", []))
    return {
        "row_id": row_key,
        "heldout_recording": row_key in heldout_recording,
        "heldout_game": row_key in heldout_game,
        "temporal_train_fraction": _temporal_fraction(split_contract),
    }


def annotate_window_records(
    records: list[dict[str, Any]],
    *,
    universe_row: dict[str, Any],
    split_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach full-corpus source/split metadata and de-collide sequence IDs.

    Existing sample extraction names sequences as ``game/recording#bin``.  Full
    D2E consumes multiple resolution tiers, so the source namespace must be part
    of the sequence ID and recording ID to avoid 480p/original collisions while
    preserving ``cross_resolution_key`` for leakage-aware statistics.
    """

    membership = split_membership_for_universe_row(universe_row, split_contract)
    source_id = str(universe_row["source_id"])
    cross_key = str(universe_row["cross_resolution_key"])
    source_recording_key = str(universe_row.get("source_recording_key") or cross_key)
    train_count = _temporal_train_count(len(records), float(membership["temporal_train_fraction"]))
    output = []
    for idx, row in enumerate(records):
        old_sequence_id = str(row["sequence_id"])
        temporal_split = "train" if idx < train_count else "heldout_temporal"
        eval_tags = []
        if temporal_split == "heldout_temporal":
            eval_tags.append("temporal")
        if membership["heldout_recording"]:
            eval_tags.append("heldout_recording")
        if membership["heldout_game"]:
            eval_tags.append("heldout_game")
        split = "train_core" if not eval_tags else "eval"
        annotated = dict(row)
        annotated.update(
            {
                "sequence_id": f"{source_id}:{old_sequence_id}",
                "source_sequence_id": old_sequence_id,
                "recording_id": f"{source_id}:{source_recording_key}",
                "source_recording_id": row.get("recording_id"),
                "source_id": source_id,
                "resolution_tier": universe_row.get("resolution_tier"),
                "source_recording_key": source_recording_key,
                "cross_resolution_key": cross_key,
                "universe_row_id": membership["row_id"],
                "split": split,
                "split_temporal": temporal_split,
                "split_heldout_recording": "heldout_recording" if membership["heldout_recording"] else "train",
                "split_heldout_game": "heldout_game" if membership["heldout_game"] else "train",
                "eval_split_tags": eval_tags,
            }
        )
        output.append(annotated)
    return output


def split_output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "all": root / "all_records.jsonl",
        "train_core": root / "train_core.jsonl",
        "target_temporal": root / "target_temporal.jsonl",
        "target_heldout_recording": root / "target_heldout_recording.jsonl",
        "target_heldout_game": root / "target_heldout_game.jsonl",
        "target_all_eval": root / "target_all_eval.jsonl",
    }


def write_split_records(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, int]:
    paths = split_output_paths(output_dir)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    by_split = {
        "all": records,
        "train_core": [row for row in records if row.get("split") == "train_core"],
        "target_temporal": [row for row in records if "temporal" in row.get("eval_split_tags", [])],
        "target_heldout_recording": [row for row in records if "heldout_recording" in row.get("eval_split_tags", [])],
        "target_heldout_game": [row for row in records if "heldout_game" in row.get("eval_split_tags", [])],
        "target_all_eval": [row for row in records if row.get("eval_split_tags")],
    }
    for name, rows in by_split.items():
        write_jsonl(paths[name], rows)
    return {name: len(rows) for name, rows in by_split.items()}
