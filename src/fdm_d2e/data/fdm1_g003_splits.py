from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fdm_d2e.io_utils import read_json


@dataclass(frozen=True)
class FDM1G002SplitIndex:
    recording_split_by_key: dict[str, str]
    heldout_game_split_by_key: dict[str, str]
    pseudo_split_by_key: dict[str, str]
    scale_keys_by_key: dict[str, tuple[str, ...]]
    fingerprints: dict[str, str | None]


def _rows_by_key(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("splits", [])
    if not isinstance(rows, list):
        raise ValueError(f"expected split list in {manifest.get('schema')}")
    return {str(row["cross_resolution_key"]): row for row in rows}


def build_g002_split_index(
    *,
    recording_level_split: dict[str, Any],
    heldout_game_split: dict[str, Any],
    pseudo_label_split: dict[str, Any],
    scale_split: dict[str, Any],
) -> FDM1G002SplitIndex:
    recording_rows = _rows_by_key(recording_level_split)
    heldout_rows = _rows_by_key(heldout_game_split)
    pseudo_rows = _rows_by_key(pseudo_label_split)
    scale_keys_by_key: dict[str, list[str]] = {}
    for scale_name, scale in scale_split.get("scales", {}).items():
        for key in scale.get("cross_resolution_keys", []) or []:
            scale_keys_by_key.setdefault(str(key), []).append(str(scale_name))
    return FDM1G002SplitIndex(
        recording_split_by_key={key: str(row.get("split", "unknown")) for key, row in recording_rows.items()},
        heldout_game_split_by_key={key: str(row.get("split", "train_pool")) for key, row in heldout_rows.items()},
        pseudo_split_by_key={key: str(row.get("pseudo_label_split", "not_in_pseudo_pool")) for key, row in pseudo_rows.items()},
        scale_keys_by_key={key: tuple(sorted(values, key=lambda item: (len(item), item))) for key, values in scale_keys_by_key.items()},
        fingerprints={
            "recording_level_split": recording_level_split.get("fingerprint"),
            "heldout_game_split": heldout_game_split.get("fingerprint"),
            "pseudo_label_split": pseudo_label_split.get("fingerprint"),
            "scale_split": scale_split.get("fingerprint"),
        },
    )


def load_g002_split_index(
    *,
    recording_level_split_path: str | Path,
    heldout_game_split_path: str | Path,
    pseudo_label_split_path: str | Path,
    scale_split_path: str | Path,
) -> FDM1G002SplitIndex:
    return build_g002_split_index(
        recording_level_split=read_json(recording_level_split_path),
        heldout_game_split=read_json(heldout_game_split_path),
        pseudo_label_split=read_json(pseudo_label_split_path),
        scale_split=read_json(scale_split_path),
    )


def fdm1_split_metadata(cross_resolution_key: str, index: FDM1G002SplitIndex) -> dict[str, Any]:
    key = str(cross_resolution_key)
    recording_split = index.recording_split_by_key.get(key, "unknown")
    heldout_game_split = index.heldout_game_split_by_key.get(key, "train_pool")
    pseudo_split = index.pseudo_split_by_key.get(key, "not_in_pseudo_pool")
    scale_memberships = list(index.scale_keys_by_key.get(key, ()))
    eval_tags: list[str] = []
    if recording_split == "val":
        eval_tags.append("recording_val")
    elif recording_split == "test":
        eval_tags.append("recording_test")
    elif recording_split == "unknown":
        eval_tags.append("unknown_recording_split")
    if heldout_game_split == "heldout_game_test":
        eval_tags.append("heldout_game")
    if pseudo_split == "D_FDM_GT_EVAL":
        eval_tags.append("pseudo_gt_eval")
    split = "train_core" if not eval_tags and recording_split == "train" else "eval"
    return {
        "fdm1_recording_split": recording_split,
        "fdm1_heldout_game_split": heldout_game_split,
        "fdm1_pseudo_label_split": pseudo_split,
        "fdm1_scale_memberships": scale_memberships,
        "eval_split_tags": eval_tags,
        "split": split,
    }


def annotate_window_records_with_fdm1_splits(
    records: Sequence[dict[str, Any]],
    *,
    universe_row: dict[str, Any],
    split_index: FDM1G002SplitIndex,
) -> list[dict[str, Any]]:
    source_id = str(universe_row["source_id"])
    cross_key = str(universe_row["cross_resolution_key"])
    source_recording_key = str(universe_row.get("source_recording_key") or cross_key)
    metadata = fdm1_split_metadata(cross_key, split_index)
    output: list[dict[str, Any]] = []
    for row in records:
        old_sequence_id = str(row["sequence_id"])
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
                "universe_row_id": f"{source_id}:{cross_key}",
                "fdm1_split_fingerprints": split_index.fingerprints,
                **metadata,
            }
        )
        output.append(annotated)
    return output


__all__ = [
    "FDM1G002SplitIndex",
    "annotate_window_records_with_fdm1_splits",
    "build_g002_split_index",
    "fdm1_split_metadata",
    "load_g002_split_index",
]
