from __future__ import annotations

import hashlib
from typing import Any


def deterministic_split(records: list[dict[str, Any]], train_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda r: (r.get('recording_id', ''), int(r['timestamp_ns'])))
    train, heldout = ordered[:train_count], ordered[train_count:]
    for row in train:
        row['split'] = 'train'
    for row in heldout:
        row['split'] = 'heldout'
    return train, heldout


def stable_bucket(value: str, *, seed: str = "fdm-d2e") -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def group_universe_recordings(recordings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in recordings:
        key = str(row.get("cross_resolution_key") or row.get("source_recording_key") or row.get("recording_id"))
        groups.setdefault(key, []).append(row)
    return groups


def _pick_heldout(values: list[str], *, fraction: float, min_heldout: int, seed: str) -> set[str]:
    if not values:
        return set()
    if len(values) == 1:
        return set(values) if min_heldout else set()
    count = max(min_heldout, int(round(len(values) * fraction)))
    count = min(count, len(values) - 1)
    ordered = sorted(values, key=lambda value: (stable_bucket(value, seed=seed), value))
    return set(ordered[:count])


def _row_id(row: dict[str, Any]) -> str:
    return f"{row.get('source_id')}:{row.get('cross_resolution_key')}"


def build_generalization_split_contract(
    data_universe_manifest: dict[str, Any],
    *,
    temporal_train_fraction: float = 0.8,
    heldout_recording_fraction: float = 0.2,
    heldout_game_fraction: float = 0.2,
    min_heldout_recordings: int = 1,
    min_heldout_games: int = 1,
    seed: str = "fdm-d2e-full-v1",
) -> dict[str, Any]:
    """Build leakage-safe full-corpus split contracts from the data-universe manifest.

    The three split regimes are intentionally independent:
    - temporal: every recording has prefix train / tail heldout windows;
    - heldout-recording: whole source recordings are held out, with all
      resolution variants assigned together;
    - heldout-game: whole games are held out, again with all resolution variants
      assigned together.
    """

    recordings = [row for row in data_universe_manifest.get("recordings", []) if row.get("status") == "included"]
    dataset_fingerprint = str(data_universe_manifest.get("dataset_fingerprint", ""))
    groups = group_universe_recordings(recordings)
    group_keys = sorted(groups)
    games = sorted({str(row.get("game")) for row in recordings})

    heldout_games = _pick_heldout(games, fraction=heldout_game_fraction, min_heldout=min_heldout_games, seed=f"{seed}:game")
    heldout_recording_keys = _pick_heldout(
        group_keys,
        fraction=heldout_recording_fraction,
        min_heldout=min_heldout_recordings,
        seed=f"{seed}:recording",
    )

    temporal_rows = []
    for key in group_keys:
        variants = sorted(_row_id(row) for row in groups[key])
        temporal_rows.append(
            {
                "cross_resolution_key": key,
                "game": groups[key][0].get("game"),
                "variant_row_ids": variants,
                "window_policy": {
                    "train": {"range": "prefix", "fraction": temporal_train_fraction},
                    "heldout": {"range": "tail", "fraction": round(1.0 - temporal_train_fraction, 6)},
                    "minimum_unit": "window_index_within_source_recording",
                },
            }
        )

    recording_train: list[str] = []
    recording_heldout: list[str] = []
    for key in group_keys:
        target = recording_heldout if key in heldout_recording_keys else recording_train
        target.extend(_row_id(row) for row in groups[key])

    game_train: list[str] = []
    game_heldout: list[str] = []
    for row in recordings:
        (game_heldout if row.get("game") in heldout_games else game_train).append(_row_id(row))

    temporal_manifest = {
        "schema": "split_manifest.v1",
        "dataset_fingerprint": dataset_fingerprint,
        "split_policy": {
            "method": "within_recording_temporal_prefix_tail",
            "train_fraction": temporal_train_fraction,
            "heldout_fraction": round(1.0 - temporal_train_fraction, 6),
            "seed": seed,
            "cross_resolution_grouping": "cross_resolution_key",
        },
        "splits": {"recordings": temporal_rows},
    }
    heldout_recording_manifest = {
        "schema": "split_manifest.v1",
        "dataset_fingerprint": dataset_fingerprint,
        "split_policy": {
            "method": "heldout_recording_by_cross_resolution_key",
            "heldout_fraction": heldout_recording_fraction,
            "min_heldout_recordings": min_heldout_recordings,
            "seed": seed,
            "cross_resolution_grouping": "cross_resolution_key",
        },
        "splits": {
            "train": sorted(recording_train),
            "heldout_recording": sorted(recording_heldout),
            "heldout_recording_keys": sorted(heldout_recording_keys),
        },
    }
    heldout_game_manifest = {
        "schema": "split_manifest.v1",
        "dataset_fingerprint": dataset_fingerprint,
        "split_policy": {
            "method": "heldout_game",
            "heldout_fraction": heldout_game_fraction,
            "min_heldout_games": min_heldout_games,
            "seed": seed,
            "cross_resolution_grouping": "game",
        },
        "splits": {
            "train": sorted(game_train),
            "heldout_game": sorted(game_heldout),
            "heldout_games": sorted(heldout_games),
        },
    }
    leakage_report = validate_generalization_splits(
        recordings,
        temporal_manifest=temporal_manifest,
        heldout_recording_manifest=heldout_recording_manifest,
        heldout_game_manifest=heldout_game_manifest,
    )
    return {
        "schema": "generalization_split_contract.v1",
        "dataset_fingerprint": dataset_fingerprint,
        "source_recording_groups": len(groups),
        "recording_variants": len(recordings),
        "games": games,
        "manifests": {
            "temporal": temporal_manifest,
            "heldout_recording": heldout_recording_manifest,
            "heldout_game": heldout_game_manifest,
        },
        "leakage_report": leakage_report,
    }


def validate_generalization_splits(
    recordings: list[dict[str, Any]],
    *,
    temporal_manifest: dict[str, Any],
    heldout_recording_manifest: dict[str, Any],
    heldout_game_manifest: dict[str, Any],
) -> dict[str, Any]:
    groups = group_universe_recordings(recordings)
    row_to_group = {_row_id(row): key for key, rows in groups.items() for row in rows}
    row_to_game = {_row_id(row): str(row.get("game")) for row in recordings}

    recording_train = set(heldout_recording_manifest["splits"].get("train", []))
    recording_heldout = set(heldout_recording_manifest["splits"].get("heldout_recording", []))
    recording_overlap = recording_train & recording_heldout
    group_assignments: dict[str, set[str]] = {}
    for row_id in recording_train:
        group_assignments.setdefault(row_to_group[row_id], set()).add("train")
    for row_id in recording_heldout:
        group_assignments.setdefault(row_to_group[row_id], set()).add("heldout_recording")
    split_groups = sorted(group for group, values in group_assignments.items() if len(values) > 1)

    game_train = set(heldout_game_manifest["splits"].get("train", []))
    game_heldout = set(heldout_game_manifest["splits"].get("heldout_game", []))
    game_overlap = game_train & game_heldout
    heldout_games = set(heldout_game_manifest["splits"].get("heldout_games", []))
    leaked_heldout_game_rows = sorted(row_id for row_id in game_train if row_to_game[row_id] in heldout_games)

    temporal_rows = temporal_manifest["splits"].get("recordings", [])
    temporal_keys = {row["cross_resolution_key"] for row in temporal_rows}
    missing_temporal = sorted(set(groups) - temporal_keys)
    errors = []
    if recording_overlap:
        errors.append(f"heldout_recording row overlap: {len(recording_overlap)}")
    if split_groups:
        errors.append(f"cross-resolution recording split across train/heldout: {split_groups[:5]}")
    if game_overlap:
        errors.append(f"heldout_game row overlap: {len(game_overlap)}")
    if leaked_heldout_game_rows:
        errors.append(f"heldout game rows leaked to train: {len(leaked_heldout_game_rows)}")
    if missing_temporal:
        errors.append(f"temporal policy missing groups: {len(missing_temporal)}")

    report = {
        "schema": "split_leakage_report.v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": {
            "heldout_recording_rows_disjoint": not recording_overlap,
            "cross_resolution_group_assignment_consistent": not split_groups,
            "heldout_game_rows_disjoint": not game_overlap,
            "heldout_game_not_in_train": not leaked_heldout_game_rows,
            "temporal_policy_covers_all_groups": not missing_temporal,
            "temporal_window_policy_disjoint_by_construction": True,
        },
        "counts": {
            "recording_train_rows": len(recording_train),
            "recording_heldout_rows": len(recording_heldout),
            "heldout_games": len(heldout_games),
            "game_train_rows": len(game_train),
            "game_heldout_rows": len(game_heldout),
            "temporal_groups": len(temporal_keys),
        },
    }
    if errors:
        raise ValueError("; ".join(errors))
    return report
