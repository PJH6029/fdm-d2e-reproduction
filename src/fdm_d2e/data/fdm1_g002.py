from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import stable_hash_json
from fdm_d2e.data.splits import stable_bucket

D2E_480P_README_URL = "https://huggingface.co/datasets/open-world-agents/D2E-480p/raw/{revision}/README.md"

# Human-authored taxonomy for ROADMAP Split-B coverage. It is intentionally broad:
# downstream reports can aggregate by primary_category or any tag.
GAME_TAXONOMY: dict[str, dict[str, Any]] = {
    "Apex_Legends": {"display_name": "Apex Legends", "primary_category": "fps_shooter", "tags": ["fps", "shooter", "battle_royale", "high_mouse_motion"]},
    "Barony": {"display_name": "Barony", "primary_category": "first_person_roguelike", "tags": ["first_person", "dungeon_crawler", "roguelike", "survival"]},
    "Battlefield_6_Open_Beta": {"display_name": "Battlefield 6 Open Beta", "primary_category": "fps_shooter", "tags": ["fps", "shooter", "vehicle", "large_map"]},
    "Brotato": {"display_name": "Brotato", "primary_category": "top_down_2d_action", "tags": ["top_down", "2d_action", "roguelike", "high_event_density"]},
    "Core_Keeper": {"display_name": "Core Keeper", "primary_category": "top_down_sandbox", "tags": ["top_down", "sandbox", "crafting", "survival"]},
    "Counter-Strike_2": {"display_name": "Counter-Strike 2", "primary_category": "fps_shooter", "tags": ["fps", "tactical_shooter", "high_mouse_motion"]},
    "Cyberpunk_2077": {"display_name": "Cyberpunk 2077", "primary_category": "open_world", "tags": ["first_person", "open_world", "rpg", "driving_segments"]},
    "Dinkum": {"display_name": "Dinkum", "primary_category": "life_sim", "tags": ["life_sim", "farming", "sandbox", "slower_paced"]},
    "Eternal_Return": {"display_name": "Eternal Return", "primary_category": "top_down_2d_action", "tags": ["top_down", "moba", "battle_royale", "ui_heavy"]},
    "Euro_Truck_Simulator_2": {"display_name": "Euro Truck Simulator 2", "primary_category": "driving_vehicle", "tags": ["driving", "vehicle", "simulation", "low_frequency_controls"]},
    "Grand_Theft_Auto_V": {"display_name": "Grand Theft Auto V", "primary_category": "open_world", "tags": ["third_person", "open_world", "driving", "shooter"]},
    "Grounded": {"display_name": "Grounded", "primary_category": "sandbox_survival", "tags": ["first_person", "sandbox", "survival", "crafting"]},
    "MapleStory_Worlds_Southperry": {"display_name": "MapleStory Worlds Southperry", "primary_category": "side_scroller_ui", "tags": ["side_scroller", "2d", "mmo", "ui_heavy", "keyboard_heavy"]},
    "Medieval_Dynasty": {"display_name": "Medieval Dynasty", "primary_category": "open_world_survival", "tags": ["first_person", "open_world", "survival", "life_sim"]},
    "Minecraft_1.21.8": {"display_name": "Minecraft 1.21.8", "primary_category": "sandbox_survival", "tags": ["first_person", "sandbox", "crafting", "survival"]},
    "Monster_Hunter_Wilds": {"display_name": "Monster Hunter Wilds", "primary_category": "third_person_action", "tags": ["third_person", "action_rpg", "open_area", "controller_like_keyboard_mouse"]},
    "OguForest": {"display_name": "OguForest", "primary_category": "top_down_2d_action", "tags": ["top_down", "2d", "adventure", "low_resource"]},
    "PEAK": {"display_name": "PEAK", "primary_category": "first_person_traversal", "tags": ["first_person", "traversal", "survival", "low_resource"]},
    "PUBG": {"display_name": "PUBG", "primary_category": "fps_shooter", "tags": ["fps", "shooter", "battle_royale", "high_mouse_motion"]},
    "Raft": {"display_name": "Raft", "primary_category": "sandbox_survival", "tags": ["first_person", "sandbox", "survival", "crafting"]},
    "Rainbow_Six": {"display_name": "Rainbow Six", "primary_category": "fps_shooter", "tags": ["fps", "tactical_shooter", "ui_heavy", "high_mouse_motion"]},
    "Ready_Or_Not": {"display_name": "Ready Or Not", "primary_category": "fps_shooter", "tags": ["fps", "tactical_shooter", "slow_tactical", "high_mouse_motion"]},
    "Satisfactory": {"display_name": "Satisfactory", "primary_category": "sandbox_factory", "tags": ["first_person", "sandbox", "factory", "crafting", "ui_heavy"]},
    "Skul": {"display_name": "Skul", "primary_category": "side_scroller_2d_action", "tags": ["side_scroller", "2d_action", "platformer_like", "low_resource"]},
    "Slime_Rancher": {"display_name": "Slime Rancher", "primary_category": "open_world_sandbox", "tags": ["first_person", "open_world", "sandbox", "slower_paced"]},
    "Stardew_Valley": {"display_name": "Stardew Valley", "primary_category": "life_sim", "tags": ["top_down", "farming", "life_sim", "ui_heavy", "slower_paced"]},
    "Super_Bunny_Man": {"display_name": "Super Bunny Man", "primary_category": "side_scroller_2d_action", "tags": ["side_scroller", "2d", "platformer_like", "physics", "low_resource"]},
    "VALORANT": {"display_name": "VALORANT", "primary_category": "fps_shooter", "tags": ["fps", "tactical_shooter", "high_mouse_motion", "low_resource"]},
    "Vampire_Survivors": {"display_name": "Vampire Survivors", "primary_category": "top_down_2d_action", "tags": ["top_down", "2d_action", "roguelike", "low_resource"]},
}

HELDOUT_GAME_CATEGORY_REQUIREMENTS = [
    {"role": "fps_or_shooter", "required_any_tag": ["fps", "shooter", "tactical_shooter"]},
    {"role": "sandbox_or_open_world", "required_any_tag": ["sandbox", "open_world", "crafting"]},
    {"role": "top_down_or_2d", "required_any_tag": ["top_down", "2d", "2d_action", "side_scroller"]},
    {"role": "ui_or_slower_paced", "required_any_tag": ["ui_heavy", "slower_paced", "life_sim", "low_frequency_controls"]},
    {"role": "low_resource", "required_any_tag": ["low_resource"]},
]


def fetch_text(url: str, *, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def readme_url_for_revision(revision: str) -> str:
    return D2E_480P_README_URL.format(revision=revision)


def _normalize_readme_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_d2e_readme_game_hours(readme_text: str) -> dict[str, float]:
    """Parse the game-hours table from the D2E-480p dataset README."""
    rows: dict[str, float] = {}
    in_games = False
    for line in readme_text.splitlines():
        if line.strip().lower() == "## games":
            in_games = True
            continue
        if in_games and line.startswith("## "):
            break
        if not in_games or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2 or cells[0].lower() in {"game", "---", "----------------------"}:
            continue
        if set(cells[1].replace(":", "").replace("-", "")) <= {""}:
            continue
        try:
            rows[cells[0]] = float(cells[1])
        except ValueError:
            continue
    return rows


def _display_name_lookup() -> dict[str, str]:
    return {_normalize_readme_name(meta["display_name"]): game for game, meta in GAME_TAXONOMY.items()}


def parse_d2e_readme_summary_hours(readme_text: str) -> float | None:
    match = re.search(r"\*\*(\d+(?:\.\d+)?)\s+hours\*\*", readme_text)
    return float(match.group(1)) if match else None


def map_readme_hours_to_games(readme_hours: dict[str, float], games: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    lookup = _display_name_lookup()
    # Explicit aliases for README names that intentionally differ from manifest folder names.
    aliases = {
        "battlefield6": "Battlefield_6_Open_Beta",
        "maplestoryworlds": "MapleStory_Worlds_Southperry",
        "minecraft": "Minecraft_1.21.8",
    }
    mapped: dict[str, float] = {}
    findings: list[dict[str, Any]] = []
    for readme_name, hours in readme_hours.items():
        key = aliases.get(_normalize_readme_name(readme_name)) or lookup.get(_normalize_readme_name(readme_name))
        if key:
            mapped[key] = hours
        else:
            findings.append({"severity": "warning", "code": "readme_game_unmapped", "readme_game": readme_name, "hours": hours})
    for game in games:
        if game not in mapped:
            findings.append({"severity": "warning", "code": "manifest_game_missing_readme_hours", "game": game})
    return mapped, findings


def _row_id(row: dict[str, Any]) -> str:
    return f"{row.get('source_id')}:{row.get('cross_resolution_key')}"


def _included_recordings(universe: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in universe.get("recordings", []) if row.get("status") == "included"]


def _group_by_cross_resolution(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("cross_resolution_key"))].append(row)
    return groups


def _primary_rows(universe: dict[str, Any], *, primary_source_id: str = "d2e_480p") -> list[dict[str, Any]]:
    return [row for row in _included_recordings(universe) if row.get("source_id") == primary_source_id]


def _decode_by_primary_row_id(decode_summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not decode_summary:
        return {}
    return {str(row.get("universe_row_id")): row for row in decode_summary.get("recordings", []) if isinstance(row, dict)}


def build_game_metadata(
    universe: dict[str, Any],
    *,
    readme_text: str | None = None,
    readme_revision: str | None = None,
    readme_url: str | None = None,
    decode_summary: dict[str, Any] | None = None,
    primary_source_id: str = "d2e_480p",
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    rows = _primary_rows(universe, primary_source_id=primary_source_id)
    games = sorted({str(row.get("game")) for row in rows})
    readme_summary_hours = parse_d2e_readme_summary_hours(readme_text or "")
    readme_hours, readme_findings = map_readme_hours_to_games(parse_d2e_readme_game_hours(readme_text or ""), games)
    decode_rows = _decode_by_primary_row_id(decode_summary)
    stats_by_game: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts_by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        rid = _row_id(row)
        game = str(row.get("game"))
        dec = decode_rows.get(rid)
        if dec:
            stats_by_game[game]["decoded_events"] += int(dec.get("num_decoded_events") or 0)
            stats_by_game[game]["frame_features"] += int(dec.get("num_frame_features") or 0)
            stats_by_game[game]["window_records"] += int(dec.get("num_window_records") or 0)
            for split, count in (dec.get("split_counts") or {}).items():
                split_counts_by_game[game][split] += int(count or 0)
        stats_by_game[game]["recordings"] += 1
        stats_by_game[game]["source_bytes"] += int(row.get("size_bytes") or 0)
    game_rows = []
    findings = list(readme_findings)
    for game in games:
        taxonomy = GAME_TAXONOMY.get(game)
        if taxonomy is None:
            findings.append({"severity": "warning", "code": "missing_game_taxonomy", "game": game})
            taxonomy = {"display_name": game.replace("_", " "), "primary_category": "uncategorized", "tags": []}
        window_records = int(stats_by_game[game].get("window_records", 0))
        estimated_decoded_hours = round(window_records * 0.05 / 3600.0, 6) if window_records else None
        decoded_events = int(stats_by_game[game].get("decoded_events", 0))
        events_per_hour = round(decoded_events / estimated_decoded_hours, 3) if estimated_decoded_hours else None
        game_rows.append(
            {
                "game": game,
                "display_name": taxonomy["display_name"],
                "primary_category": taxonomy["primary_category"],
                "tags": list(taxonomy.get("tags", [])),
                "published_hours_480p_readme": readme_hours.get(game),
                "recording_count_480p": int(stats_by_game[game].get("recordings", 0)),
                "source_bytes_480p": int(stats_by_game[game].get("source_bytes", 0)),
                "coarse_action_statistics": {
                    "source": "d2e_full_corpus_decode_summary" if decode_rows else "not_available_until_decode",
                    "granularity": "pre_tokenization_event_and_50ms_window_counts",
                    "decoded_events": decoded_events if decode_rows else None,
                    "frame_features": int(stats_by_game[game].get("frame_features", 0)) if decode_rows else None,
                    "window_records_50ms": window_records if decode_rows else None,
                    "estimated_decoded_hours_from_50ms_windows": estimated_decoded_hours,
                    "decoded_events_per_hour": events_per_hour,
                    "split_window_counts": dict(sorted(split_counts_by_game[game].items())) if decode_rows else {},
                    "token_level_breakdown_deferred_to": "G003 action-token dataset pipeline",
                },
            }
        )
    total_published_hours = round(sum(v for v in readme_hours.values()), 6)
    if readme_summary_hours is not None and total_published_hours and abs(readme_summary_hours - total_published_hours) > 0.5:
        findings.append({
            "severity": "info",
            "code": "readme_summary_hours_differs_from_table_sum",
            "summary_hours": readme_summary_hours,
            "table_sum_hours": total_published_hours,
            "interpretation": "Pinned README text and per-game table disagree; downstream reports should cite both and use manifest/table hours for per-game allocation.",
        })
    payload = {
        "schema": "fdm1_d2e_game_metadata.v1",
        "generated_at_utc": generated_at_utc or dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "canonical_roadmap": "ROADMAP.md",
        "primary_source_id": primary_source_id,
        "dataset_fingerprint": universe.get("dataset_fingerprint"),
        "readme_source": {
            "url": readme_url,
            "revision": readme_revision,
            "sha256": hashlib.sha256((readme_text or "").encode("utf-8")).hexdigest() if readme_text is not None else None,
        },
        "totals": {
            "games": len(games),
            "primary_recordings": len(rows),
            "published_summary_hours_from_readme_text": readme_summary_hours,
            "published_hours_from_readme_table_sum": total_published_hours or None,
            "published_hours_from_readme": total_published_hours or None,
            "decoded_window_records_50ms": sum(int(row["coarse_action_statistics"].get("window_records_50ms") or 0) for row in game_rows),
            "decoded_events": sum(int(row["coarse_action_statistics"].get("decoded_events") or 0) for row in game_rows),
        },
        "games": game_rows,
        "findings": findings,
    }
    payload["fingerprint"] = stable_hash_json({k: v for k, v in payload.items() if k != "fingerprint"})
    return payload


def _stratified_three_way_split(keys: list[str], *, seed: str, train_fraction: float, val_fraction: float) -> dict[str, str]:
    n = len(keys)
    ordered = sorted(keys, key=lambda key: (stable_bucket(key, seed=seed), key))
    if n == 0:
        return {}
    if n == 1:
        counts = (1, 0, 0)
    elif n == 2:
        counts = (1, 1, 0)
    else:
        val_count = max(1, round(n * val_fraction))
        test_count = max(1, n - round(n * (train_fraction + val_fraction)))
        # Keep at least one training recording for every game with >=3 recordings.
        train_count = max(1, n - val_count - test_count)
        while train_count + val_count + test_count > n and train_count > 1:
            train_count -= 1
        while train_count + val_count + test_count > n and val_count > 1:
            val_count -= 1
        while train_count + val_count + test_count > n and test_count > 1:
            test_count -= 1
        counts = (train_count, val_count, n - train_count - val_count)
    split: dict[str, str] = {}
    train_count, val_count, _ = counts
    for idx, key in enumerate(ordered):
        split[key] = "train" if idx < train_count else "val" if idx < train_count + val_count else "test"
    return split


def build_recording_level_split_manifest(
    universe: dict[str, Any],
    *,
    seed: str = "fdm1-d2e-recording-v1",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    primary_source_id: str = "d2e_480p",
) -> dict[str, Any]:
    del test_fraction  # implied by the first two fractions after stratified rounding.
    primary = _primary_rows(universe, primary_source_id=primary_source_id)
    by_game: dict[str, list[str]] = defaultdict(list)
    for row in primary:
        by_game[str(row.get("game"))].append(str(row.get("cross_resolution_key")))
    assignments: dict[str, str] = {}
    feasibility: dict[str, Any] = {}
    for game, keys in sorted(by_game.items()):
        unique_keys = sorted(set(keys))
        game_split = _stratified_three_way_split(unique_keys, seed=f"{seed}:{game}", train_fraction=train_fraction, val_fraction=val_fraction)
        assignments.update(game_split)
        counts = Counter(game_split.values())
        feasibility[game] = {
            "recordings": len(unique_keys),
            "has_train": counts.get("train", 0) > 0,
            "has_val": counts.get("val", 0) > 0,
            "has_test": counts.get("test", 0) > 0,
            "all_three_splits_possible": len(unique_keys) >= 3,
            "counts": dict(counts),
        }
    groups = _group_by_cross_resolution(_included_recordings(universe))
    split_rows = []
    for key, split in sorted(assignments.items()):
        variants = sorted(groups.get(key, []), key=lambda row: str(row.get("source_id")))
        split_rows.append(
            {
                "cross_resolution_key": key,
                "game": variants[0].get("game") if variants else key.split("/", 1)[0],
                "split": split,
                "primary_row_id": f"{primary_source_id}:{key}",
                "variant_row_ids": [_row_id(row) for row in variants],
            }
        )
    counts = Counter(row["split"] for row in split_rows)
    payload = {
        "schema": "fdm1_recording_level_split_manifest.v1",
        "dataset_fingerprint": universe.get("dataset_fingerprint"),
        "canonical_roadmap": "ROADMAP.md",
        "source_scope": {"primary_source_id": primary_source_id, "high_resolution_variants_grouped_for_leakage": True},
        "split_policy": {
            "method": "per_game_stratified_recording_level_80_10_10",
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": round(1.0 - train_fraction - val_fraction, 6),
            "seed": seed,
            "unit": "cross_resolution_key",
            "low_resource_policy": "games with fewer than three recordings preserve train coverage first and report infeasible missing val/test cells",
        },
        "splits": split_rows,
        "counts": dict(counts),
        "per_game_feasibility": feasibility,
    }
    payload["fingerprint"] = stable_hash_json({k: v for k, v in payload.items() if k != "fingerprint"})
    return payload


def choose_category_heldout_games(game_metadata: dict[str, Any], *, seed: str = "fdm1-d2e-heldout-game-v1") -> dict[str, Any]:
    games = {row["game"]: row for row in game_metadata.get("games", [])}
    selected: dict[str, str] = {}
    used: set[str] = set()
    for requirement in HELDOUT_GAME_CATEGORY_REQUIREMENTS:
        tags = set(requirement["required_any_tag"])
        candidates = [
            row for row in games.values()
            if row["game"] not in used and tags.intersection(row.get("tags", []))
        ]
        if not candidates:
            selected[requirement["role"]] = "<missing>"
            continue
        # Prefer moderate/low-resource heldouts to avoid destroying train coverage, but keep deterministic hashing.
        def rank(row: dict[str, Any]) -> tuple[int, int, str]:
            recordings = int(row.get("recording_count_480p") or 0)
            size_bucket = 0 if 3 <= recordings <= 16 else 1 if recordings > 16 else 2
            return (size_bucket, stable_bucket(row["game"], seed=f"{seed}:{requirement['role']}"), row["game"])
        chosen = sorted(candidates, key=rank)[0]
        selected[requirement["role"]] = chosen["game"]
        used.add(chosen["game"])
    missing = {role: game for role, game in selected.items() if game == "<missing>"}
    return {"selected_by_role": selected, "heldout_games": sorted(game for game in selected.values() if game != "<missing>"), "missing_roles": missing}


def build_heldout_game_manifest(universe: dict[str, Any], game_metadata: dict[str, Any], *, primary_source_id: str = "d2e_480p", seed: str = "fdm1-d2e-heldout-game-v1") -> dict[str, Any]:
    selection = choose_category_heldout_games(game_metadata, seed=seed)
    heldout_games = set(selection["heldout_games"])
    groups = _group_by_cross_resolution(_included_recordings(universe))
    rows = []
    for key, variants in sorted(groups.items()):
        primary_variant = next((row for row in variants if row.get("source_id") == primary_source_id), None)
        if not primary_variant:
            continue
        game = str(primary_variant.get("game"))
        rows.append(
            {
                "cross_resolution_key": key,
                "game": game,
                "split": "heldout_game_test" if game in heldout_games else "train_pool",
                "primary_row_id": f"{primary_source_id}:{key}",
                "variant_row_ids": [_row_id(row) for row in sorted(variants, key=lambda row: str(row.get("source_id")))],
            }
        )
    counts = Counter(row["split"] for row in rows)
    payload = {
        "schema": "fdm1_heldout_game_split_manifest.v1",
        "dataset_fingerprint": universe.get("dataset_fingerprint"),
        "canonical_roadmap": "ROADMAP.md",
        "split_policy": {
            "method": "category_coverage_heldout_game",
            "seed": seed,
            "selection_roles": HELDOUT_GAME_CATEGORY_REQUIREMENTS,
            "source_scope": primary_source_id,
        },
        "heldout_selection": selection,
        "splits": rows,
        "counts": dict(counts),
    }
    payload["fingerprint"] = stable_hash_json({k: v for k, v in payload.items() if k != "fingerprint"})
    return payload


def build_pseudo_label_split_manifest(recording_manifest: dict[str, Any], *, seed: str = "fdm1-d2e-pseudo-v1") -> dict[str, Any]:
    train_rows = [row for row in recording_manifest["splits"] if row["split"] == "train"]
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_game[str(row["game"])].append(row)
    splits: list[dict[str, Any]] = []
    for game, rows in sorted(by_game.items()):
        ordered = sorted(rows, key=lambda row: (stable_bucket(row["cross_resolution_key"], seed=f"{seed}:{game}"), row["cross_resolution_key"]))
        n = len(ordered)
        if n == 1:
            labeled_count, pseudo_count = 1, 0
        elif n == 2:
            labeled_count, pseudo_count = 1, 1
        else:
            labeled_count = max(1, round(n * 0.5))
            pseudo_count = max(1, round(n * 0.3))
            if labeled_count + pseudo_count >= n:
                pseudo_count = max(1, n - labeled_count - 1)
        for idx, row in enumerate(ordered):
            subset = "D_IDM_LABELED_A" if idx < labeled_count else "D_PSEUDO_B" if idx < labeled_count + pseudo_count else "D_FDM_GT_EVAL"
            out = dict(row)
            out["pseudo_label_split"] = subset
            splits.append(out)
    counts = Counter(row["pseudo_label_split"] for row in splits)
    payload = {
        "schema": "fdm1_pseudo_label_simulation_split_manifest.v1",
        "dataset_fingerprint": recording_manifest.get("dataset_fingerprint"),
        "canonical_roadmap": "ROADMAP.md",
        "source_manifest_fingerprint": recording_manifest.get("fingerprint"),
        "split_policy": {
            "method": "within_recording_train_pool_game_stratified_labeled_pseudo_eval",
            "seed": seed,
            "subsets": {
                "D_IDM_LABELED_A": "GT actions visible for IDM training",
                "D_PSEUDO_B": "GT hidden for IDM pseudo-label generation before FDM training",
                "D_FDM_GT_EVAL": "GT-labeled heldout rows for FDM-GT vs FDM-Pseudo evaluation",
            },
            "target_fractions_within_train_pool": {"labeled_a": 0.5, "pseudo_b": 0.3, "gt_eval": 0.2},
        },
        "splits": sorted(splits, key=lambda row: (row["pseudo_label_split"], row["game"], row["cross_resolution_key"])),
        "counts": dict(counts),
    }
    payload["fingerprint"] = stable_hash_json({k: v for k, v in payload.items() if k != "fingerprint"})
    return payload


def _recording_hours(row: dict[str, Any], game_metadata_by_game: dict[str, dict[str, Any]]) -> float:
    game = str(row.get("game"))
    meta = game_metadata_by_game.get(game, {})
    published = meta.get("published_hours_480p_readme")
    count = int(meta.get("recording_count_480p") or 0)
    if published and count:
        return float(published) / count
    coarse = meta.get("coarse_action_statistics") or {}
    decoded = coarse.get("estimated_decoded_hours_from_50ms_windows")
    if decoded and count:
        return float(decoded) / count
    return 1.0


def build_scale_split_manifest(recording_manifest: dict[str, Any], game_metadata: dict[str, Any], *, seed: str = "fdm1-d2e-scale-v1", scales: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)) -> dict[str, Any]:
    train_rows = [row for row in recording_manifest["splits"] if row["split"] == "train"]
    game_meta = {row["game"]: row for row in game_metadata.get("games", [])}
    ordered = sorted(train_rows, key=lambda row: (stable_bucket(row["cross_resolution_key"], seed=seed), row["cross_resolution_key"]))
    total_hours = sum(_recording_hours(row, game_meta) for row in ordered)
    scale_rows: dict[str, Any] = {}
    previous_keys: set[str] = set()
    for scale in scales:
        label = f"{int(scale * 100)}pct"
        target = total_hours * scale
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set(previous_keys)
        # Preserve nesting by carrying prior selected keys forward, then add deterministic rows until target hours.
        current_hours = 0.0
        for row in ordered:
            if row["cross_resolution_key"] in selected_keys:
                selected.append(row)
                current_hours += _recording_hours(row, game_meta)
        for row in ordered:
            if current_hours >= target and selected:
                break
            if row["cross_resolution_key"] in selected_keys:
                continue
            selected_keys.add(row["cross_resolution_key"])
            selected.append(row)
            current_hours += _recording_hours(row, game_meta)
        previous_keys = selected_keys
        scale_rows[label] = {
            "fraction": scale,
            "target_hours": round(target, 6),
            "selected_estimated_hours": round(current_hours, 6),
            "recordings": len(selected_keys),
            "games": sorted({row["game"] for row in selected}),
            "cross_resolution_keys": sorted(selected_keys),
            "primary_row_ids": sorted(row["primary_row_id"] for row in selected if row["cross_resolution_key"] in selected_keys),
        }
    payload = {
        "schema": "fdm1_data_scale_split_manifest.v1",
        "dataset_fingerprint": recording_manifest.get("dataset_fingerprint"),
        "canonical_roadmap": "ROADMAP.md",
        "source_manifest_fingerprint": recording_manifest.get("fingerprint"),
        "split_policy": {
            "method": "nested_training_pool_scales_by_estimated_hours",
            "seed": seed,
            "scale_fractions": list(scales),
            "hour_source_preference": ["D2E-480p README per-game hours divided by recording count", "decoded 50ms window hours", "unit recording fallback"],
        },
        "scales": scale_rows,
    }
    payload["fingerprint"] = stable_hash_json({k: v for k, v in payload.items() if k != "fingerprint"})
    return payload


def validate_g002_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    universe = bundle["data_universe"]
    game_metadata = bundle["game_metadata"]
    recording = bundle["recording_level_split"]
    heldout = bundle["heldout_game_split"]
    pseudo = bundle["pseudo_label_split"]
    scale = bundle["scale_split"]

    if universe.get("coverage", {}).get("games_count") != 29:
        findings.append({"severity": "error", "code": "expected_29_games", "actual": universe.get("coverage", {}).get("games_count")})
    if universe.get("coverage", {}).get("status_counts", {}).get("included") != universe.get("coverage", {}).get("recording_variants"):
        findings.append({"severity": "error", "code": "not_all_recording_variants_included"})
    source_480p = next((s for s in universe.get("d2e_sources", []) if s.get("source_id") == "d2e_480p"), {})
    if not source_480p.get("resolved_revision"):
        findings.append({"severity": "error", "code": "missing_d2e_480p_revision"})
    if len(game_metadata.get("games", [])) != 29:
        findings.append({"severity": "error", "code": "game_metadata_missing_games"})
    if any(row.get("published_hours_480p_readme") is None for row in game_metadata.get("games", [])):
        findings.append({"severity": "error", "code": "missing_published_hours"})
    if any(row.get("primary_category") == "uncategorized" for row in game_metadata.get("games", [])):
        findings.append({"severity": "error", "code": "uncategorized_game"})
    # Split A: no cross-resolution key appears in multiple splits.
    key_to_split: dict[str, str] = {}
    for row in recording.get("splits", []):
        key = row["cross_resolution_key"]
        split = row["split"]
        if key in key_to_split and key_to_split[key] != split:
            findings.append({"severity": "error", "code": "recording_split_leakage", "cross_resolution_key": key})
        key_to_split[key] = split
    if not {"train", "val", "test"}.issubset(set(recording.get("counts", {}))):
        findings.append({"severity": "error", "code": "recording_split_missing_train_val_test"})
    if heldout.get("heldout_selection", {}).get("missing_roles"):
        findings.append({"severity": "error", "code": "heldout_game_missing_required_role", "missing": heldout["heldout_selection"]["missing_roles"]})
    pseudo_counts = pseudo.get("counts", {})
    if not {"D_IDM_LABELED_A", "D_PSEUDO_B", "D_FDM_GT_EVAL"}.issubset(set(pseudo_counts)):
        findings.append({"severity": "error", "code": "pseudo_split_missing_subset", "counts": pseudo_counts})
    required_scales = {"1pct", "5pct", "10pct", "25pct", "50pct", "100pct"}
    if set(scale.get("scales", {})) != required_scales:
        findings.append({"severity": "error", "code": "scale_split_missing_required_scales", "actual": sorted(scale.get("scales", {}))})
    prior: set[str] = set()
    for label in ["1pct", "5pct", "10pct", "25pct", "50pct", "100pct"]:
        keys = set(scale["scales"][label]["cross_resolution_keys"])
        if not prior.issubset(keys):
            findings.append({"severity": "error", "code": "scale_split_not_nested", "scale": label})
        prior = keys
    status = "pass" if not any(item["severity"] == "error" for item in findings) else "fail"
    return {"schema": "fdm1_g002_contract_validation.v1", "status": status, "findings": findings, "error_count": sum(1 for item in findings if item["severity"] == "error")}
