from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _candidate_plan(path: Path) -> None:
    _write_json(
        path,
        {
            "schema": "aux_game_action_dataset_candidates.v1",
            "candidates": [
                {
                    "id": "aux_a",
                    "selection_status": "selected_candidate",
                    "source_url": "https://example.invalid/aux-a",
                    "license_id": "mit",
                    "domain": "Minecraft human demonstrations",
                },
                {
                    "id": "aux_b",
                    "selection_status": "review_required_not_selected",
                    "source_url": "https://example.invalid/aux-b",
                    "license_id": "review_required",
                    "domain": "Atari review candidate",
                },
            ],
        },
    )


def test_g005_namespace_builder_writes_fail_closed_template(tmp_path: Path):
    candidates = tmp_path / "aux_candidates.json"
    output = tmp_path / "namespace.json"
    _candidate_plan(candidates)

    subprocess.run(
        [
            sys.executable,
            "scripts/build_g005_aux_namespace_manifest.py",
            "--aux-candidates",
            str(candidates),
            "--output",
            str(output),
            "--allow-template",
        ],
        check=True,
    )

    payload = json.loads(output.read_text())
    assert payload["schema"] == "g005_aux_namespace_manifest.v1"
    assert payload["completion_ready"] is False
    assert payload["selected_aux_source_ids"] == ["aux_a"]
    assert payload["aux_sources"][0]["template_only"] is True
    assert payload["d2e_eval_manifests"]["same_as_d2e_only"] is False


def test_g005_namespace_builder_completion_ready_requires_materialized_sources_and_eval_hashes(tmp_path: Path):
    candidates = tmp_path / "aux_candidates.json"
    evidence = tmp_path / "source_evidence.json"
    eval_hashes = tmp_path / "eval_hashes.json"
    output = tmp_path / "namespace.json"
    _candidate_plan(candidates)
    _write_json(
        evidence,
        {
            "aux_sources": [
                {
                    "id": "aux_a",
                    "namespace": str(tmp_path / "outputs/aux/aux_a"),
                    "source_url": "https://example.invalid/aux-a",
                    "license_id": "mit",
                    "provenance_sha256": "abc123",
                    "action_head": {"type": "minecraft_keyboard_mouse", "namespace": "aux_a"},
                    "d2e_heldout_overlap_count": 0,
                    "d2e_heldout_overlap_recording_ids": [],
                    "split_hashes": {"train": {"sha256": "train-hash"}, "val": {"sha256": "val-hash"}, "test": {"sha256": "test-hash"}},
                }
            ]
        },
    )
    _write_json(
        eval_hashes,
        {
            "temporal": {"sha256": "temporal-hash"},
            "heldout_recording": {"sha256": "heldout-recording-hash"},
            "heldout_game": {"sha256": "heldout-game-hash"},
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_g005_aux_namespace_manifest.py",
            "--aux-candidates",
            str(candidates),
            "--source-evidence",
            str(evidence),
            "--eval-manifest-hashes",
            str(eval_hashes),
            "--completion-ready",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text())
    assert payload["completion_ready"] is True
    assert payload["aux_sources"][0]["namespace"] == "outputs/aux/aux_a/"
    assert payload["d2e_eval_manifests"]["same_as_d2e_only"] is True
    assert payload["d2e_eval_manifests"]["splits"]["heldout_game"]["same_hash"] is True
    assert payload["aux_sources"][0]["materialized"] is True


def test_g005_namespace_builder_rejects_unselected_source_evidence(tmp_path: Path):
    candidates = tmp_path / "aux_candidates.json"
    evidence = tmp_path / "source_evidence.json"
    _candidate_plan(candidates)
    _write_json(evidence, {"id": "aux_b", "namespace": "outputs/aux/aux_b/train/"})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_g005_aux_namespace_manifest.py",
            "--aux-candidates",
            str(candidates),
            "--source-evidence",
            str(evidence),
            "--allow-template",
            "--output",
            str(tmp_path / "namespace.json"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "not selected" in result.stderr or "not selected" in result.stdout


def test_g005_namespace_builder_rejects_candidate_plan_without_selected_sources(tmp_path: Path):
    candidates = tmp_path / "aux_candidates.json"
    _write_json(candidates, {"candidates": [{"id": "aux_review", "selection_status": "review_required"}]})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_g005_aux_namespace_manifest.py",
            "--aux-candidates",
            str(candidates),
            "--allow-template",
            "--output",
            str(tmp_path / "namespace.json"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no selected auxiliary sources" in result.stderr
