from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fdm_d2e.eval.gidm_adapter import build_gidm_inference_manifest, convert_gidm_mcap_predictions
from fdm_d2e.eval.gidm_runner import (
    GidmRunPlan,
    _run_one,
    build_gidm_chunk_run_plan,
    build_gidm_run_plan,
    prepare_desktop_minimal_inference_script,
    write_chunked_gidm_manifest,
)
from fdm_d2e.eval.gidm_targets import extract_gidm_target_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_gidm_manifest_joins_target_rows_to_decode_summary(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    _write_jsonl(
        target,
        [
            {
                "sequence_id": "d2e_480p:Apex/001#000001",
                "source_id": "d2e_480p",
                "universe_row_id": "d2e_480p:Apex/001",
                "cross_resolution_key": "Apex/001",
                "source_recording_id": "001",
                "game": "Apex",
                "timestamp_ns": 50_000_000,
                "bin_index": 1,
                "eval_split_tags": ["temporal"],
            },
            {
                "sequence_id": "d2e_480p:Apex/001#000002",
                "source_id": "d2e_480p",
                "universe_row_id": "d2e_480p:Apex/001",
                "cross_resolution_key": "Apex/001",
                "source_recording_id": "001",
                "game": "Apex",
                "timestamp_ns": 100_000_000,
                "bin_index": 2,
                "eval_split_tags": ["heldout_recording"],
            },
        ],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Apex/001",
                        "source_id": "d2e_480p",
                        "cross_resolution_key": "Apex/001",
                        "game": "Apex",
                        "recording_id": "001",
                        "video_source": "/data/Apex/001.mkv",
                        "mcap_path": "/data/Apex/001.mcap",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_gidm_inference_manifest(
        target_record_paths=[target],
        decode_summary_path=summary,
        output_dir=tmp_path / "gidm",
    )

    assert payload["recording_count"] == 1
    assert payload["target_rows"] == 2
    row = payload["recordings"][0]
    assert row["video_path"] == "/data/Apex/001.mkv"
    assert row["ground_truth_mcap_path"] == "/data/Apex/001.mcap"
    assert row["split_tags"] == ["heldout_recording", "temporal"]
    assert row["prediction_mcap_path"].endswith("d2e_480p_Apex_001.mcap")


def test_build_gidm_manifest_can_use_decode_summary_split_counts(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Apex/001",
                        "source_id": "d2e_480p",
                        "cross_resolution_key": "Apex/001",
                        "game": "Apex",
                        "recording_id": "001",
                        "video_source": "/data/Apex/001.mkv",
                        "mcap_path": "/data/Apex/001.mcap",
                        "split_counts": {
                            "train_core": 80,
                            "target_temporal": 20,
                            "target_heldout_recording": 100,
                            "target_heldout_game": 0,
                        },
                    },
                    {
                        "universe_row_id": "d2e_480p:Apex/002",
                        "source_id": "d2e_480p",
                        "cross_resolution_key": "Apex/002",
                        "game": "Apex",
                        "recording_id": "002",
                        "video_source": "/data/Apex/002.mkv",
                        "mcap_path": "/data/Apex/002.mcap",
                        "split_counts": {
                            "train_core": 160,
                            "target_temporal": 40,
                            "target_heldout_recording": 0,
                            "target_heldout_game": 0,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_gidm_inference_manifest(
        target_record_paths=[],
        decode_summary_path=summary,
        output_dir=tmp_path / "gidm",
        use_decode_summary_counts=True,
    )

    assert payload["source_mode"] == "decode_summary_split_counts"
    assert payload["recording_count"] == 2
    assert payload["target_rows"] == 140
    assert payload["recordings"][0]["split_tags"] == ["heldout_recording", "temporal"]
    assert payload["recordings"][0]["row_count"] == 100
    assert payload["recordings"][1]["split_tags"] == ["temporal"]
    assert payload["recordings"][1]["row_count"] == 40


def test_convert_gidm_mcap_predictions_bins_events_to_target_rows(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    _write_jsonl(
        target,
        [
            {
                "sequence_id": "rec#000000",
                "universe_row_id": "d2e_480p:Game/rec",
                "recording_id": "d2e_480p:Game/rec",
                "source_id": "d2e_480p",
                "cross_resolution_key": "Game/rec",
                "game": "Game",
                "timestamp_ns": 1_000_000_000,
            },
            {
                "sequence_id": "rec#000001",
                "universe_row_id": "d2e_480p:Game/rec",
                "recording_id": "d2e_480p:Game/rec",
                "source_id": "d2e_480p",
                "cross_resolution_key": "Game/rec",
                "game": "Game",
                "timestamp_ns": 1_050_000_000,
            },
        ],
    )
    pred_mcap = tmp_path / "pred.mcap"
    pred_mcap.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec",
                        "prediction_mcap_path": str(pred_mcap),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_decode(_path, *, topics):
        assert "keyboard" in topics
        return [
            {"type": "keyboard", "event_type": "press", "key": "87", "timestamp_ns": 1_001_000_000},
            {"type": "mouse_move", "dx": 10, "dy": -2, "timestamp_ns": 1_052_000_000},
        ]

    out = tmp_path / "predictions.jsonl"
    payload = convert_gidm_mcap_predictions(
        manifest_path=manifest,
        target_record_paths=[target],
        output_path=out,
        summary_out=tmp_path / "summary.json",
        decode_fn=fake_decode,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert payload["rows_written"] == 2
    assert rows[0]["predicted_tokens"] == ["KEY_PRESS_87"]
    assert rows[1]["predicted_tokens"] == ["MOUSE_DX_P4", "MOUSE_DY_N2"]


def test_convert_gidm_mcap_predictions_can_auto_align_first_screen_timestamp(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    _write_jsonl(
        target,
        [
            {
                "sequence_id": "rec#000000",
                "universe_row_id": "d2e_480p:Game/rec",
                "recording_id": "d2e_480p:Game/rec",
                "source_id": "d2e_480p",
                "cross_resolution_key": "Game/rec",
                "game": "Game",
                "timestamp_ns": 2_400_000_000,
            }
        ],
    )
    gt_mcap = tmp_path / "gt.mcap"
    pred_mcap = tmp_path / "pred.mcap"
    gt_mcap.write_text("placeholder", encoding="utf-8")
    pred_mcap.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec",
                        "ground_truth_mcap_path": str(gt_mcap),
                        "prediction_mcap_path": str(pred_mcap),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_decode(path, *, topics, limit=None):
        if topics == ["screen"]:
            if str(path) == str(gt_mcap):
                return [{"type": "screen", "timestamp_ns": 2_400_000_000}]
            return [{"type": "screen", "timestamp_ns": 50_000_000}]
        return [{"type": "keyboard", "event_type": "press", "key": "87", "timestamp_ns": 50_000_000}]

    out = tmp_path / "predictions.jsonl"
    payload = convert_gidm_mcap_predictions(
        manifest_path=manifest,
        target_record_paths=[target],
        output_path=out,
        summary_out=tmp_path / "summary.json",
        auto_timestamp_shift_from_screen=True,
        decode_fn=fake_decode,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert payload["timestamp_shifts_by_key"]["d2e_480p:Game/rec"] == 2_350_000_000
    assert rows[0]["predicted_tokens"] == ["KEY_PRESS_87"]


def test_convert_gidm_mcap_predictions_reads_chunked_aligned_paths(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    _write_jsonl(
        target,
        [
            {
                "sequence_id": "rec#000000",
                "universe_row_id": "d2e_480p:Game/rec",
                "recording_id": "d2e_480p:Game/rec",
                "source_id": "d2e_480p",
                "cross_resolution_key": "Game/rec",
                "game": "Game",
                "timestamp_ns": 2_400_000_000,
            },
            {
                "sequence_id": "rec#000001",
                "universe_row_id": "d2e_480p:Game/rec",
                "recording_id": "d2e_480p:Game/rec",
                "source_id": "d2e_480p",
                "cross_resolution_key": "Game/rec",
                "game": "Game",
                "timestamp_ns": 2_450_000_000,
            },
        ],
    )
    chunk_a = tmp_path / "chunk_a.mcap"
    chunk_b = tmp_path / "chunk_b.mcap"
    chunk_a.write_text("placeholder", encoding="utf-8")
    chunk_b.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec",
                        "ground_truth_mcap_path": str(tmp_path / "gt.mcap"),
                        "prediction_mcap_paths": [str(chunk_a), str(chunk_b)],
                        "prediction_timestamps_aligned_to_ground_truth": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_decode(path, *, topics, limit=None):
        if topics == ["screen"]:
            raise AssertionError("screen auto-shift should be skipped for aligned chunk manifests")
        if str(path) == str(chunk_a):
            return [{"type": "keyboard", "event_type": "press", "key": "87", "timestamp_ns": 2_401_000_000}]
        return [{"type": "mouse_move", "dx": 1, "dy": -1, "timestamp_ns": 2_451_000_000}]

    out = tmp_path / "predictions.jsonl"
    payload = convert_gidm_mcap_predictions(
        manifest_path=manifest,
        target_record_paths=[target],
        output_path=out,
        summary_out=tmp_path / "summary.json",
        auto_timestamp_shift_from_screen=True,
        decode_fn=fake_decode,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert payload["timestamp_shifts_by_key"]["d2e_480p:Game/rec"] == 0
    assert payload["auto_shift_skipped_by_key"]["d2e_480p:Game/rec"] is True
    assert payload["decoded_event_counts_by_path"]["d2e_480p:Game/rec"][str(chunk_a)] == 1
    assert rows[0]["predicted_tokens"] == ["KEY_PRESS_87"]
    assert rows[1]["predicted_tokens"] == ["MOUSE_DX_P1", "MOUSE_DY_N1"]


def test_prepare_desktop_minimal_inference_script_keeps_desktop_constants(tmp_path: Path):
    repo = tmp_path / "D2E"
    repo.mkdir()
    source = repo / "inference.py"
    source.write_text(
        "\n".join(
            [
                '#     "owa-cli @ git+https://example/owa-cli",',
                '#     "owa-env-desktop @ git+https://example/owa-env-desktop",',
                '#     "owa-env-gst @ git+https://example/owa-env-gst",',
                "def preprocess_video(input_path: str, output_path: str, duration: float) -> str:",
                "    cmd = [",
                '        "ffmpeg",',
                '        "-y",',
                '        "-i",',
                "        input_path,",
                "    ]",
                "    return output_path",
                "def create_mcap_from_video(video_path: str, mcap_path: str, fps: float = 20.0):",
                "    interval_ns = int(1e9 / fps)",
                "    num_frames = 1",
                "    with writer:",
                "        for i in range(num_frames):",
                "            timestamp_ns = i * interval_ns",
                "            screen_msg = ScreenCaptured(",
                "                utc_ns=timestamp_ns,",
                '                media_ref={"uri": video_abs_path, "pts_ns": timestamp_ns},',
                "            )",
                '            writer.write_message(screen_msg, topic="screen", timestamp=timestamp_ns)',
                "def main():",
                "    parser.add_argument(",
                '        "--max-duration", type=float, default=None, help="Max video duration in seconds (default: no limit)"',
                "    )",
                "    with tempfile.TemporaryDirectory() as tmpdir:",
                "        # Preprocess video if max_duration is specified",
                "        if args.max_duration is not None:",
                '            logger.info(f"Preprocessing video (max {args.max_duration}s)...")',
                '            processed_video = str(tmpdir / "processed.mkv")',
                "            preprocess_video(str(input_video), processed_video, args.max_duration)",
                "        else:",
                "            processed_video = str(input_video)",
                "        create_mcap_from_video(processed_video, input_mcap)",
            ]
        ),
        encoding="utf-8",
    )

    target = prepare_desktop_minimal_inference_script(repo)
    text = target.read_text(encoding="utf-8")
    assert "owa-env-desktop @" in text
    assert "owa-cli @" not in text
    assert "owa-env-gst @" not in text
    assert "--start-time" in text
    assert "--timestamp-offset" in text
    assert "timestamp_offset_seconds" in text


def test_build_gidm_run_plan_assigns_devices_and_resumes_existing_outputs(tmp_path: Path):
    existing = tmp_path / "existing.mcap"
    existing.write_text("done", encoding="utf-8")
    missing = tmp_path / "missing.mcap"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {"universe_row_id": "rec/existing", "video_path": "a.mkv", "prediction_mcap_path": str(existing)},
                    {"universe_row_id": "rec/missing", "video_path": "b.mkv", "prediction_mcap_path": str(missing)},
                ]
            }
        ),
        encoding="utf-8",
    )

    plans = build_gidm_run_plan(
        manifest_path=manifest,
        cuda_devices=["0", "1"],
        log_dir=tmp_path / "logs",
        resume=True,
    )

    assert [plan.universe_row_id for plan in plans] == ["rec/missing"]
    assert plans[0].cuda_device == "0"


def test_gidm_runner_passes_absolute_output_to_upstream(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    d2e_repo = repo / "outputs" / "external" / "D2E"
    d2e_repo.mkdir(parents=True)
    script = d2e_repo / "inference_desktop_minimal.py"
    script.write_text("print('placeholder')\n", encoding="utf-8")
    output = Path("outputs/gidm_exact_split/predicted_mcap/example.mcap")
    log = repo / "artifacts/eval/run.log"
    seen = {}

    def fake_run(cmd, *, cwd, env, stdout, stderr, text):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        out = Path(cmd[4])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mcap")
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(repo)
    monkeypatch.setattr("fdm_d2e.eval.gidm_runner.subprocess.run", fake_run)

    row = _run_one(
        GidmRunPlan(
            index=0,
            universe_row_id="d2e_480p:Game/rec",
            video_path="/data/video.mkv",
            prediction_mcap_path=str(output),
            cuda_device="0",
            log_path=str(log),
        ),
        script_path=script,
        d2e_repo=d2e_repo,
        model="open-world-agents/Generalist-IDM-1B",
        max_context_length=2048,
        max_duration=None,
        uv_cache_dir=repo / "uv-cache",
        hf_home=repo / "hf-home",
    )

    assert seen["cwd"] == d2e_repo
    assert Path(seen["cmd"][4]).is_absolute()
    assert Path(seen["cmd"][4]).name.startswith(f"{output.name}.tmp.")
    assert row["prediction_mcap_path"] == str(output)
    assert row["resolved_prediction_mcap_path"] == str((repo / output).resolve())
    assert row["temp_prediction_mcap_path"] == str(Path(seen["cmd"][4]))
    assert row["success"] is True
    assert row["output_exists"] is True


def test_gidm_runner_passes_chunk_offsets_to_upstream(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    d2e_repo = repo / "outputs" / "external" / "D2E"
    d2e_repo.mkdir(parents=True)
    script = d2e_repo / "inference_desktop_minimal.py"
    script.write_text("print('placeholder')\n", encoding="utf-8")
    output = Path("outputs/gidm_exact_split/predicted_mcap/example.mcap")
    log = repo / "artifacts/eval/run.log"
    seen = {}

    def fake_run(cmd, *, cwd, env, stdout, stderr, text):
        seen["cmd"] = cmd
        out = Path(cmd[4])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mcap")
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(repo)
    monkeypatch.setattr("fdm_d2e.eval.gidm_runner.subprocess.run", fake_run)

    row = _run_one(
        GidmRunPlan(
            index=0,
            universe_row_id="d2e_480p:Game/rec",
            video_path="/data/video.mkv",
            prediction_mcap_path=str(output),
            cuda_device="0",
            log_path=str(log),
        ),
        script_path=script,
        d2e_repo=d2e_repo,
        model="open-world-agents/Generalist-IDM-1B",
        max_context_length=2048,
        max_duration=12.5,
        start_time=10.0,
        timestamp_offset=319.5,
        uv_cache_dir=repo / "uv-cache",
        hf_home=repo / "hf-home",
    )

    assert seen["cmd"][seen["cmd"].index("--max-duration") + 1] == "12.5"
    assert seen["cmd"][seen["cmd"].index("--start-time") + 1] == "10.0"
    assert seen["cmd"][seen["cmd"].index("--timestamp-offset") + 1] == "319.5"
    assert row["start_time_seconds"] == 10.0
    assert row["timestamp_offset_seconds"] == 319.5


def test_gidm_runner_skips_existing_output_at_run_time(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    d2e_repo = repo / "outputs" / "external" / "D2E"
    d2e_repo.mkdir(parents=True)
    script = d2e_repo / "inference_desktop_minimal.py"
    script.write_text("print('placeholder')\n", encoding="utf-8")
    output = repo / "outputs/gidm_exact_split/predicted_mcap/example.mcap"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called for existing output")

    monkeypatch.chdir(repo)
    monkeypatch.setattr("fdm_d2e.eval.gidm_runner.subprocess.run", fail_run)

    row = _run_one(
        GidmRunPlan(
            index=0,
            universe_row_id="d2e_480p:Game/rec",
            video_path="/data/video.mkv",
            prediction_mcap_path=str(output),
            cuda_device="0",
            log_path=str(repo / "artifacts/eval/run.log"),
        ),
        script_path=script,
        d2e_repo=d2e_repo,
        model="open-world-agents/Generalist-IDM-1B",
        max_context_length=2048,
        max_duration=None,
        uv_cache_dir=repo / "uv-cache",
        hf_home=repo / "hf-home",
    )

    assert row["success"] is True
    assert row["skipped_existing_at_run"] is True
    assert row["output_size"] == len(b"existing")


def test_extract_gidm_target_records_uses_by_recording_roots(tmp_path: Path):
    root = tmp_path / "shard_0" / "by_recording"
    records = root / "d2e_480p" / "Game" / "rec_001" / "all_records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "sequence_id": "rec_001#000000",
                "source_id": "d2e_480p",
                "universe_row_id": "d2e_480p:Game/rec_001",
                "cross_resolution_key": "Game/rec_001",
                "source_recording_id": "rec_001",
                "recording_id": "d2e_480p:Game/rec_001",
                "game": "Game",
                "timestamp_ns": 100,
                "bin_index": 0,
                "eval_split_tags": ["heldout_recording"],
                "ground_truth_tokens": ["KEY_PRESS_87"],
            },
            {
                "sequence_id": "rec_001#000001",
                "source_id": "d2e_480p",
                "universe_row_id": "d2e_480p:Game/rec_001",
                "cross_resolution_key": "Game/rec_001",
                "source_recording_id": "rec_001",
                "recording_id": "d2e_480p:Game/rec_001",
                "game": "Game",
                "timestamp_ns": 50,
                "bin_index": 1,
                "eval_split_tags": ["temporal"],
                "ground_truth_tokens": ["MOUSE_DX_P1"],
            },
            {
                "sequence_id": "rec_001#train",
                "source_id": "d2e_480p",
                "universe_row_id": "d2e_480p:Game/rec_001",
                "cross_resolution_key": "Game/rec_001",
                "source_recording_id": "rec_001",
                "recording_id": "d2e_480p:Game/rec_001",
                "game": "Game",
                "timestamp_ns": 0,
                "eval_split_tags": [],
                "ground_truth_tokens": [],
            },
        ],
    )
    pred = tmp_path / "pred.mcap"
    pred.write_bytes(b"mcap")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec_001",
                        "source_id": "d2e_480p",
                        "game": "Game",
                        "recording_id": "rec_001",
                        "row_count": 2,
                        "prediction_mcap_path": str(pred),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "targets.jsonl"
    payload = extract_gidm_target_records(
        manifest_path=manifest,
        by_recording_roots=[tmp_path / "shard_*" / "by_recording"],
        output_path=out,
        summary_out=tmp_path / "summary.json",
        only_existing_predictions=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert payload["status"] == "pass"
    assert payload["rows_written"] == 2
    assert [row["sequence_id"] for row in rows] == ["rec_001#000001", "rec_001#000000"]


def test_chunked_gidm_plan_uses_bin_indices_and_writes_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gidm_inference_manifest.v1",
                "recordings": [
                    {
                        "universe_row_id": "d2e_480p:Game/rec_001",
                        "video_path": "/data/video.mkv",
                        "prediction_mcap_path": str(tmp_path / "predicted" / "rec_001.mcap"),
                        "timestamp_min_ns": 319_503_399_000,
                        "timestamp_max_ns": 320_503_399_000,
                        "bin_index_min": 20,
                        "bin_index_max": 40,
                        "row_count": 21,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plans, paths_by_key = build_gidm_chunk_run_plan(
        manifest_path=manifest,
        cuda_devices=["0", "1"],
        log_dir=tmp_path / "logs",
        chunk_seconds=0.5,
        chunk_context_seconds=0.1,
        bin_ms=50,
        resume=False,
    )

    assert len(plans) >= 3
    assert plans[0].start_time_seconds == 0.9
    assert plans[0].timestamp_offset_seconds == 319.403399
    assert paths_by_key["d2e_480p:Game/rec_001"][0].endswith(".mcap")

    chunk_manifest = tmp_path / "chunk_manifest.json"
    payload = write_chunked_gidm_manifest(
        manifest_path=manifest,
        output_path=chunk_manifest,
        chunk_paths_by_key=paths_by_key,
        chunk_seconds=0.5,
        chunk_context_seconds=0.1,
        bin_ms=50,
    )
    row = payload["recordings"][0]
    assert row["prediction_timestamps_aligned_to_ground_truth"] is True
    assert row["prediction_mcap_paths"] == paths_by_key["d2e_480p:Game/rec_001"]
