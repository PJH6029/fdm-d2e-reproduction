import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from synthesize_state_streaming_stats import synthesize_stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_synthesizes_state_label_stats_from_feature_seed_prefix(tmp_path: Path):
    config = tmp_path / "config.json"
    train = tmp_path / "state/shard_00/train_core.jsonl"
    seed = tmp_path / "seed_stats.json"
    output = tmp_path / "out/streaming_stats.json"
    summary = tmp_path / "artifacts/stats_summary.json"
    config.write_text(
        json.dumps(
            {
                "train_records": str(train),
                "train_records_glob": str(train),
                "feature_mode": "summary_compact_luma16_pair_shift_time",
                "categorical_min_count": 1,
            }
        ),
        encoding="utf-8",
    )
    seed.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "seed",
                "feature_mode": "summary_compact_luma16_pair_shift_time",
                "input_dim": 5,
                "mean": [1, 2, 3, 4, 5],
                "std": [1, 1, 2, 2, 3],
                "action_history_dim": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        train,
        [
            {
                "sequence_id": "a",
                "recording_id": "rec",
                "game": "game",
                "timestamp_ns": 1,
                "ground_truth_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0", "KEY_DOWN_W"],
            },
            {
                "sequence_id": "b",
                "recording_id": "rec",
                "game": "game",
                "timestamp_ns": 2,
                "ground_truth_tokens": ["MOUSE_DX_P1", "MOUSE_DY_Z0"],
            },
        ],
    )
    payload = synthesize_stats(config, seed_stats_path=seed, output_path=output, summary_path=summary, workers=1)
    stats = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert stats["input_dim"] == 3
    assert stats["mean"] == [1.0, 2.0, 3.0]
    assert stats["std"] == [1.0, 1.0, 2.0]
    assert stats["keyboard_class_counts"] == {'["KEY_DOWN_W"]': 1, "[]": 1}
    assert stats["action_history_dim"] == 0
