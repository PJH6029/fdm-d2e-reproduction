import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_d2e_state_corpus import materialize_state_corpus


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_materializer_carries_train_state_into_target_split(tmp_path: Path):
    source = tmp_path / "outputs/data/d2e_full_corpus_shards_accel64"
    train = source / "shard_00/train_core.jsonl"
    target = source / "shard_00/target_all_eval.jsonl"
    _write_jsonl(
        train,
        [
            {
                "sequence_id": "rec#000000",
                "recording_id": "rec",
                "ground_truth_tokens": ["KEY_PRESS_w", "MOUSE_LEFT_DOWN", "MOUSE_DX_P2", "MOUSE_DY_Z0"],
            }
        ],
    )
    _write_jsonl(
        target,
        [
            {
                "sequence_id": "rec#000001",
                "recording_id": "rec",
                "ground_truth_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0"],
            },
            {
                "sequence_id": "rec#000002",
                "recording_id": "rec",
                "ground_truth_tokens": ["KEY_RELEASE_w", "MOUSE_LEFT_UP"],
            },
        ],
    )
    summary = materialize_state_corpus(
        train_inputs=[train],
        target_inputs=[target],
        input_root=source,
        output_root=tmp_path / "outputs/data/d2e_state_corpus_shards_accel64",
        summary_path=tmp_path / "artifacts/idm/state_summary.json",
        mouse_emit_mode="single",
    )
    assert summary["status"] == "pass"
    out_target = tmp_path / "outputs/data/d2e_state_corpus_shards_accel64/shard_00/target_all_eval.jsonl"
    rows = [json.loads(line) for line in out_target.read_text(encoding="utf-8").splitlines()]
    assert "KEY_DOWN_W" in rows[0]["ground_truth_tokens"]
    assert "MOUSE_LEFT_DOWN" in rows[0]["ground_truth_tokens"]
    assert rows[1]["ground_truth_tokens"] == ["MOUSE_DX_Z0", "MOUSE_DY_Z0"]
    assert rows[0]["raw_event_tokens"] == ["MOUSE_DX_Z0", "MOUSE_DY_Z0"]
