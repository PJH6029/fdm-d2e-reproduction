import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_d2e_luma_window_corpus import materialize_luma_window_corpus


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(sequence: str, recording: str, value: float) -> dict:
    return {
        "sequence_id": sequence,
        "recording_id": recording,
        "frame": {"features": [value] * 5, "luma16": [value] * 256},
        "next_frame_features": [value] * 5,
        "frame_delta_features": [0.0] * 5,
        "ground_truth_tokens": ["MOUSE_DX_Z0", "MOUSE_DY_Z0"],
    }


def test_luma_window_materializer_carries_context_across_train_target_boundary(tmp_path: Path):
    source = tmp_path / "outputs/data/d2e_full_corpus_shards_accel64"
    train = source / "shard_00/train_core.jsonl"
    target = source / "shard_00/target_all_eval.jsonl"
    _write_jsonl(train, [_row("rec#000000", "rec", 1.0), _row("rec#000001", "rec", 2.0)])
    _write_jsonl(target, [_row("rec#000002", "rec", 3.0), _row("rec#000003", "rec", 4.0)])

    summary = materialize_luma_window_corpus(
        train_inputs=[train],
        target_inputs=[target],
        input_root=source,
        output_root=tmp_path / "outputs/data/d2e_luma_window5_corpus_shards_accel64",
        summary_path=tmp_path / "artifacts/idm/luma_window_summary.json",
        offsets=(-2, -1, 0, 1, 2),
        workers=1,
    )

    assert summary["status"] == "pass"
    rows = [
        json.loads(line)
        for line in (tmp_path / "outputs/data/d2e_luma_window5_corpus_shards_accel64/shard_00/target_all_eval.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["compact_luma_window_offsets"] == [-2, -1, 0, 1, 2]
    assert [plane[0] for plane in rows[0]["compact_luma_window"]] == [1.0, 2.0, 3.0, 4.0, 0.0]
    assert rows[0]["compact_luma_window_mask"] == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert rows[0]["ground_truth_tokens"] == ["MOUSE_DX_Z0", "MOUSE_DY_Z0"]
