from __future__ import annotations

from pathlib import Path

from scripts.watch_wandb_gidm_inference import _mcap_status, _planned_prediction_outputs


def test_planned_prediction_outputs_counts_chunked_manifest():
    payload = {
        "recordings": [
            {"prediction_mcap_path": "flat_a.mcap"},
            {"prediction_mcap_paths": ["chunk_a.mcap", "chunk_b.mcap"]},
            {"prediction_mcap_paths": []},
        ]
    }

    planned = _planned_prediction_outputs(payload)

    assert planned == {"planned_recordings": 3, "planned_outputs": 3, "chunked": True}


def test_mcap_status_counts_nested_chunk_outputs(tmp_path: Path):
    predicted = tmp_path / "predicted_mcap"
    flat = predicted / "flat.mcap"
    chunk = predicted / "flat_chunks" / "rec" / "chunk_0000.mcap"
    temp = predicted / "flat_chunks" / "rec" / "chunk_0001.mcap.tmp.123.1"
    zero = predicted / "flat_chunks" / "rec" / "zero.mcap"
    flat.parent.mkdir(parents=True)
    chunk.parent.mkdir(parents=True)
    flat.write_bytes(b"flat")
    chunk.write_bytes(b"chunk")
    temp.write_bytes(b"temp")
    zero.write_bytes(b"")

    status = _mcap_status(predicted)

    assert status["final_mcap_count"] == 2
    assert status["zero_final_mcap_count"] == 1
    assert status["temp_output_count"] == 1
    assert status["final_mcap_bytes"] == len(b"flat") + len(b"chunk")
