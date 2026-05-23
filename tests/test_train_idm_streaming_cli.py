from __future__ import annotations

from scripts.train_idm_streaming import _summary_message


def test_summary_message_handles_distributed_worker_summary():
    message = _summary_message(
        {
            "schema": "streaming_idm_worker_summary.v1",
            "rank": 2,
            "world_size": 4,
            "status": "worker_complete",
        }
    )

    assert "worker complete" in message
    assert "rank=2" in message
    assert "world_size=4" in message


def test_summary_message_handles_rank0_training_summary():
    message = _summary_message(
        {
            "metadata": {
                "model": "idm",
                "train_records": 10,
                "target_records": 5,
                "metrics_path": "metrics.json",
            },
            "device": "cuda:0",
        }
    )

    assert "trained streaming IDM" in message
    assert "model=idm" in message
    assert "train=10" in message
