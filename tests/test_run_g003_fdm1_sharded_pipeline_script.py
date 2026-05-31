from __future__ import annotations

from pathlib import Path


def test_sharded_pipeline_contains_parallel_extract_merge_and_gates():
    text = Path("scripts/run_g003_fdm1_action_dataset_sharded_pipeline.sh").read_text()
    assert "NUM_SHARDS" in text
    assert "MAX_PARALLEL_SHARDS" in text
    assert "scripts/extract_d2e_full_corpus.py" in text
    assert "--shard-index" in text
    assert "--num-shards" in text
    assert "scripts/merge_d2e_full_corpus_shards.py" in text
    assert "scripts/finalize_g003_fdm1_action_dataset.py" in text
    assert "scripts/build_fdm1_g003_checkpoint_handoff.py" in text
    assert "fdm1_g003_sharded_pipeline_summary.v1" in text
    assert "preflight_self_pid_args" in text
    assert "--allow-active-pid" in text
    assert text.index("one or more shard processes failed before merge") < text.index(
        "scripts/merge_d2e_full_corpus_shards.py"
    )
