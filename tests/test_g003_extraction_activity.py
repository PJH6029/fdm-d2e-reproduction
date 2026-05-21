from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_g003_extraction_activity import build_activity_report


def test_g003_extraction_activity_reports_latest_partial_recording(tmp_path):
    shard_root = tmp_path / "outputs/data/d2e_full_corpus_shards_accel64"
    log_dir = tmp_path / "artifacts/sources/g003_accel64"
    rec0 = shard_root / "shard_0/by_recording/d2e_480p/Game/rec_0"
    rec1 = shard_root / "shard_0/by_recording/d2e_480p/Game/rec_1"
    rec0.mkdir(parents=True)
    rec1.mkdir(parents=True)
    (rec0 / "decode_summary.json").write_text("{}\n")
    (rec0 / "features.jsonl").write_text("{}\n")
    (rec1 / "features.tmp").write_text("partial")
    log_dir.mkdir(parents=True)
    (log_dir / "d2e_full_corpus_shard_0.log").write_text(
        '{"decoded": 1, "total_selected": 2, "universe_row_id": "d2e_480p:Game/rec_0"}\n'
        '{"decoded": 2, "total_selected": 2, "universe_row_id": "d2e_480p:Game/rec_1"}\n'
    )

    output = tmp_path / "artifacts/idm/activity.json"
    report = build_activity_report(
        shard_root=shard_root,
        log_dir=log_dir,
        num_shards=2,
        output=output,
        max_latest_dirs=2,
        log_tail=1,
    )

    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded["schema"] == "g003_extraction_activity.v1"
    shard = report["shards"][0]
    assert shard["recording_summary_count"] == 1
    assert shard["last_log_rows"][-1]["universe_row_id"] == "d2e_480p:Game/rec_1"
    assert any(not row["has_decode_summary"] and row["total_bytes"] > 0 for row in shard["latest_recording_dirs"])
    assert report["shards"][1]["latest_recording_dirs"] == []
