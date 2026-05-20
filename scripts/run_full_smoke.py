#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fdm_d2e.evidence import write_evidence_index

COMMANDS = [
    [sys.executable, "scripts/prepare_d2e_smoke.py", "--config", "configs/data/d2e_smoke.yaml"],
    [sys.executable, "scripts/run_idm_smoke.py", "--config", "configs/model/idm_smoke.yaml"],
    [sys.executable, "scripts/run_fdm_smoke.py", "--config", "configs/model/fdm_smoke.yaml", "--labels", "outputs/idm/pseudolabels.jsonl"],
    [sys.executable, "scripts/run_eval_smoke.py", "--predictions", "outputs/fdm/predictions.jsonl", "--ground-truth", "outputs/data/heldout.jsonl"],
    [sys.executable, "scripts/run_rollout_smoke.py", "--mode", "stub"],
]


def main() -> None:
    transcript = []
    for cmd in COMMANDS:
        print("+", " ".join(cmd))
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout, end="")
        transcript.append({"cmd": cmd, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/command_transcript.json").write_text(__import__('json').dumps(transcript, indent=2) + "\n")
    write_evidence_index()
    print("evidence index: outputs/evidence_index.json and docs/evidence_index.md")


if __name__ == "__main__":
    main()
