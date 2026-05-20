from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import sha256_file, write_json

KEY_ARTIFACTS = [
    "outputs/data/manifest.json",
    "outputs/tokenization/action_vocab.json",
    "outputs/tokenization/sample_sequence_pack.json",
    "outputs/idm/pseudolabels.jsonl",
    "outputs/idm/metrics.json",
    "outputs/fdm/checkpoint_metadata.json",
    "outputs/fdm/predictions.jsonl",
    "outputs/fdm/train_log.json",
    "outputs/eval/metrics.json",
    "outputs/rollout/rollout_smoke.json",
]


def collect_evidence() -> dict[str, Any]:
    artifacts = []
    for name in KEY_ARTIFACTS:
        p = Path(name)
        artifacts.append({
            "path": name,
            "exists": p.exists(),
            "sha256": sha256_file(p) if p.exists() else None,
            "bytes": p.stat().st_size if p.exists() else None,
        })
    return {
        "schema": "evidence_index.v1",
        "objective": "recipe-faithful scaled FDM-1 reproduction smoke pipeline on D2E",
        "non_parity_notice": "scaled recipe reproduction; not FDM-1 parity",
        "non_commercial_notice": "D2E-derived artifacts must honor upstream CC-BY-NC-4.0 non-commercial terms",
        "artifacts": artifacts,
    }


def write_evidence_index(json_path: str = "outputs/evidence_index.json", markdown_path: str = "docs/evidence_index.md") -> dict[str, Any]:
    evidence = collect_evidence()
    write_json(json_path, evidence)
    lines = [
        "# Evidence Index",
        "",
        "This index is generated from the latest smoke run. It documents evidence for the recipe-faithful scaled FDM-1-on-D2E reproduction and is not an FDM-1 parity claim.",
        "",
        "D2E-derived artifacts remain non-commercial unless separate rights are provided.",
        "",
        "| Artifact | Exists | SHA-256 | Bytes |",
        "| --- | --- | --- | ---: |",
    ]
    for art in evidence["artifacts"]:
        lines.append(f"| `{art['path']}` | {art['exists']} | `{art['sha256'] or ''}` | {art['bytes'] or 0} |")
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines) + "\n")
    return evidence
