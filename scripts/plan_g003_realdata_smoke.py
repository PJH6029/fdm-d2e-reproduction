#!/usr/bin/env python3
"""Plan a tiny real-D2E G003 smoke without running downloads by default."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.config import load_config
from fdm_d2e.data.full_corpus import included_universe_rows, universe_row_id
from fdm_d2e.io_utils import read_json, write_json

DEFAULT_CONFIG = "configs/data/fdm1_g003_realdata_smoke.yaml"
DEFAULT_OUTPUT = "artifacts/cluster/fdm1_g003_realdata_smoke_plan.json"
DEFAULT_SHELL = "artifacts/cluster/fdm1_g003_realdata_smoke.sh"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_plan(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    extract_cfg = load_config(root_path / str(config["extract_config"]))
    universe = read_json(root_path / str(extract_cfg["data_universe"]))
    rows = included_universe_rows(universe, source_ids=["d2e_480p"], resolution_tiers=["480p"])
    selected = rows[: int(config.get("max_recordings", 1))]
    if not selected:
        raise ValueError("no D2E-480p rows available for realdata smoke")
    extract_cmd = [
        "uv", "run", "python", "scripts/extract_d2e_full_corpus.py",
        "--config", str(config["extract_config"]),
        "--output-dir", str(config["output_dir"]),
        "--summary-out", str(config["summary_out"]),
        "--cache-dir", str(config["cache_dir"]),
        "--max-recordings", str(config.get("max_recordings", 1)),
        "--max-bins-per-recording", str(config.get("max_bins_per_recording", 8)),
        "--event-limit", str(config.get("event_limit", 2000)),
        "--video-mode", str(config.get("video_mode", "remote")),
    ]
    if config.get("force", True):
        extract_cmd.append("--force")
    smoke_root = Path(config["artifacts_dir"]).parent
    final_completion = smoke_root / "completion_config.json"
    finalization = smoke_root / "finalization_config.json"
    finalize_cmd = ["uv", "run", "python", "scripts/finalize_g003_fdm1_action_dataset.py", "--config", str(finalization), "--allow-fail"]
    shell_lines = [
        "set -euo pipefail",
        "mkdir -p " + q(smoke_root),
        "# Build smoke-specific finalization/completion configs after extraction.",
        " ".join(q(part) for part in extract_cmd),
        "uv run python - <<'PY'",
        "import json, pathlib",
        f"smoke_root = pathlib.Path({str(smoke_root)!r})",
        f"base_completion = json.loads(pathlib.Path({str(config['completion_config'])!r}).read_text())",
        f"base_final = json.loads(pathlib.Path({str(config['finalization_config'])!r}).read_text())",
        "paths = base_completion['paths']",
        f"paths['decode_summary'] = {str(config['summary_out'])!r}",
        f"paths['fitted_mouse_bins'] = {str(smoke_root / 'fitted_mouse_bins.json')!r}",
        f"paths['fitted_tokenization_config'] = {str(smoke_root / 'fitted_tokenization_config.json')!r}",
        f"paths['action_slots'] = {str(config['action_output_dir'] + '/action_slots.jsonl')!r}",
        f"paths['dataset_summary'] = {str(config['action_output_dir'] + '/dataset_summary.json')!r}",
        f"paths['overflow_summary'] = {str(config['action_output_dir'] + '/overflow_summary.json')!r}",
        f"paths['alignment_summary'] = {str(config['action_output_dir'] + '/alignment_summary.json')!r}",
        f"paths['sequence_pack'] = {str(config['action_output_dir'] + '/sequence_pack.json')!r}",
        f"paths['visual_alignment_audit'] = {str(smoke_root / 'visual_alignment.json')!r}",
        f"paths['visual_alignment_report'] = {str(smoke_root / 'visual_alignment.md')!r}",
        "for key in list(paths):\n    if key.endswith('_slots') and key not in {'action_slots'}:\n        role = key.removesuffix('_slots')\n        paths[key] = " + repr(str(config['action_output_dir'])) + " + '/splits/' + role + '.jsonl'",
        "base_completion['expected_recording_variants'] = 1",
        "base_completion['min_unique_tokens'] = 1",
        "base_completion['min_visual_rows'] = 1",
        f"base_completion['output_path'] = {str(smoke_root / 'completion_audit.json')!r}",
        "base_final['decoded_records'] = " + repr(str(config['output_dir']) + "/all_records.jsonl"),
        f"base_final['completion_config'] = {str(final_completion)!r}",
        f"base_final['action_output_dir'] = {str(config['action_output_dir'])!r}",
        f"base_final['output_path'] = {str(smoke_root / 'finalization_summary.json')!r}",
        f"base_final['paths']['fitted_mouse_bins'] = {str(smoke_root / 'fitted_mouse_bins.json')!r}",
        f"base_final['paths']['fitted_tokenization_config'] = {str(smoke_root / 'fitted_tokenization_config.json')!r}",
        f"base_final['paths']['action_slots'] = {str(config['action_output_dir'] + '/action_slots.jsonl')!r}",
        f"base_final['paths']['dataset_summary'] = {str(config['action_output_dir'] + '/dataset_summary.json')!r}",
        f"base_final['paths']['visual_alignment_audit'] = {str(smoke_root / 'visual_alignment.json')!r}",
        f"base_final['paths']['visual_alignment_report'] = {str(smoke_root / 'visual_alignment.md')!r}",
        "smoke_root.mkdir(parents=True, exist_ok=True)",
        f"pathlib.Path({str(final_completion)!r}).write_text(json.dumps(base_completion, indent=2) + '\\n')",
        f"pathlib.Path({str(finalization)!r}).write_text(json.dumps(base_final, indent=2) + '\\n')",
        "PY",
        " ".join(q(part) for part in finalize_cmd),
    ]
    return {
        "schema": "fdm1_g003_realdata_smoke_plan.v1",
        "canonical_roadmap": "ROADMAP.md",
        "status": "planned",
        "selected_rows": [{"universe_row_id": universe_row_id(row), "game": row.get("game"), "recording_id": row.get("recording_id"), "repo_id": row.get("repo_id")} for row in selected],
        "max_recordings": int(config.get("max_recordings", 1)),
        "max_bins_per_recording": int(config.get("max_bins_per_recording", 8)),
        "event_limit": int(config.get("event_limit", 2000)),
        "extract_command": extract_cmd,
        "shell_lines": shell_lines,
        "claim_boundary": "Tiny real-D2E smoke plan only; not full-corpus G003 completion evidence.",
    }


def write_shell(plan: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/usr/bin/env bash\n" + "\n".join(plan["shell_lines"]) + "\n", encoding="utf-8")
    p.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a tiny real-D2E G003 smoke command path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--shell-out", default=DEFAULT_SHELL)
    args = parser.parse_args()
    plan = build_plan(load_config(args.config))
    write_json(args.output, plan)
    write_shell(plan, args.shell_out)
    print(json.dumps({"status": plan["status"], "output": args.output, "shell_out": args.shell_out, "selected_rows": plan["selected_rows"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
