#!/usr/bin/env bash
set -euo pipefail
mkdir -p .omx/tmp/fdm1_g003_realdata_smoke
# Build smoke-specific finalization/completion configs after extraction.
uv run python scripts/extract_d2e_full_corpus.py --config configs/data/fdm1_d2e_480p_full_corpus_extract.yaml --output-dir .omx/tmp/fdm1_g003_realdata_smoke/window_records --summary-out .omx/tmp/fdm1_g003_realdata_smoke/decode_summary.json --cache-dir .omx/tmp/fdm1_g003_realdata_smoke/cache --max-recordings 1 --max-bins-per-recording 8 --event-limit 2000 --video-mode remote --force
uv run python - <<'PY'
import json, pathlib
smoke_root = pathlib.Path('.omx/tmp/fdm1_g003_realdata_smoke')
base_completion = json.loads(pathlib.Path('configs/eval/fdm1_g003_action_dataset_completion.yaml').read_text())
base_final = json.loads(pathlib.Path('configs/data/fdm1_g003_action_dataset_finalization.yaml').read_text())
paths = base_completion['paths']
paths['decode_summary'] = '.omx/tmp/fdm1_g003_realdata_smoke/decode_summary.json'
paths['fitted_mouse_bins'] = '.omx/tmp/fdm1_g003_realdata_smoke/fitted_mouse_bins.json'
paths['fitted_tokenization_config'] = '.omx/tmp/fdm1_g003_realdata_smoke/fitted_tokenization_config.json'
paths['action_slots'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/action_slots.jsonl'
paths['dataset_summary'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/dataset_summary.json'
paths['overflow_summary'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/overflow_summary.json'
paths['alignment_summary'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/alignment_summary.json'
paths['sequence_pack'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/sequence_pack.json'
paths['visual_alignment_audit'] = '.omx/tmp/fdm1_g003_realdata_smoke/visual_alignment.json'
paths['visual_alignment_report'] = '.omx/tmp/fdm1_g003_realdata_smoke/visual_alignment.md'
for key in list(paths):
    if key.endswith('_slots') and key not in {'action_slots'}:
        role = key.removesuffix('_slots')
        paths[key] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots' + '/splits/' + role + '.jsonl'
base_completion['expected_recording_variants'] = 1
base_completion['min_unique_tokens'] = 1
base_completion['min_visual_rows'] = 1
base_completion['output_path'] = '.omx/tmp/fdm1_g003_realdata_smoke/completion_audit.json'
base_final['decoded_records'] = '.omx/tmp/fdm1_g003_realdata_smoke/window_records/all_records.jsonl'
base_final['completion_config'] = '.omx/tmp/fdm1_g003_realdata_smoke/completion_config.json'
base_final['action_output_dir'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots'
base_final['output_path'] = '.omx/tmp/fdm1_g003_realdata_smoke/finalization_summary.json'
base_final['paths']['fitted_mouse_bins'] = '.omx/tmp/fdm1_g003_realdata_smoke/fitted_mouse_bins.json'
base_final['paths']['fitted_tokenization_config'] = '.omx/tmp/fdm1_g003_realdata_smoke/fitted_tokenization_config.json'
base_final['paths']['action_slots'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/action_slots.jsonl'
base_final['paths']['dataset_summary'] = '.omx/tmp/fdm1_g003_realdata_smoke/action_slots/dataset_summary.json'
base_final['paths']['visual_alignment_audit'] = '.omx/tmp/fdm1_g003_realdata_smoke/visual_alignment.json'
base_final['paths']['visual_alignment_report'] = '.omx/tmp/fdm1_g003_realdata_smoke/visual_alignment.md'
smoke_root.mkdir(parents=True, exist_ok=True)
pathlib.Path('.omx/tmp/fdm1_g003_realdata_smoke/completion_config.json').write_text(json.dumps(base_completion, indent=2) + '\n')
pathlib.Path('.omx/tmp/fdm1_g003_realdata_smoke/finalization_config.json').write_text(json.dumps(base_final, indent=2) + '\n')
PY
uv run python scripts/finalize_g003_fdm1_action_dataset.py --config .omx/tmp/fdm1_g003_realdata_smoke/finalization_config.json --allow-fail
