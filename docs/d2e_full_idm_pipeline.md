# D2E Full-Corpus IDM Pipeline

This is the G003 execution path for the full-corpus D2E-only IDM story. It is
designed for MLXP/PVC execution, not local smoke-only evidence.

## Inputs

- Data universe: `artifacts/sources/d2e_full_data_universe_manifest.json`
- Leakage-safe split contract: `artifacts/sources/d2e_full_split_contract.json`
- Extraction config: `configs/data/d2e_full_corpus.yaml`
- Streaming IDM config: `configs/model/idm_streaming_d2e_full_compact.yaml`

The extraction config includes both `d2e_480p` and `d2e_original`. Do not add a
per-recording/bin cap for final G003 evidence; caps are only for local debugging.

## Cluster command

From the MLXP PVC checkout:

```bash
cd /root/work/code/continuous-gui-poc/fdm-d2e-reproduction
git pull --ff-only origin main
uv sync --frozen --extra d2e --extra test --extra train
NUM_SHARDS=16 bash scripts/run_g003_d2e_full_idm_parallel.sh
```

The parallel script launches disjoint recording-variant extraction shards,
merges split-aware JSONL files, trains a streaming IDM without loading all D2E
windows into GPU memory, and writes a run evidence JSON under `artifacts/idm/`.

Use the sequential `scripts/run_g003_d2e_full_idm.sh` only for debugging. The
uncapped full 480p+original corpus should use parallel shards; `NUM_SHARDS=16`
is the current MLXP setting for a 128-core H200 production pod.

## Expected outputs

- `outputs/data/d2e_full_corpus/all_records.jsonl`
- `outputs/data/d2e_full_corpus/train_core.jsonl`
- `outputs/data/d2e_full_corpus/target_temporal.jsonl`
- `outputs/data/d2e_full_corpus/target_heldout_recording.jsonl`
- `outputs/data/d2e_full_corpus/target_heldout_game.jsonl`
- `outputs/data/d2e_full_corpus/target_all_eval.jsonl`
- `artifacts/sources/d2e_full_corpus_decode_summary.json`
- `outputs/idm_streaming_d2e_full_compact/checkpoint.pt`
- `outputs/idm_streaming_d2e_full_compact/pseudolabels.jsonl`
- `outputs/idm_streaming_d2e_full_compact/predictions.jsonl`
- `outputs/idm_streaming_d2e_full_compact/metrics.json`
- `artifacts/idm/idm_streaming_d2e_full_compact_summary.json`
- `artifacts/idm/g003_d2e_full_idm_run_full_compact.json`
- `artifacts/idm/g003_d2e_full_idm_run_full_compact_parallel.json` for the
  parallel run path.
- Downstream G004 FDM path after these artifacts exist:
  `docs/d2e_full_fdm_pipeline.md`.

## Operational notes from the first MLXP attempt

- Production base image may not include `uv`; install with
  `python3 -m pip install --user uv` and export `PATH="$HOME/.local/bin:$PATH"`.
- Production base image may not include `ffmpeg`; install with
  `apt-get update -y && apt-get install -y ffmpeg`.
- Keep the D2E cache source-namespaced (`.../cache/d2e_480p`,
  `.../cache/d2e_original`) because the two Hugging Face repos share
  game/recording filenames.
- The first sequential uncapped attempt decoded one 480p recording in about
  nine minutes; do not rely on the sequential script for the full corpus.

## Claim boundary

G003 is complete only after the uncapped run consumes all included D2E source
variants from the data universe and produces checkpoint/metrics/pseudolabel
artifacts. The compact-feature trainer is a D2E-only IDM baseline/labeler; it is
not an FDM-1 parity claim and it is not live-game evidence.
