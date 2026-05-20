# Real D2E Ingestion Contract

G1 promotes the repo from source-only/smoke fixtures to actual D2E decoding.

## Entry points

- `scripts/prepare_d2e_real.py --config configs/data/d2e_real_sample.yaml`
  writes repo/file/split manifests from the Hugging Face D2E inventory.
- `scripts/extract_d2e_real_sample.py --config configs/data/d2e_real_sample_decode.yaml`
  decodes one paired real D2E `.mcap`/`.mkv` sample into ignored training artifacts.

## Produced local artifacts

Raw/derived D2E data is written under `outputs/` and remains ignored by git:

- `outputs/data/real_sample/<game>/<recording>/decoded_events.jsonl`
- `outputs/data/real_sample/<game>/<recording>/frame_features.jsonl`
- `outputs/data/real_sample/<game>/<recording>/all_records.jsonl`
- `outputs/data/real_sample/<game>/<recording>/train.jsonl`
- `outputs/data/real_sample/<game>/<recording>/heldout.jsonl`
- `outputs/data/real_sample/<game>/<recording>/sequence_pack.v2.json`

A source-control-safe summary is copied to `artifacts/sources/d2e_decoded_sample_summary.json`.

## Normalization choices

- MCAP decoding uses `mcap_owa.highlevel.OWAMcapReader` when the D2E optional stack is installed.
- Default action topics are `screen`, `keyboard`, and `mouse/raw`; the higher-level `mouse` topic is opt-in to avoid duplicate raw/button labels.
- Raw mouse flags are mapped to left/right/middle/x button press/release events and wheel deltas.
- Action windows use 50 ms bins by default, matching the D2E evaluation timebase.
- The sample extractor chooses an action-dense window, not the no-op prefix, so contract tests and later training receive meaningful action tokens.
- Video features are extracted from the paired MKV with `ffmpeg` into temporary PPM frames and reduced to RGB/luma statistics; raw frames stay in ignored `outputs/`.

This is still a small decoded sample for contract verification. Full-scale G4/G5 training must expand this path across the selected D2E recordings on MLXP storage.
