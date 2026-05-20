# D2E Source Contract

This spike pins the public D2E assumptions used by the smoke pipeline.

## Public sources

- D2E project: https://worv-ai.github.io/d2e/
- D2E code/readme: https://github.com/worv-ai/D2E
- D2E-480p dataset: https://huggingface.co/datasets/open-world-agents/D2E-480p
- D2E-Original dataset: https://huggingface.co/datasets/open-world-agents/D2E-Original

## Contract

- Dataset records are modeled as paired screen video (`.mkv`) and input-event logs (`.mcap`).
- Public D2E examples use `mcap-owa-support`, `owa-msgs`, and Hugging Face Hub download helpers.
- Relevant event topics include screen frames, keyboard events, raw mouse movement, mouse button events, and scroll-like input where available.
- Timestamps are treated as nanoseconds in local artifacts; evaluation uses 50ms bins to match the public D2E evaluation convention.
- Default smoke execution does not download the full dataset. It uses a deterministic D2E-shaped fixture that preserves paired video/MCAP paths and event semantics.
- Real D2E execution should replace the synthetic fixture paths in `outputs/data/manifest.json` with actual Hugging Face-downloaded files.

## Dependencies to pin during real-data expansion

- `huggingface_hub` for sample downloads.
- `mcap-owa-support` and `owa-msgs` for OWAMcap records.
- FFmpeg/OpenCV or equivalent for video frame decoding.
- Optional `owa-data` if the later implementation chooses the D2E-provided pipeline.

## Scope notice

This source contract supports a recipe-faithful scaled reproduction. It is not an FDM-1 parity claim and does not authorize commercial distribution of D2E-derived artifacts.
