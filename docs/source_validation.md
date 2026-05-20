# Source Validation — G0 FDM-D2E Reproduction

- Generated: 2026-05-20
- Ultragoal story: G001 / G0 source-resource prerequisite validation
- Binding plan: `.omx/plans/prd-fdm-d2e-reproduction.md`, `.omx/plans/test-spec-fdm-d2e-reproduction.md`
- Machine-readable inventory: `artifacts/sources/d2e_hf_inventory.json`
- Local dependency probe: `artifacts/sources/local_dependency_probe.json`
- D2E evaluator summary: `artifacts/sources/d2e_evaluate_summary.json`
- Actual D2E MCAP probe: `artifacts/sources/d2e_sample_mcap_probe.json`
- Actual D2E video probe: `artifacts/sources/d2e_sample_video_ffprobe.json`

## Verdict

G0 source compatibility is sufficient to start G1/G2 implementation. Public D2E repositories expose paired `.mkv`/`.mcap` recordings, the public D2E evaluator confirms the MCAP + 50ms-bin metric contract, and public Generalist-IDM can be used as a teacher/reference baseline if execution chooses that path. The local development environment is not yet a training environment; G2 must install the missing HF/OWA/MCAP/OpenCV/Torch/Transformers/distributed training stack in Docker/MLXP.

## Public source findings

### D2E project and recipe relevance

The D2E project page describes OWA Toolkit, Generalist-IDM, NEP-τ, and public dataset/model/code links. It states that OWA captured hundreds of hours of desktop demonstrations across games and that Generalist-IDM pseudo-labels additional gameplay. It also describes NEP-τ as a future-context temporal offset for inverse dynamics, with τ=100ms used by default in D2E experiments. Source: https://worv-ai.github.io/d2e/.

### D2E GitHub repository

The D2E GitHub README announces:

- release of `evaluate.py` for standardized desktop-IDM metrics;
- Generalist-IDM weights and inference code;
- D2E-480p and D2E-Original datasets;
- 267 hours of synchronized video/audio/input events from 29 PC games for vision-action training.

Source: https://github.com/worv-ai/D2E.

### Hugging Face dataset inventory

Inventory fetched via Hugging Face API on 2026-05-20:

| Repository | License | Paired recordings | Games | Notes |
| --- | --- | ---: | ---: | --- |
| `open-world-agents/D2E-480p` | `cc-by-nc-4.0` | 459 | 29 | 918 video/MCAP files plus metadata; intended for vision-action model training. |
| `open-world-agents/D2E-Original` | `cc-by-nc-4.0` | 459 | 29 | FHD/QHD original-resolution counterpart for world/video models; useful only if storage/time allows. |

Representative game directories include Apex Legends, Barony, Battlefield 6 Open Beta, Brotato, Core Keeper, Counter-Strike 2, Cyberpunk 2077, Minecraft 1.21.8, PUBG, Raft, Stardew Valley, VALORANT, and Vampire Survivors.

### Generalist-IDM inventory

`open-world-agents/Generalist-IDM-1B` is an Apache-2.0 Hugging Face model tagged as image-text-to-text / inverse-dynamics-model / game-ai. The model card exposes Transformers usage and vLLM/SGLang OpenAI-compatible serving examples. It references D2E-480p and D2E-Original datasets. Source: https://huggingface.co/open-world-agents/Generalist-IDM-1B.

Use in this project: teacher/reference baseline and compatibility oracle only. Canonical success still requires training/evaluating this repo's own IDM/FDM path.

### D2E evaluator contract

The public `evaluate.py` script in `worv-ai/D2E` declares:

- dependencies on `mcap-owa-support`, `owa-core`, and `owa-msgs` from the open-world-agents repository;
- comparison of ground-truth and predicted MCAP files;
- non-overlapping 50ms temporal bins by default;
- mouse Pearson/scale-ratio, keyboard per-key accuracy, and mouse-button per-button accuracy;
- topics including `screen`, `keyboard`, and `mouse/raw`.

Execution impact: G1 should normalize D2E records to preserve original MCAP/video references and also produce local token/sequence artifacts. G3 should implement repo-native metrics while preserving compatibility with D2E's MCAP evaluator for cross-checks.


### Actual MCAP probe

G0 downloaded one D2E-480p sample MCAP (`Apex_Legends/0805_01.mcap`, 13.7 MB) to `/tmp/fdm-d2e-sample/` and inspected it with the Python `mcap` reader. The probe sampled 20,000 messages and confirmed topics/schemas for `screen`, `keyboard`, `mouse/raw`, `mouse`, `keyboard/state`, `mouse/state`, and `window`. This validates that G1 can start from real MCAP channel/topic contracts before full OWA decoded-message support is installed. Full decoded event extraction still belongs to G1/G2 with `mcap-owa-support`/`owa-msgs` installed.


### Actual video probe

G1 also probed the paired D2E-480p sample video URL (`Apex_Legends/0805_01.mkv`) with `ffprobe` without storing the 470 MB video in the repo. The stream probe confirmed H.264 video at 448×448 and 60 FPS. This supports the G1/G2 plan to use D2E-480p as the first training target and to keep full video caches on MLXP/PVC storage rather than in git.

### FDM-1 public recipe constraints

The FDM-1 public post describes a three-stage recipe: train an IDM, label a large video corpus with the IDM, then train an autoregressive FDM on interleaved frame/action data. It also describes masked-diffusion inverse dynamics, non-causal labeling, discrete key/scroll tokens, and exponentially binned mouse movements. Source: https://si.inc/posts/fdm1/.

Execution impact: reproduce the recipe shape and test FDM-1-inspired methods, but do not claim parity with FDM-1 scale or closed-source implementation.

## Local dependency probe

Current local environment has command-line basics (`git`, `curl`, `uv`, `docker`, `kubectl`, `ffmpeg`) but lacks the Python packages needed for real D2E/training work:

- missing: `huggingface_hub`, `mcap_owa`, `owa`, `owa.msgs`, `opencv-python/cv2`, `torch`, `transformers`, `accelerate`, `deepspeed`, `mcap`.
- present: `/usr/bin/ffmpeg`, `/home/top321902/.local/bin/uv`, `/usr/local/bin/kubectl`.

Execution impact: G2 must own dependency installation and image strategy before full real-data training. G1 can still implement source inventory and interfaces locally with optional imports guarded by tests.

## Implementation prerequisites for G1/G2

1. Add `configs/data/d2e_real.yaml` with HF repo id, local cache dir, sample recording list, split policy, and output path.
2. Add optional dependency extras for real-data work: `huggingface_hub`, D2E/OWA MCAP packages from the public git source, OpenCV or PyAV/FFmpeg bridge, and schema/statistics dependencies.
3. Preserve a fallback path that can run contract tests without full dataset download.
4. Treat D2E-480p as first training target; D2E-Original is deferred unless storage/time allows.
5. Preserve `cc-by-nc-4.0` metadata in every derived manifest and report.
6. Keep Generalist-IDM as teacher/reference baseline only; do not let it replace local IDM training as canonical proof.

## Open but non-blocking items

- Exact package versions for OWA/MCAP should be pinned during G2 after import/install tests.
- Full D2E download size should be measured in G1/G2 before reserving long production windows.
- D2E API/data may evolve; scripts should record HF revision hashes and file fingerprints.

