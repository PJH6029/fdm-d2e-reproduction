# Generalist-IDM Baseline Contract

G003 pins the released D2E Generalist-IDM baseline before any new IDM training claims.

## Paper Target

The first renewed IDM target is the paper/repo G-IDM result from D2E Tables 4 and 5.
The six in-distribution G-IDM rows aggregate to the paper tau=100 row:

- Pearson X: `0.796`
- Pearson Y: `0.783`
- Scale ratio X: `1.23` or lower
- Scale ratio Y: `1.31` or lower
- Keyboard accuracy: `0.730`
- Mouse-button accuracy: `0.957`

D2E does not paper-report mouse-button F1 or no-button false-positive rate. Those remain local
postprocessed metrics and must not be fabricated from paper tables.

## Released Model Target

After the paper-target gate is beaten, the next gate is exact-split inference with the released model:

- Model: `open-world-agents/Generalist-IDM-1B`
- Revision: `eae16486df3169aafe4a1d74fb2375185b5dc641`
- Official repo commit: `80e98e26e4dc584ec76fec5789b4a97c275dd032`
- Metric protocol: non-overlapping 50 ms bins, `empty_bins_as_correct=false`
- Inference defaults: 20 Hz screen/mouse resampling, keyboard pass-through, 2048 context, 0.1 s time shift, `fps=60,scale=448:448`

The exact-split baseline must run on the same local heldout target rows and split manifests used
by our renewed evaluation. Paper metrics are not a substitute for this run.

## Fail-Closed Rule

If the HF model, official dependencies, original media, or MCAP conversion path is unavailable,
G006 is blocked with an error log. The fallback may run an official short smoke for diagnosis, but
it cannot satisfy the exact-split released-baseline comparison.
