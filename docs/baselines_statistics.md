# Baselines and Statistical Evaluation

G3 defines the baseline/evaluation contract used before neural IDM/FDM runs.

## Baseline policies

Implemented in `src/fdm_d2e/eval/baselines.py`:

- `noop`: predicts `NOOP` for every window.
- `global_majority`: predicts the most common train token sequence.
- `game_majority`: predicts the most common train token sequence for the same game, falling back to global majority.
- `last_seen_train`: non-oracle replay of the latest train action for the same recording/game.

`scripts/run_baselines_eval.py` writes each baseline prediction JSONL under `outputs/eval/baselines/` and a source-control-safe summary at `artifacts/eval/baseline_stat_eval_sample.json`.

## Primary endpoints

`configs/eval/primary_endpoints.yaml` predeclares:

- keyboard exact category accuracy,
- mouse button exact category accuracy,
- mouse movement Pearson correlation,
- mouse movement scale-ratio distance using `abs(log(scale_ratio))`.

The default reference baseline is `noop`; movement endpoints use `last_seen_train` as their endpoint-local reference because it is the strongest movement-valued non-neural baseline in the current real-D2E preflight.

## Statistical protocol

Implemented in `src/fdm_d2e/eval/statistics.py`:

- group by recording cluster (`recording_id` by default),
- bootstrap candidate-vs-reference deltas over recording clusters,
- report confidence intervals and two-sided bootstrap p-values,
- apply Holm-Bonferroni correction across all computed comparisons.

The local G3 artifact runs this protocol on the decoded D2E sample. Full G4/G5 claims must rerun the same endpoint config on the real heldout split with enough recording clusters for the strong statistical bar.
