# G006 Evaluation and Failure-Analysis Readiness

G006 is the final split-aware evaluation/statistics/failure-analysis gate. It is
not satisfied by historical Shooter64-only reports, deterministic harness smoke,
or a single aggregate metric file.

Machine-readable files:

- Config: `configs/eval/g006_evaluation_readiness.yaml`
- Validator: `scripts/validate_g006_evaluation_readiness.py`
- Finalizer: `scripts/finalize_g006_evaluation.py`
- Module: `src/fdm_d2e/reporting/evaluation_readiness.py`
- Current audit: `artifacts/eval/g006_evaluation_readiness_audit.json`

Run during active work:

```bash
uv run python scripts/validate_g006_evaluation_readiness.py --allow-fail
```

Run without `--allow-fail` before checkpointing G006 complete.

For read-only launch/readiness planning before the finalizer, use:

```bash
uv run python scripts/plan_g006_readiness.py --allow-fail
```

This writes `artifacts/eval/g006_readiness_plan.json` and checks prerequisite
goal states, required split-statistical comparison sources, checkpoint metadata
sources, and claim-evidence paths. It is a planner only: it does not build final
artifacts, checkpoint G006, or weaken G003/G004/G005 prerequisites. Missing
live-suite evidence remains a warning while `live_open_game_suite` is explicitly
`not_claimed_until_g008`.

## Required final artifacts

The readiness audit expects these final artifacts to exist only after G003/G004
prerequisites are complete and final evaluation has actually run:

- `artifacts/eval/final_endpoint_statistics.json`
- `artifacts/eval/final_failure_analysis.json`
- `artifacts/eval/final_claim_taxonomy.json`

The final quality gate also checks these artifacts, including `status == pass`
assertions, before aggregate completion.

## Required evaluation coverage

`final_endpoint_statistics.json` must include split-aware comparisons for:

- temporal
- heldout-recording
- heldout-game

and endpoints:

- keyboard accuracy
- mouse-button accuracy
- mouse-button precision
- mouse-button F1
- no-button false-positive rate
- mouse-move Pearson
- mouse-move scale-ratio distance

Each comparison must include model/reference identities, raw candidate/baseline
values, delta, p-value, Holm-adjusted p-value, rejection flag, and source artifact
path/hash.

## Required failure analysis and claims

`final_failure_analysis.json` must report failures by action, game, resolution,
source, and calibration axes, include representative examples, and explicitly
report non-rejections/negative results.

`final_claim_taxonomy.json` must separate D2E-only IDM, D2E-only FDM, D2E+aux
comparison, live open-game suite, and negative-result claims so report wording
cannot silently overclaim.


## G006 completion audit

Before checkpointing `G006-evaluation-failure-analysis` complete, run:

```bash
uv run python scripts/validate_g006_completion.py
```

During upstream G003/G004/G005 execution this may be run with `--allow-fail`, but a
terminal G006 checkpoint requires `artifacts/eval/g006_completion_audit.json` to
report `status == pass`. The audit checks G006 goal state, G003/G004/G005 prerequisite
goals, endpoint statistics, failure analysis, claim taxonomy, readiness audit,
final artifact-build summary, required splits/endpoints, required failure axes,
negative examples/non-rejections, required claim states/evidence paths, and
forbidden claim boundaries.

## Final artifact builder

After G003, G004, and G005 have completed with split-aware statistical comparison
artifacts, build the final G006 evidence files with:

```bash
uv run python scripts/build_g006_final_eval_artifacts.py
```

During active work this command may be run with `--allow-fail`, but do not
checkpoint G006 complete unless all three generated artifacts report
`status == pass`:

- `artifacts/eval/final_endpoint_statistics.json`
- `artifacts/eval/final_failure_analysis.json`
- `artifacts/eval/final_claim_taxonomy.json`

The builder config is `configs/eval/g006_final_artifacts.yaml`. It requires
split-aware G003/G004 comparison sources for temporal, heldout-recording, and
heldout-game splits, plus G005 completion for the D2E+aux comparison claim to
be `claimable`. It intentionally fails if only aggregate or historical
bounded-run statistics are available, or if claimable/documented claims lack
evidence paths.

After G003/G004/G005 are complete and the split-stat artifacts exist, prefer the
fail-closed finalizer:

```bash
uv run python scripts/finalize_g006_evaluation.py
```

The finalizer rebuilds final endpoint statistics, failure analysis, and claim
taxonomy, writes `artifacts/eval/g006_final_artifact_build_summary.json`, runs
the readiness audit, runs the G006 completion audit, and writes
`artifacts/eval/g006_finalization_summary.json`. It does not mutate OMX state;
checkpoint `G006-evaluation-failure-analysis` only after the finalizer and
`artifacts/eval/g006_completion_audit.json` report pass.

For unattended handoff, the non-mutating watcher can poll readiness and run the
same finalizer once all prerequisites and input artifacts are ready:

```bash
nohup uv run python scripts/watch_g006_then_finalize.py \
  --output artifacts/eval/g006_postrun_watcher_summary.json \
  > artifacts/eval/g006_postrun_watcher.log 2>&1 &
```

While inputs are missing it writes `waiting_for_g006_inputs` plus the latest
readiness-plan findings. When ready it runs `scripts/finalize_g006_evaluation.py`
and writes a watcher summary. It never mutates OMX/Codex state.

## Split-aware comparison builder

Before running the final G006 artifact builder, create per-split statistical
comparison files from completed G003/G004 predictions:

```bash
uv run python scripts/build_split_statistical_comparisons.py --config configs/eval/g003_split_statistics.yaml
uv run python scripts/build_split_statistical_comparisons.py --config configs/eval/g004_split_statistics.yaml
```

These commands write `split_temporal_statistical_comparison.json`,
`split_heldout_recording_statistical_comparison.json`, and
`split_heldout_game_statistical_comparison.json` under each model output
directory, plus summaries under `artifacts/eval/`. The primary endpoint config
includes the seven G006-required endpoints, including mouse-button precision/F1
and no-button false-positive rate.
