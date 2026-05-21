# G006 Evaluation and Failure-Analysis Readiness

G006 is the final split-aware evaluation/statistics/failure-analysis gate. It is
not satisfied by historical Shooter64-only reports, deterministic harness smoke,
or a single aggregate metric file.

Machine-readable files:

- Config: `configs/eval/g006_evaluation_readiness.yaml`
- Validator: `scripts/validate_g006_evaluation_readiness.py`
- Module: `src/fdm_d2e/reporting/evaluation_readiness.py`
- Current audit: `artifacts/eval/g006_evaluation_readiness_audit.json`

Run during active work:

```bash
uv run python scripts/validate_g006_evaluation_readiness.py --allow-fail
```

Run without `--allow-fail` before checkpointing G006 complete.

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
