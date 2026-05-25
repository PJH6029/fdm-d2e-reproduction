# G005 State-Token IDM Failure Snapshot

- Generated: 2026-05-26 KST
- Run: `g005_idm_state_luma_pair_paper_target`
- Evidence:
  - `artifacts/idm/g005_idm_state_luma_pair_paper_metrics.json`
  - `artifacts/idm/g005_idm_state_luma_pair_paper_target_audit.json`
  - `artifacts/idm/g005_idm_state_luma_pair_postrun_finalization_summary.json`
  - W&B run: `https://wandb.ai/pjh6029-seoul-national-university/fdm-d2e-reproduction/runs/prk8by82`

The state-token corpus, luma-pair features, and 4xH200 training completed over
the full D2E target set (`16,698,646` aligned rows), but the model did not beat
paper-reported G-IDM targets. Aggregate paper-compatible metrics were:

- keyboard key accuracy: `0.1808398270` vs target `0.73`
- mouse-button accuracy: `0.6854049083` vs target `0.957`
- mouse-move Pearson x/y: `0.3976691691` / `0.0978742843` vs targets `0.796` / `0.783`
- mouse scale ratio x/y: `1.4005837354` / `6.0380071352` vs max `1.23` / `1.31`

Strict local mouse-button F1 improved to `0.1901259216`, but no-button false
positive rate remained too high at `0.2413612857`. This branch is therefore a
useful failed full-corpus baseline, not a successful paper-target model.

Immediate implication: compact luma state-token features are not sufficient for
the paper target. The next IDM branch should add stronger visual/context features
or use a released/pretrained visual backbone while preserving the same full-corpus
split/evaluation evidence.
