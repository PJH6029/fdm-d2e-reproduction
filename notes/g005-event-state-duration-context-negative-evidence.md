# G005 event-state-duration IDM negative evidence

Snapshot: 2026-05-27 KST, pod `prod-rsv-jeonghunpark-20260527-aec8fd`, reservation `rsv-jeonghunpark-20260527-aec8fd`, checkout `141cf1f`.

This full-corpus 4xH200 run used `configs/model/idm_streaming_d2e_full_event_state_duration_context_paper_target.yaml` and completed training, full target prediction, paper-compatible metrics, split statistics, W&B sidecar logging, and the fail-closed G005 paper-target audit.

Evidence paths copied locally:

- `artifacts/idm/g005_idm_event_state_duration_context_4xh200_run.json`
- `artifacts/idm/g005_idm_event_state_duration_context_paper_metrics.json`
- `artifacts/idm/g005_idm_event_state_duration_context_paper_target_audit.json`
- `artifacts/eval/g005_idm_event_state_duration_context_split_statistical_comparisons_summary.json`
- `outputs/idm_streaming_d2e_full_event_state_duration_context_paper_target/*` small metadata/metric/report JSON files only; raw predictions, pseudolabels, caches, and checkpoint remain on PVC.

Result: **negative for G005 completion**. The candidate improved mouse motion and strict local FPR versus earlier G005 attempts, but it did not beat paper-reported G-IDM targets.

Aggregate paper-compatible metrics:

| Metric | Actual | Target | Result |
| --- | ---: | ---: | --- |
| mouse pearson x | 0.7502068587 | 0.796 | fail |
| mouse pearson y | 0.6969669280 | 0.783 | fail |
| keyboard key accuracy | 0.2026294331 | 0.730 | fail |
| mouse-button accuracy | 0.1776986467 | 0.957 | fail |
| scale ratio x | 1.0358216696 | <=1.23 | pass |
| scale ratio y | 1.1778988962 | <=1.31 | pass |

Strict local diagnostics:

- mouse-button F1: `0.29363531698271716` (passes the local transition-hazard +0.02 target of `0.29259490490800827`)
- no-button FPR: `0.04803163838603627` (passes <=0.10)

Implication: do not checkpoint `G005-g014-idm-full-paper-target`. The next G005 branch must address paper-compatible keyboard/button accuracy, not only strict local FPR. The strongest current open issue is the mismatch between local strict metrics and paper-compatible event-count metrics for action tokens.
