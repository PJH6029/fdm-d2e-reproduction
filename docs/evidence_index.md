# Evidence Index

This index names the authoritative evidence artifacts for the current
full-corpus FDM-D2E ultragoal. It is not an FDM-1 parity claim and it does not
claim live commercial-game control.

> Current full-corpus ultragoal is not complete until the final G009 cleanup,
> review, Codex goal completion, and OMX checkpoint are recorded.

For exhaustive byte counts and SHA-256 hashes, use
`artifacts/reproducibility/package_manifest.json`. Multi-GB/TB full-corpus
JSONL artifacts that remain on the MLXP PVC are listed in
`artifacts/reproducibility/external_artifact_manifest.json` with storage URI,
byte count, and hash or deterministic fingerprint evidence.

## Workflow state

| Scope | Evidence |
| --- | --- |
| Ultragoal state | `.omx/ultragoal/goals.json`, `.omx/ultragoal/ledger.jsonl` |
| Operating handoff | `AGENTS.md`, `notes/ultragoal-operating-notes.md` |
| GPU-utilization rule | `notes/gpu-utilization-operating-rule.md` |

## D2E-only data and split gates

| Story | Evidence |
| --- | --- |
| G001 data universe | `artifacts/sources/d2e_full_data_universe_manifest.json`, `docs/d2e_full_data_universe.md` |
| G002 split/leakage | `artifacts/sources/d2e_full_split_contract.json`, `artifacts/sources/d2e_full_split_leakage_report.json`, `docs/d2e_full_split_contract.md` |
| Full-corpus decode | `artifacts/sources/d2e_full_corpus_decode_summary.json` |
| PVC-resident train/eval JSONL | `artifacts/reproducibility/external_artifact_manifest.json` |

## D2E-only training gates

| Story | Evidence |
| --- | --- |
| G003 IDM completion | `artifacts/idm/g003_full_idm_completion_audit.json`, `artifacts/idm/idm_streaming_d2e_full_compact_summary.json` |
| G003 4×H200 run | `artifacts/idm/g003_d2e_full_idm_4xh200_train_run.json`, `artifacts/idm/g003_d2e_full_idm_4xh200_gpu_monitor.csv` |
| G003 split statistics | `artifacts/eval/g003_split_statistical_comparisons_summary.json`, `outputs/idm_streaming_d2e_full_compact/split_*_statistical_comparison.json` |
| G004 FDM completion | `artifacts/fdm/g004_full_fdm_completion_audit.json`, `artifacts/fdm/fdm_streaming_d2e_full_compact_summary.json` |
| G004 4×H200 run | `artifacts/fdm/g004_d2e_full_fdm_4xh200_run.json`, `artifacts/fdm/g004_d2e_full_fdm_4xh200_gpu_monitor.csv` |
| G004 split statistics | `artifacts/eval/g004_split_statistical_comparisons_summary.json`, `outputs/fdm_streaming_d2e_full_compact/split_*_statistical_comparison.json` |

Large D2E JSONL files, IDM prediction streams, FDM train-core pseudolabels, and
FDM prediction streams are intentionally not committed to git. Their hashes or
fingerprints are recorded in the external artifact manifest.

## D2E+aux, evaluation, and runtime gates

| Story | Evidence |
| --- | --- |
| G005 aux namespace/provenance | `artifacts/aux/g005_aux_namespace_manifest.json`, `artifacts/aux/g005_aux_completion_audit.json` |
| G005 ablation | `artifacts/aux/d2e_aux_ablation_summary.json`, `artifacts/eval/g005_split_statistical_comparisons_summary.json` |
| G006 endpoint/failure/claim reports | `artifacts/eval/final_endpoint_statistics.json`, `artifacts/eval/final_failure_analysis.json`, `artifacts/eval/final_claim_taxonomy.json`, `artifacts/eval/g006_completion_audit.json` |
| G007 adapter contract | `artifacts/runtime/g007_runtime_replay_adapter_contract.json`, `artifacts/runtime/g007_completion_audit.json`, `docs/runtime_sdk_adapter.md` |

## Live open-source graphical game suite

| Evidence | Path |
| --- | --- |
| Live suite run summary | `artifacts/harness/g008_repo_live_suite/run_summary.json` |
| Live suite evidence bundle | `artifacts/harness/g008_repo_live_suite/live_suite_evidence.json` |
| Evidence quality gate | `artifacts/harness/g008_live_open_game_suite_evidence_validation.json` |
| G008 completion audit | `artifacts/harness/g008_live_suite_completion_audit.json` |
| Protocol/finalization | `artifacts/harness/g008_live_open_game_suite_protocol.json`, `artifacts/harness/g008_live_open_game_suite_finalization_summary.json` |

G008 evidence covers repo-local open-source Tk graphical mini-games driven
through live X11/xdotool input with trained-checkpoint forward-pass evidence,
videos/replays, latency logs, failure logs, and baseline comparison. It does not
claim commercial-game control.

## Final report and reproducibility package

| Artifact | Purpose |
| --- | --- |
| `docs/final_research_report.md` | Research report and claim-boundary summary |
| `docs/failure_analysis.md` | Consolidated failure analysis |
| `docs/reproducibility_runbook.md` | Reproduction/finalization commands |
| `docs/final_quality_gates.md` | Gate definition and completion policy |
| `artifacts/reproducibility/package_manifest.json` | Exhaustive tracked package hashes |
| `artifacts/reproducibility/external_artifact_manifest.json` | PVC-resident large artifact evidence |
| `artifacts/reproducibility/claim_boundary_audit.json` | Forbidden-claim audit |
| `artifacts/reproducibility/final_quality_gate_audit.json` | Final quality gate audit |
| `artifacts/reproducibility/g009_completion_audit.json` | G009 report/package audit |
| `artifacts/reproducibility/g009_finalization_summary.json` | Non-mutating finalizer summary |
