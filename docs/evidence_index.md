# Evidence Index

> Current full-corpus ultragoal is not complete. This index includes
> historical bounded Shooter64 evidence plus current setup artifacts; final
> G001–G009 evidence must be regenerated after G003–G008 complete.

This index records smoke-contract evidence plus the real-D2E H200 IDM/FDM, ablation, harness, and final package artifacts. It is not an FDM-1 parity claim.

D2E-derived artifacts remain non-commercial unless separate rights are provided.

| Artifact | Exists | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `outputs/data/manifest.json` | True | `5a18a855fd693974c9f08a62abc61201adc2c9c5b9ee06ec2dcb3a231b1586b3` | 1117 |
| `outputs/tokenization/action_vocab.json` | True | `768b7d4e704b5513a2a0b36f90c65c0f6b8f7088b4b2f993aefa75e0adc3c901` | 786 |
| `outputs/tokenization/sample_sequence_pack.json` | True | `d564013ba3c9938384591b20faea3e9df9eb14d01e06a02c3979a16b9b23962b` | 5561 |
| `outputs/idm/pseudolabels.jsonl` | True | `67383de38d82aa4e4de6311002d853068104113dcb4165974fcce93c13cd68af` | 5751 |
| `outputs/idm/metrics.json` | True | `962f7d3b7b30b2b472b4bd7dce575cf243fe197c93db85de0c1bd9c3eae49b97` | 202 |
| `outputs/fdm/checkpoint_metadata.json` | True | `9ff205ef87ec52bc49ffb13b13e339be896353f2b158ddb06c25e68f07a51339` | 396 |
| `outputs/fdm/predictions.jsonl` | True | `bd071f61cc4861098d705334b92af80e484f681464e026078dcb74c4d5d31287` | 4501 |
| `outputs/fdm/train_log.json` | True | `82f8f86a5cebdb76fd497a73b6b53bc62650c4f3ce18a3f46ab1a23cbebb68a6` | 296 |
| `outputs/eval/metrics.json` | True | `bdbda6495065ed959ba86b990e0598f7ecfe821232a196ee79945ec6aa412de1` | 411 |
| `outputs/rollout/rollout_smoke.json` | True | `03084a0797e92f3959407ce43b332f42f27b06d40ff93970c4c6a43d0125d197` | 2360 |

## G5 FDM H200 artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json` | `cf99fbc7e5f9a1561663b5b5af29c17894adcb86f8dc7c1f020c3aebc592318e` | 32206 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/checkpoint_metadata.json` | `128fd509de096c4f67e0a2244c56747e21e69311d636fa2269229bc3e6c2e2d0` | 22415 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/metrics.json` | `191e456426892ea97507ea213fa2ee53499950fa889da417023d8fb9407a7a42` | 845 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/statistical_comparison.json` | `6e0a94c2d48a6e50ae6afa843f1c2427f4c6927289c4c4d27d73db3c7c4fdb70` | 6893 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/predictions.jsonl` | `012844afe3c4c8eebe50c8f809247815f7056c8f0b763c5148257229e37d587a` | 392094 |
| `artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/summary.json` | `bff41d0dde60f4c188ba908b2c78099343dc2d73abb5cdb690652f71bc2393f7` | 32528 |
| `artifacts/fdm/fdm_bth05_d2e_train_prediction_scale_calibrated_h200/checkpoint_metadata.json` | `d60888c0271cd32241587b2da9dac6d5a1b779841ee7186f217e5e0c1e4cc92b` | 22671 |
| `artifacts/fdm/fdm_shooter64_fulltrain_button_sweep_h200.json` | `409f5ebce55c7a7134ee2ac6e19da890ff221cc4d3b97f5b92d576054ba4d12d` | 916294 |
| `artifacts/fdm/fdm_shooter64_recall_beta_sweep_h200.json` | `4bb2cffd53ca5732258b1715f27f68758a943422f926bd108b191c38dc1eafce` | 934437 |
| `artifacts/fdm/fdm_knn_shooter64_surface_sweep_h200.json` | `b6f58d189c0fd761423bf89e54aa3d998bcde793ca0a52928161023bebb6abea` | 259229 |

## G6 ablation/scaling artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `artifacts/ablation_scaling/g007_ablation_scaling_summary.json` | `d02a6477a1add5ad7c4a577ef5aaea82f5f7b967ce37c3444efc5dd13df42d3c` | 90844 |
| `docs/ablation_scaling.md` | `881b5e1443170762f5e8674110a495030e6a85da1d47c667376a83bc1c1a5f73` | 2732 |

## G7 harness artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `artifacts/harness/g008_game_harness_eval.json` | `7092f3a21525240b0575cfb20ea3c6a7b0fd192cdabfc060e7eb26757880c9f7` | 13111 |

## G8 final report/package artifacts

| Artifact | SHA-256 | Bytes | Purpose |
| --- | --- | ---: | --- |
| `docs/final_research_report.md` | `ab4331c29638cec8fac14b9caeddc10e52391aff490b48be8ed7d082cf40d2ba` | 6671 | Final research report and result summary |
| `docs/failure_analysis.md` | `cd77f696656010ae78060083a601644ee3eba1a8ea798ac065c9971f6b3e7cbe` | 2210 | Consolidated IDM/FDM failure analysis |
| `docs/reproducibility_runbook.md` | `98d25125f82b23956abdcea5577f5d26b151b4dbd67c631e7b12cde0b6284d3e` | 2505 | Reproduction commands and cluster path |
| `artifacts/reproducibility/package_manifest.json` | `b96b98a655269dc1d0aeae6b173e0e0c38f7b2923e191c514a37d946f8f74046` | 13360 | Hash manifest for final evidence package |
| `artifacts/reproducibility/final_cleanup_review.md` | `83700affd42811904cde86505d797aeaa464a3db2c452c32b4a5f7e10e714dce` | 2171 | Final ai-slop-cleaner no-op cleanup report |
