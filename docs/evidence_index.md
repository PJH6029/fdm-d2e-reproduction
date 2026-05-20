# Evidence Index

This index is generated from the latest smoke run. It documents evidence for the recipe-faithful scaled FDM-1-on-D2E reproduction and is not an FDM-1 parity claim.

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
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/summary.json` | `0d6ebc68a2fe8fdb98fd95bf159b1eeb166015be41fd73e7dc1db228099dbd1b` | 32007 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/checkpoint_metadata.json` | `38fa66dd65276ad3d818a6a6de38f86bcd3db440a8bd9a034bbc11f478df65fd` | 22224 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/metrics.json` | `191e456426892ea97507ea213fa2ee53499950fa889da417023d8fb9407a7a42` | 845 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/statistical_comparison.json` | `6e0a94c2d48a6e50ae6afa843f1c2427f4c6927289c4c4d27d73db3c7c4fdb70` | 6893 |
| `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/predictions.jsonl` | `012844afe3c4c8eebe50c8f809247815f7056c8f0b763c5148257229e37d587a` | 392094 |
| `artifacts/fdm/fdm_shooter64_fulltrain_button_sweep_h200.json` | `409f5ebce55c7a7134ee2ac6e19da890ff221cc4d3b97f5b92d576054ba4d12d` | 916294 |
| `artifacts/fdm/fdm_shooter64_recall_beta_sweep_h200.json` | `4bb2cffd53ca5732258b1715f27f68758a943422f926bd108b191c38dc1eafce` | 934437 |
| `artifacts/fdm/fdm_knn_shooter64_surface_sweep_h200.json` | `b6f58d189c0fd761423bf89e54aa3d998bcde793ca0a52928161023bebb6abea` | 259229 |

## G6 ablation/scaling artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `artifacts/ablation_scaling/g007_ablation_scaling_summary.json` | `d78f263fc9996b09f765c54a9b311048f28d776a30269d052f669228c71f854f` | 89249 |
| `docs/ablation_scaling.md` | `13755da33b717b8561caeadc01fb8c15033795ae6e423b5897145b7b86124ae1` | 2615 |
