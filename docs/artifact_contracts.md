# Artifact Contracts

The smoke pipeline uses explicit artifacts so `$ultragoal`, `$ralph`, and `$team` lanes can work without hidden coupling.

| Artifact | Schema | Producer | Consumer |
| --- | --- | --- | --- |
| Data manifest | `schemas/data_manifest.schema.json` | `prepare_d2e_smoke.py` | data/token/model stages |
| Action vocabulary | `schemas/action_vocab.schema.json` | tokenizer | token/model/eval stages |
| Sequence pack | `schemas/sequence_pack.schema.json` | video/tokenization stage | encoder/IDM/FDM stages |
| IDM pseudo-labels | `schemas/idm_pseudolabel.schema.json` | `run_idm_smoke.py` | canonical FDM smoke training |
| FDM checkpoint metadata | `schemas/fdm_checkpoint_metadata.schema.json` | `run_fdm_smoke.py` | verifier/eval/docs |
| Metrics JSON | `schemas/metrics.schema.json` | `run_eval_smoke.py` | verifier/docs |
| Rollout action | `schemas/rollout_action.schema.json` | rollout harness | verifier/docs |
| Real D2E manifest | `schemas/data_manifest_v2.schema.json` | `prepare_d2e_real.py` | real-data training/eval stages |
| Real D2E recording manifest | `schemas/recording_manifest.schema.json` | `prepare_d2e_real.py` | downloader/cluster stages |
| Real D2E split manifest | `schemas/split_manifest.schema.json` | `prepare_d2e_real.py` | train/eval selection |
| Real D2E window record | `schemas/d2e_window_record.schema.json` | `extract_d2e_real_sample.py` and full extractor | IDM/FDM real training |
| Real D2E sequence pack | `schemas/sequence_pack_v2.schema.json` | real D2E extraction | IDM/FDM real training |

The canonical FDM smoke path must consume `idm_pseudolabel.v1`. Ground-truth-only training is only an oracle-control path and must not be used as the canonical completion proof.
