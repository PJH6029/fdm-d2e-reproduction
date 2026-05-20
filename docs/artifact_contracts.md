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

The canonical FDM smoke path must consume `idm_pseudolabel.v1`. Ground-truth-only training is only an oracle-control path and must not be used as the canonical completion proof.
