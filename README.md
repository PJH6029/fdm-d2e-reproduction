# FDM-D2E: recipe-faithful scaled FDM-1 reproduction on D2E

This repository is a greenfield smoke-scale implementation plan and code scaffold for reproducing the **recipe shape** of FDM-1 using the public D2E dataset.

## Scope guardrails

- This is a **recipe-faithful scaled reproduction**, not an FDM-1 parity claim.
- It does not collect or label an 11M-hour internet-scale corpus.
- It does not target commercial distribution of D2E-derived artifacts; D2E is documented as CC-BY-NC-4.0 upstream.
- It does not include real-world robot/car/hardware demos in the first pass.
- The canonical FDM smoke path must consume an **IDM-generated pseudo-label artifact**. Ground-truth-only training is only an oracle-control path.

## Smoke pipeline

The default smoke pipeline uses a deterministic D2E-shaped fixture so a fresh checkout can run without downloading the full dataset. Real D2E files can be wired into the same contracts after `docs/d2e_source_contract.md` is confirmed in the target environment.

```bash
python scripts/prepare_d2e_smoke.py --config configs/data/d2e_smoke.yaml
python scripts/run_idm_smoke.py --config configs/model/idm_smoke.yaml
python scripts/run_fdm_smoke.py --config configs/model/fdm_smoke.yaml --labels outputs/idm/pseudolabels.jsonl
python scripts/run_eval_smoke.py --predictions outputs/fdm/predictions.jsonl --ground-truth outputs/data/heldout.jsonl
python scripts/run_rollout_smoke.py --mode stub
```

Run tests with either:

```bash
python -m unittest discover -s tests
```

or, if pytest is installed:

```bash
pytest
```

## Artifact chain

1. `outputs/data/manifest.json` and split JSONL files prove the D2E-shaped source contract and splits.
2. `outputs/tokenization/action_vocab.json` and `outputs/tokenization/sample_sequence_pack.json` prove action/video token contracts.
3. `outputs/idm/pseudolabels.jsonl` proves the IDM labeler stage emitted model-generated pseudo-labels.
4. `outputs/fdm/checkpoint_metadata.json` proves the FDM smoke stage consumed the pseudo-label artifact.
5. `outputs/eval/metrics.json` and `outputs/rollout/rollout_smoke.json` prove evaluation and rollout-harness smoke paths.
