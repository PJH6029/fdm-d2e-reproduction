# D2E Full Data Universe Audit

- Generated: 2026-05-21T00:39:27Z
- Manifest fingerprint: `a4d96c1b35a770a5ce426a3407af4fbd728ddac9a648cdb4746e2f827f803216`
- D2E source count: `2`
- Recording variants: `918`
- Unique cross-resolution recordings: `459`
- Games: `29`
- Status counts: `{"included": 918}`

## Sources

| Source | Resolution | Revision | License | Recordings | Games | Size GiB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| d2e_480p | 480p | `f075f7e25df6` | cc-by-nc-4.0 | 459 | 29 | 180.56 |
| d2e_original | original_fhd_qhd | `6ab8c2489b2b` | cc-by-nc-4.0 | 459 | 29 | 1701.40 |

## Storage budget

- Budget: `5.0` TiB
- D2E source total: `1881.96` GiB
- Auxiliary planned bytes: `0`
- Source bytes within budget: `True`
- Requires staged cache or extra storage: `False`
- Working-set policy: Use staged/streaming cache if source total exceeds the 5TiB working-set limit; do not silently drop required D2E sources.

## Auxiliary candidates

| Candidate | License | Status | Supervision |
| --- | --- | --- | --- |
| Atari-HEAD | CC-BY-4.0 | candidate_needs_integration_review | frame-level human keystroke action labels with gaze/reward metadata |
| p-doom/atari-assault-dataset | cc0-1.0 | candidate_needs_integration_review | action-conditioned video prediction frames/actions |
| NetHack Learning Dataset | needs_review | candidate_needs_license_review | state-action trajectories/key actions |
| MineRL | needs_review | candidate_needs_license_review | human gameplay trajectories with actions/rewards |

## Gate verdict

- Inventory status coverage: PASS
- Exclusion audit: PASS; no non-included D2E recording variants are present in the current HF tree.
- Storage: PASS within 5TiB source budget.

## Claim boundary

This audit is an inventory and storage-planning artifact. It does not prove full-corpus training, D2E-only convergence, D2E+aux model quality, or live harness performance.
