# D2E Full Generalization Split Contract

- Dataset fingerprint: `a4d96c1b35a770a5ce426a3407af4fbd728ddac9a648cdb4746e2f827f803216`
- Source recording groups: `459`
- Recording variants: `918`
- Games: `29`
- Leakage status: `pass`

## Split counts

- Temporal groups with prefix/tail policy: `459`
- Heldout-recording train rows: `734`
- Heldout-recording heldout rows: `184`
- Heldout-recording groups: `92`
- Heldout-game train rows: `800`
- Heldout-game heldout rows: `118`
- Heldout games: `Barony, Eternal_Return, OguForest, Rainbow_Six, Skul, Vampire_Survivors`

## Leakage checks

| Check | Pass |
| --- | --- |
| heldout_recording_rows_disjoint | True |
| cross_resolution_group_assignment_consistent | True |
| heldout_game_rows_disjoint | True |
| heldout_game_not_in_train | True |
| temporal_policy_covers_all_groups | True |
| temporal_window_policy_disjoint_by_construction | True |

## Contract boundary

This contract assigns recording/resolution variants and policies. It does not decode frames or create training windows; downstream ingestion must apply the temporal prefix/tail policy at window construction time.
