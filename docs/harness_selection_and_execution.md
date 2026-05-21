# G7 Harness Selection and Execution

> Current G007 runtime SDK work is documented in
> `docs/runtime_sdk_adapter.md`. The deterministic game-adjacent harness below
> remains useful as a replay/stability contract, but the renewed G008 success
> bar requires live open-source/offline graphical games with seeds, videos,
> latency/failure logs, and statistical baseline comparisons.

This milestone evaluates trained FDM prediction tokens in deterministic repo-local game/game-adjacent harnesses. It is not a commercial-game benchmark or FDM-1 parity claim; it is a bounded stability/progress gate for action-sequence execution before heavier desktop/game integrations.

## Candidate catalog

`src/fdm_d2e/rollout/game_harness.py` defines five dependency-free candidates:

| Candidate | Type | Control family |
| --- | --- | --- |
| `grid_target_arena` | grid navigation | WASD key movement |
| `aim_click_arena` | mouse aiming/clicking | binned mouse DX/DY + left click |
| `dodge_runner_arena` | runner survival | lateral A/D movement |
| `pong_paddle_arena` | paddle tracking | W/S movement |
| `combo_door_arena` | key interaction | E/F/Space interaction keys |

All five candidates passed repo-local deterministic install/control probes in `artifacts/harness/g008_game_harness_eval.json`; the G008 threshold required at least three.

## Trained action source

- Model: `fdm_bth05_d2e_train_scale_calibrated`.
- Prediction artifact: `artifacts/fdm/fdm_bth05_d2e_train_scale_calibrated_h200/predictions.jsonl`.
- Prediction SHA-256 in harness output: `012844afe3c4c8eebe50c8f809247815f7056c8f0b763c5148257229e37d587a`.
- Harness config: `configs/harness/g008_game_harness.yaml`.

## Execution result

`uv run python scripts/run_game_harness_eval.py --config configs/harness/g008_game_harness.yaml` produced `artifacts/harness/g008_game_harness_eval.json`.

Quality gate:

| Metric | Result | Threshold |
| --- | ---: | ---: |
| Candidate count | 5 | 5 |
| Install/control probes passing | 5 | 3 |
| Tasks passing | 5 | 5 |
| Environments passing | 5 | 5 |
| Status | pass | pass |

Task results:

| Task | Environment | Progress | Stability | Pass |
| --- | --- | ---: | ---: | --- |
| `grid_forward_right_navigation` | `grid_target_arena` | `0.600` | valid action rate `1.0`, crashes `0` | yes |
| `aim_left_sweep_click_stability` | `aim_click_arena` | `0.893` | valid action rate `1.0`, crashes `0` | yes |
| `dodge_runner_survival` | `dodge_runner_arena` | `1.000` | valid action rate `1.0`, crashes `0` | yes |
| `pong_paddle_tracking` | `pong_paddle_arena` | `1.000` | valid action rate `1.0`, crashes `0` | yes |
| `combo_door_interaction` | `combo_door_arena` | `1.000` | valid action rate `1.0`, crashes `0` | yes |

## Caveats and next integration path

The harnesses are deterministic game-adjacent environments built to test stable execution of D2E-shaped action tokens without external game installs. They prove that the trained checkpoint emits usable keyboard/mouse action streams and that those streams can be applied across multiple game-like control surfaces. They do not prove robust live play in commercial games. The final report should present this as a bounded action-sequence stability result and recommend future work with OS-level window/input adapters for open-source games.
