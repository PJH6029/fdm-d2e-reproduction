# G008 Live Open-Source Graphical Game Suite Protocol

This is the strict evidence contract for the renewed G008 live harness bar. It is
separate from the older deterministic repo-local game-adjacent replay harness in
`docs/harness_selection_and_execution.md`.

Machine-readable files:

- Config: `configs/harness/g008_live_open_game_suite.yaml`
- Protocol artifact: `artifacts/harness/g008_live_open_game_suite_protocol.json`
- Validator: `scripts/validate_live_game_suite.py`
- Validation module: `src/fdm_d2e/rollout/live_suite.py`
- Contract tests: `tests/test_live_game_suite_contract.py`

## Claim boundary

`protocol_ready` means the live suite is specified; it is **not** live game
success. G008 remains pending until a trained checkpoint is run in live graphical
open-source/offline games and `validate_live_game_suite.py --evidence ...`
returns `quality_gate.status == pass`.

The deterministic `g008_game_harness_eval.json` artifact remains useful as an
action-sequence stability replay, but it cannot satisfy this live-suite gate and
must not be described as live graphical-game control.

## Current suite candidates

| Game | Why selected | License/provenance basis | Planned task | Seeds |
| --- | --- | --- | --- | ---: |
| SuperTuxKart | 3D graphical kart racing with keyboard controls and offline play. | Official site provides desktop downloads; official add-ons/about pages describe it as a free 3D kart racing game. License recorded as GPLv3 code with project-packaged free assets. | `time_trial_finish_lap` | 5 |
| Luanti / Minetest Game | Open-source voxel first-person environment with keyboard/mouse-like control. | Official Luanti site calls it an open-source voxel game creation platform playable solo; Minetest licensing wiki reports LGPL 2.1+ for engine/game. | `navigate_and_collect_block` | 5 |
| Xonotic | Open-source FPS with keyboard/mouse-like control and offline map/bot modes. | Official page says it is free to play/modify under GPLv3+ and available for Linux/Windows/macOS. | `aim_move_and_fire_range` | 5 |

Sources used for the current candidate metadata:

- SuperTuxKart official site: https://www.supertuxkart.net/
- SuperTuxKart add-ons/about page: https://online.supertuxkart.net/about.php
- Luanti official site: https://www.luanti.org/en/
- Minetest/Luanti licensing wiki: https://wiki.minetest.org/Licensing
- Xonotic official site: https://xonotic.org/

## Evidence required to pass

For each planned game/task/seed episode, the evidence JSON must include:

- `evidence_mode: live_desktop_control` or another non-dry-run live mode.
- `status` indicating pass/success/completion.
- `score` and `baseline_score` for baseline win-rate checks.
- `latency.p95_ms` at or below the configured threshold.
- Existing `video_path`, `replay_path`, `latency_log_path`, and `failure_log_path`.

The suite also requires a statistical-comparison artifact and an explicit strong
statistical bar field:

```json
{
  "statistical_comparison": {
    "path": "artifacts/harness/<run>/statistical_comparison.json",
    "holm_adjusted_p_lt_0_05": true
  }
}
```

## Current protocol gate

Current protocol artifact was generated with:

```bash
uv run python scripts/validate_live_game_suite.py \
  --config configs/harness/g008_live_open_game_suite.yaml \
  --output artifacts/harness/g008_live_open_game_suite_protocol.json
```

It reports:

- planned graphical/offline open-source games: 3 / 3
- planned tasks: 3 / 3
- planned seeded episodes: 15, with 5 seeds per task
- required evidence classes: video, replay, latency log, failure log, statistical comparison

Again, this is protocol readiness only. It does not prove trained-model live
control and does not complete G008.
