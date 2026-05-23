# G008 Live Open-Source Graphical Game Suite Protocol

This is the strict evidence contract for the renewed G008 live harness bar. It is
separate from the older deterministic repo-local game-adjacent replay harness in
`docs/harness_selection_and_execution.md`.

Machine-readable files:

- Config: `configs/harness/g008_live_open_game_suite.yaml`
- Readiness planner: `scripts/plan_g008_readiness.py`
- Readiness artifact: `artifacts/harness/g008_readiness_plan.json`
- Protocol artifact: `artifacts/harness/g008_live_open_game_suite_protocol.json`
- Validator: `scripts/validate_live_game_suite.py`
- Finalizer: `scripts/finalize_g008_live_suite.py`
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

## Readiness planning

Before attempting a live graphical-game collection, run the non-mutating
readiness planner:

```bash
uv run python scripts/plan_g008_readiness.py --allow-fail
```

The planner checks the live-suite protocol, G003/G004/G007 prerequisite goal
statuses, trained checkpoint metadata, runtime adapter contract, live-suite
documentation, planned launch commands, and available control backends. It writes
`artifacts/harness/g008_readiness_plan.json`. A `blocked` readiness plan is
expected while D2E-only training is still incomplete or when open-source game
binaries are not installed in the execution environment.

Diagnostic flags:

- `--allow-precheckpoint`: downgrade incomplete prerequisite goals to warnings
  for dry-run planning only.
- `--skip-system-checks`: skip local launch-binary/control-backend checks when
  building a plan outside the eventual graphical desktop host.
- `--allow-overwrite-evidence`: allow an existing evidence-validation artifact
  to be overwritten by a later live collection/finalization pass.

The readiness planner does not launch games, collect evidence, validate live
success, or checkpoint G008.

## Current suite candidates

The current terminal G008 target is a repo-local open-source graphical mini-game
suite. This avoids unverifiable commercial-game claims and avoids depending on
large desktop game packages in the MLXP pod while still exercising a real X11
graphical window, focus guard, live `xdotool` key events, frame observation,
action decoding through the runtime SDK, video/replay/latency/failure logs, and
statistical comparison against a no-op baseline.
The runner loads the trained D2E-only FDM checkpoint and performs one FDM
forward pass per live control step. Because the repo-local games are outside the
D2E training distribution, a small visual goal adapter selects safe
task-specific key tokens from the trained checkpoint vocabulary; the evidence
logs both raw FDM tokens and the adapter-selected live tokens.

| Game | Why selected | License/provenance basis | Planned task | Seeds |
| --- | --- | --- | --- | ---: |
| Repo Grid Chase | Small graphical target-navigation game with deterministic visual state and WASD control. | Implemented in `scripts/run_g008_repo_live_suite.py` under the repository license. | `reach_green_goal` | 5 |
| Repo Lane Align | Small graphical lane/gate alignment game using live keyboard control and visible gate state. | Implemented in `scripts/run_g008_repo_live_suite.py` under the repository license. | `align_and_advance_through_gate` | 5 |
| Repo Click Target | Small graphical crosshair/target activation game using WASD + Space live key events. | Implemented in `scripts/run_g008_repo_live_suite.py` under the repository license. | `move_crosshair_and_activate_target` | 5 |

Earlier external candidates (SuperTuxKart, Luanti/Minetest, Xonotic) remain good
future targets, but the current G008 completion path uses the repo-local suite so
the evidence can be reproduced in a headless MLXP pod with `xvfb` and `xdotool`.


## G008 completion audit

Before checkpointing `G008-live-game-suite` complete, run:

```bash
uv run python scripts/validate_g008_live_suite_completion.py
```

During preparation this may be run with `--allow-fail`, but a terminal G008
checkpoint requires `artifacts/harness/g008_live_suite_completion_audit.json` to
report `status == pass`. The audit rejects protocol-only readiness and requires
completed D2E-only training prerequisites, passing G003/G004 completion audits
with full D2E source/resolution tier counts (`d2e_480p=459`,
`d2e_original=459`, `480p=459`, `original_fhd_qhd=459`), reusable
runtime-adapter evidence, trained checkpoint metadata, live evidence validation
status `pass`, at least the configured games/tasks/episodes, statistical
comparison evidence, and hashed video/replay/latency/failure artifacts. If the
trained checkpoint uses `source_namespace=d2e_aux`, the audit additionally
requires `G005-aux-data-best-model` to be complete and
`g005_aux_completion_audit.status == pass`; D2E+aux cannot be used to satisfy a
live control claim before the D2E-only vs D2E+aux gate is proven.

## Evidence required to pass

For each planned game/task/seed episode, the evidence JSON must include:

- `evidence_mode: live_desktop_control` or `live_graphical_game_control`; dry-run,
  deterministic replay, game-adjacent, or missing modes are rejected.
- `status` indicating pass/success/completion.
- `score` and `baseline_score` for baseline win-rate checks.
- `latency.p95_ms` at or below the configured threshold.
- `runtime` metadata proving the policy was attached to a live graphical
  process rather than a replay: allowed `control_backend`, trained-policy
  `agent_mode`, `process_name`, `window_title` matching the suite config,
  existing `checkpoint_path`, existing `adapter_config_path`, action count, and
  start/end timestamps.
- Existing `video_path`, `replay_path`, `latency_log_path`, and `failure_log_path`.

The suite also requires a statistical-comparison artifact and an explicit strong
statistical bar. The artifact must be parseable JSON with the baseline name,
method, adjusted p-value, positive effect size, score delta, and episode count;
an empty artifact plus a boolean is rejected.

```json
{
  "statistical_comparison": {
    "path": "artifacts/harness/<run>/statistical_comparison.json",
    "holm_adjusted_p_lt_0_05": true
  }
}
```

The referenced `statistical_comparison.json` should include at least:

```json
{
  "schema": "live_suite_statistical_comparison.v1",
  "method": "paired_bootstrap_holm",
  "baseline_name": "random_or_noop_smoke_baseline",
  "adjusted_p_value": 0.01,
  "effect_size": 0.25,
  "agent_mean_score": 10.0,
  "baseline_mean_score": 1.0,
  "episode_count": 15,
  "holm_adjusted_p_lt_0_05": true
}
```

The G008 completion audit hashes the episode video/replay/latency/failure
artifacts plus runtime checkpoint and adapter-config artifacts. A precomputed
validation JSON that omits these runtime artifact hashes is not sufficient.


For the repo-local suite, collect evidence in the MLXP pod with a real X11
framebuffer and xdotool control backend, for example:

```bash
xvfb-run -a uv run python scripts/run_g008_repo_live_suite.py \
  --checkpoint-path outputs/fdm_streaming_d2e_full_compact/torch_model/checkpoint.pt \
  --adapter-config-path configs/runtime/game_adapter_demo.yaml \
  --output-dir artifacts/harness/g008_repo_live_suite
uv run python scripts/finalize_g008_live_suite.py \
  --evidence artifacts/harness/g008_repo_live_suite/live_suite_evidence.json
```

The runner must be treated as live open-source graphical-game evidence only for
these repo-local tasks. It does not imply commercial-game control or parity with
FDM-1.

After collecting live evidence, prefer the fail-closed finalizer rather than
manually chaining validators:

```bash
uv run python scripts/finalize_g008_live_suite.py \
  --evidence artifacts/harness/<run>/live_suite_evidence.json
```

The finalizer writes the protocol report, validates the explicit live evidence
to `artifacts/harness/g008_live_open_game_suite_evidence_validation.json`, runs
the G008 completion audit, and writes
`artifacts/harness/g008_live_open_game_suite_finalization_summary.json`. It does
not mutate OMX state; checkpoint `G008-live-game-suite` only after the finalizer
and `artifacts/harness/g008_live_suite_completion_audit.json` report pass.

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
