# Runtime SDK and Safe Game Adapter

This is the G007 reusable inference/runtime adapter surface. It is designed so
trained IDM/FDM artifacts can later be connected to open-source/offline
graphical game targets without turning the research code into an unsafe generic
input-injection tool.

## Implemented SDK pieces

- `ActionDecoder`: converts D2E/FDM action tokens into keyboard, mouse-motion,
  and mouse-button commands.
- `RuntimeSafetyConfig`: declares focus guard, kill switch, key/button allow
  lists, action-rate limit, and per-frame mouse-delta clamp.
- `SafeActionAdapter`: wraps a backend with focus guard, kill switch, and
  rate-limiting checks before any action is applied.
- `DryRunInputBackend`: deterministic replay backend used for tests and
  artifact validation; it performs no OS-level injection.
- `LatencyLogger`: records per-action latency rows and p50/p95/max summaries.

## Demo command

The deterministic contract fixture can be run without waiting for G004:

```bash
uv run python scripts/run_runtime_replay_adapter.py \
  --config configs/runtime/game_adapter_contract_fixture.yaml \
  --focus-title "fdm-adapter-demo open-source offline"
```

This writes `artifacts/runtime/g007_runtime_replay_adapter_contract.json` and
proves decode, focus-guard allow path, rate-limit allow path, dry-run backend,
and p50/p95 latency schema on a tiny keyboard/mouse/click action stream.

After a trained FDM predictions artifact exists, use the full demo config:

```bash
uv run python scripts/run_runtime_replay_adapter.py \
  --config configs/runtime/game_adapter_demo.yaml \
  --focus-title "fdm-adapter-demo open-source offline"
```

Expected output:

- `artifacts/runtime/g007_runtime_replay_adapter_demo.json`
- action replay log with applied/blocked counts
- latency summary with p50/p95/max fields
- safety config echo including focus, kill-switch, rate, and mouse clamp

## Open-source/offline target candidates

`configs/runtime/game_adapter_demo.yaml` records three open-source/offline
graphical target candidates for the later G008 live suite:

- `supertuxkart_local_offline`
- `minetest_local_offline`
- `xonotic_local_offline_botmatch`

These are only target candidates until installation, license/provenance,
window-focus, video/replay capture, and live closed-loop evidence are collected.


## G007 completion audit

Before relying on the completed `G007-runtime-sdk-adapter` story, run:

```bash
uv run python scripts/validate_g007_completion.py
```

The audit writes `artifacts/runtime/g007_completion_audit.json` and must report
`status == pass`. It verifies the deterministic dry-run replay contract, safety
settings, latency schema, kill-switch/focus-guard presence, demo target claim
boundaries, and the explicit no-live/no-commercial-game claim boundary.

## Claim boundary

The SDK proves that trained action streams can be decoded, safety-checked,
rate-limited, replayed, and logged through a game-ready adapter interface. It
does **not** by itself prove live game control and it does not support any
commercial-game claim. No G008 live-suite claim is made here. G008 must still run multiple open-source/offline
graphical games with seeds/episodes, latency/failure logs, replay/video hashes,
and statistical baseline comparisons.
