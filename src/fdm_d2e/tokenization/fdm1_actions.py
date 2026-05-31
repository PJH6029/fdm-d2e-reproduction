from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

SPECIAL_ACTION_TOKENS = ("MASK_ACTION", "NO_ACTION", "PAD_ACTION", "BOS_ACTION", "EOS_ACTION", "EVENT_OVERFLOW")
DEFAULT_MOUSE_BOUNDARIES_24 = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
_MOUSE_REPRESENTATIVES = (0,) + DEFAULT_MOUSE_BOUNDARIES_24


def clean_key(value: Any) -> str:
    text = str(value if value is not None else "UNKNOWN").upper()
    aliases = {
        " ": "SPACE",
        "SPACEBAR": "SPACE",
        "CONTROL": "CTRL",
        "CTRL_L": "CTRL",
        "CTRL_R": "CTRL",
        "SHIFT_L": "SHIFT",
        "SHIFT_R": "SHIFT",
        "ALT_L": "ALT",
        "ALT_R": "ALT",
    }
    text = aliases.get(text, text)
    return "".join(ch for ch in text if ch.isalnum() or ch == "_") or "UNKNOWN"


def clean_button(value: Any) -> str:
    text = str(value if value is not None else "UNKNOWN").upper()
    aliases = {"1": "LEFT", "2": "RIGHT", "3": "MIDDLE", "L": "LEFT", "R": "RIGHT", "M": "MIDDLE"}
    text = aliases.get(text, text)
    return "".join(ch for ch in text if ch.isalnum() or ch == "_") or "UNKNOWN"


@dataclass(frozen=True)
class MouseMoveBinner:
    """Signed 49-bin mouse movement tokenizer (24 negative + zero + 24 positive)."""

    boundaries: tuple[float, ...] = DEFAULT_MOUSE_BOUNDARIES_24
    compound: bool = True

    def __post_init__(self) -> None:
        if len(self.boundaries) != 24:
            raise ValueError("FDM-1 default mouse binner expects exactly 24 positive boundaries for 49 bins per axis")
        if sorted(self.boundaries) != list(self.boundaries) or any(float(v) <= 0 for v in self.boundaries):
            raise ValueError("mouse boundaries must be strictly positive and sorted")

    def axis_label(self, value: float | int) -> str:
        value = float(value)
        if value == 0:
            return "Z00"
        sign = "P" if value > 0 else "N"
        mag = abs(value)
        idx = 1
        for boundary in self.boundaries:
            if mag <= boundary:
                break
            idx += 1
        idx = min(idx, len(self.boundaries))
        return f"{sign}{idx:02d}"

    def tokenize(self, dx: float | int, dy: float | int) -> list[str]:
        xbin = self.axis_label(dx)
        ybin = self.axis_label(dy)
        if self.compound:
            return [f"MOUSE_MOVE_BIN_{xbin}_{ybin}"]
        return [f"MOUSE_DX_BIN_{xbin}", f"MOUSE_DY_BIN_{ybin}"]

    def representative(self, axis_label: str) -> int:
        if axis_label == "Z00":
            return 0
        sign = 1 if axis_label.startswith("P") else -1
        idx = int(axis_label[1:])
        idx = max(1, min(idx, len(self.boundaries)))
        lower = 0 if idx == 1 else float(self.boundaries[idx - 2])
        upper = float(self.boundaries[idx - 1])
        return int(round(sign * ((lower + upper) / 2.0)))


def fit_signed_exponential_boundaries(values: Iterable[float | int], *, num_positive_bins: int = 24) -> tuple[float, ...]:
    """Fit monotonic signed-magnitude boundaries from training deltas.

    The first bins stay fine-grained around zero, then quantiles of observed
    non-zero magnitudes fill the remaining positive bins. This preserves the
    ROADMAP requirement that bins are global and train-set-fitted without adding
    a dependency on numpy.
    """

    magnitudes = sorted(abs(float(v)) for v in values if abs(float(v)) > 0)
    if not magnitudes:
        return DEFAULT_MOUSE_BOUNDARIES_24[:num_positive_bins]
    anchors = [1.0, 2.0, 3.0, 4.0]
    boundaries: list[float] = []
    for value in anchors:
        if len(boundaries) < num_positive_bins:
            boundaries.append(value)
    remaining = num_positive_bins - len(boundaries)
    for i in range(remaining):
        q = (i + 1) / max(remaining, 1)
        pos = min(len(magnitudes) - 1, int(round(q * (len(magnitudes) - 1))))
        boundaries.append(max(boundaries[-1] + 1.0, magnitudes[pos]))
    # Ensure exactly num_positive_bins strictly increasing boundaries.
    fixed: list[float] = []
    last = 0.0
    for boundary in boundaries[:num_positive_bins]:
        last = max(float(boundary), last + 1.0)
        fixed.append(last)
    return tuple(fixed)


def fit_signed_exponential_boundaries_from_histogram(
    magnitude_counts: dict[int | float, int],
    *,
    num_positive_bins: int = 24,
) -> tuple[float, ...]:
    """Fit positive signed-magnitude boundaries from a magnitude histogram.

    This is the full-corpus-friendly variant of
    :func:`fit_signed_exponential_boundaries`: D2E can contain tens of millions
    of 50ms rows, so callers should not keep every raw dx/dy in memory just to
    fit global mouse bins.
    """

    weighted = sorted((abs(float(mag)), int(count)) for mag, count in magnitude_counts.items() if abs(float(mag)) > 0 and int(count) > 0)
    if not weighted:
        return DEFAULT_MOUSE_BOUNDARIES_24[:num_positive_bins]
    anchors = [1.0, 2.0, 3.0, 4.0]
    boundaries: list[float] = []
    for value in anchors:
        if len(boundaries) < num_positive_bins:
            boundaries.append(value)
    remaining = num_positive_bins - len(boundaries)
    total = sum(count for _mag, count in weighted)
    cumulative: list[tuple[float, int]] = []
    running = 0
    for mag, count in weighted:
        running += count
        cumulative.append((mag, running))
    for i in range(remaining):
        target = ((i + 1) / max(remaining, 1)) * total
        chosen = cumulative[-1][0]
        for mag, running_count in cumulative:
            if running_count >= target:
                chosen = mag
                break
        boundaries.append(max(boundaries[-1] + 1.0, chosen))
    fixed: list[float] = []
    last = 0.0
    for boundary in boundaries[:num_positive_bins]:
        last = max(float(boundary), last + 1.0)
        fixed.append(last)
    return tuple(fixed)


@dataclass
class BinnedInputEvent:
    timestamp_ns: int
    token: str
    priority: int
    original_index: int
    raw: dict[str, Any] = field(default_factory=dict)


def _event_timestamp(event: dict[str, Any], default: int = 0) -> int:
    for key in ("timestamp_ns", "time_ns", "timestamp"):
        if key in event and event[key] is not None:
            return int(event[key])
    return default


def _event_key(event: dict[str, Any]) -> Any:
    return event.get("key", event.get("vk", event.get("code", event.get("button", "UNKNOWN"))))


def token_from_discrete_event(event: dict[str, Any]) -> str | None:
    etype = str(event.get("type", event.get("topic", ""))).lower()
    if etype in {"keyboard", "key"}:
        event_type = str(event.get("event_type", event.get("action", ""))).lower()
        action = "UP" if event_type in {"release", "up", "keyup"} else "DOWN"
        return f"KEY_{action}_{clean_key(_event_key(event))}"
    if etype in {"mouse_button", "button"}:
        event_type = str(event.get("event_type", event.get("action", ""))).lower()
        action = "UP" if event_type in {"release", "up", "mouseup"} else "DOWN"
        return f"MOUSE_{clean_button(event.get('button', event.get('name', 'UNKNOWN')))}_{action}"
    if etype == "scroll":
        dx = float(event.get("dx", event.get("scroll_x", 0)) or 0)
        dy = float(event.get("dy", event.get("scroll_y", 0)) or 0)
        if abs(dx) >= abs(dy) and dx:
            return "SCROLL_RIGHT" if dx > 0 else "SCROLL_LEFT"
        if dy:
            return "SCROLL_UP" if dy > 0 else "SCROLL_DOWN"
        return "NO_ACTION"
    return None


def _priority(token: str) -> int:
    if token.startswith("MOUSE_") and token.endswith(("_DOWN", "_UP")) and not token.startswith("MOUSE_MOVE_BIN_"):
        return 0
    if token.startswith("KEY_DOWN_"):
        return 1
    if token.startswith("KEY_UP_"):
        return 2
    if token.startswith("SCROLL_"):
        return 3
    return 4


def _mouse_delta(event: dict[str, Any]) -> tuple[float, float]:
    etype = str(event.get("type", event.get("topic", ""))).lower()
    if etype not in {"mouse_move", "mouse/raw", "mouse_raw", "raw_mouse"}:
        return 0.0, 0.0
    dx = float(event.get("dx", event.get("last_x", 0)) or 0)
    dy = float(event.get("dy", event.get("last_y", 0)) or 0)
    return dx, dy


def bin_events(events: Sequence[dict[str, Any]], *, bin_ms: int = 50, start_ns: int | None = None) -> list[dict[str, Any]]:
    """Group timestamped raw events into non-overlapping 50ms bins."""

    if not events:
        return []
    start = int(start_ns if start_ns is not None else min(_event_timestamp(event) for event in events))
    bin_ns = int(bin_ms * 1_000_000)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, event in enumerate(events):
        timestamp = _event_timestamp(event, default=start)
        bin_index = max(0, (timestamp - start) // bin_ns)
        enriched = dict(event)
        enriched.setdefault("timestamp_ns", timestamp)
        enriched["_original_index"] = idx
        grouped[int(bin_index)].append(enriched)
    rows = []
    for bin_index in range(max(grouped) + 1):
        rows.append({"bin_index": bin_index, "start_ns": start + bin_index * bin_ns, "end_ns": start + (bin_index + 1) * bin_ns, "events": grouped.get(bin_index, [])})
    return rows


@dataclass(frozen=True)
class ActionSlotTokenizer:
    k_event_slots: int = 8
    mouse_binner: MouseMoveBinner = field(default_factory=MouseMoveBinner)
    bin_ms: int = 50

    def serialize_bin(self, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
        dx = 0.0
        dy = 0.0
        discrete: list[BinnedInputEvent] = []
        for idx, event in enumerate(events):
            mdx, mdy = _mouse_delta(event)
            dx += mdx
            dy += mdy
            token = token_from_discrete_event(event)
            if token is not None and token != "NO_ACTION":
                discrete.append(BinnedInputEvent(timestamp_ns=_event_timestamp(event), token=token, priority=_priority(token), original_index=int(event.get("_original_index", idx)), raw=event))
        ordered = sorted(discrete, key=lambda item: (item.priority, item.timestamp_ns, item.original_index, item.token))
        overflow_count = max(0, len(ordered) - self.k_event_slots)
        if self.k_event_slots <= 0:
            event_slots: list[str] = []
        elif overflow_count:
            event_slots = [item.token for item in ordered[: max(0, self.k_event_slots - 1)]] + ["EVENT_OVERFLOW"]
        else:
            event_slots = [item.token for item in ordered]
        event_slots.extend(["NO_ACTION"] * max(0, self.k_event_slots - len(event_slots)))
        action_tokens = self.mouse_binner.tokenize(dx, dy) + event_slots
        return {
            "schema": "fdm1_action_slots.v1",
            "bin_ms": int(self.bin_ms),
            "k_event_slots": self.k_event_slots,
            "mouse_dx_sum": dx,
            "mouse_dy_sum": dy,
            "action_tokens": action_tokens,
            "movement_token_count": 1 if self.mouse_binner.compound else 2,
            "event_slots": event_slots,
            "discrete_event_count": len(ordered),
            "overflow_count": overflow_count,
        }

    def serialize_bins(self, binned_events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for row in binned_events:
            payload = self.serialize_bin(row.get("events", []))
            payload.update({k: row[k] for k in ("bin_index", "start_ns", "end_ns") if k in row})
            rows.append(payload)
        return rows

    def mask_for_idm(self, serialized: dict[str, Any]) -> list[str]:
        movement_count = int(serialized.get("movement_token_count", 1))
        return list(serialized["action_tokens"][:movement_count]) + ["MASK_ACTION"] * self.k_event_slots


def next_click_position_targets(
    binned_events: Sequence[dict[str, Any]],
    *,
    horizon_bins: int = 20,
    grid_width: int = 32,
    grid_height: int = 18,
    screen_width: int = 854,
    screen_height: int = 480,
) -> list[str]:
    click_bins: list[tuple[int, int, int]] = []
    for idx, row in enumerate(binned_events):
        for event in row.get("events", []):
            etype = str(event.get("type", event.get("topic", ""))).lower()
            action = str(event.get("event_type", event.get("action", ""))).lower()
            if etype == "mouse_button" and action in {"press", "down", "mousedown"}:
                x = int(event.get("x", event.get("cursor_x", 0)) or 0)
                y = int(event.get("y", event.get("cursor_y", 0)) or 0)
                click_bins.append((idx, x, y))
    targets: list[str] = []
    for idx, _row in enumerate(binned_events):
        target = "NO_CLICK_WITHIN_H"
        for click_idx, x, y in click_bins:
            if idx <= click_idx <= idx + horizon_bins:
                gx = max(0, min(grid_width - 1, int((x / max(1, screen_width)) * grid_width)))
                gy = max(0, min(grid_height - 1, int((y / max(1, screen_height)) * grid_height)))
                target = f"NEXT_CLICK_POSITION_BIN_{gx}_{gy}"
                break
        targets.append(target)
    return targets


def summarize_slot_overflow(serialized_bins: Sequence[dict[str, Any]], *, by_game: Sequence[str] | None = None) -> dict[str, Any]:
    total = len(serialized_bins)
    overflow_bins = sum(1 for row in serialized_bins if int(row.get("overflow_count", 0)) > 0)
    dropped = sum(int(row.get("overflow_count", 0)) for row in serialized_bins)
    by_game_counts: dict[str, Counter[str]] = defaultdict(Counter)
    if by_game is not None:
        for row, game in zip(serialized_bins, by_game):
            by_game_counts[str(game)]["bins"] += 1
            by_game_counts[str(game)]["overflow_bins"] += int(int(row.get("overflow_count", 0)) > 0)
            by_game_counts[str(game)]["overflow_events"] += int(row.get("overflow_count", 0))
    return {
        "schema": "fdm1_action_slot_overflow_summary.v1",
        "bins": total,
        "overflow_bins": overflow_bins,
        "overflow_events": dropped,
        "overflow_rate": (overflow_bins / total) if total else 0.0,
        "per_game": {game: dict(counter) for game, counter in sorted(by_game_counts.items())},
        "recommended_threshold": 0.001,
        "threshold_exceeded": (overflow_bins / total) > 0.001 if total else False,
    }


def detokenize_mouse_move(token: str, *, binner: MouseMoveBinner | None = None) -> tuple[int, int] | None:
    binner = binner or MouseMoveBinner()
    if not token.startswith("MOUSE_MOVE_BIN_"):
        return None
    suffix = token.removeprefix("MOUSE_MOVE_BIN_")
    try:
        xbin, ybin = suffix.split("_", 1)
    except ValueError:
        return None
    return binner.representative(xbin), binner.representative(ybin)
