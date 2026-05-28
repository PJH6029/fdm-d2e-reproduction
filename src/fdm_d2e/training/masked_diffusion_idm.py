from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from fdm_d2e.tokenization.actions import _clean_key, bin_delta, token_to_delta_class

FDM1_ACTION_PAD = "<FDM1_ACTION_PAD>"
FDM1_ACTION_MASK = "<FDM1_ACTION_MASK>"
FDM1_ACTION_NOOP = "NOOP"
FDM1_MOUSE_AXIS_BINS = 49
FDM1_MOUSE_AXIS_ZERO_INDEX = FDM1_MOUSE_AXIS_BINS // 2
FDM1_MOUSE_EXP_BASE = 1024.0
SPECIAL_ACTION_TOKENS = (FDM1_ACTION_PAD, FDM1_ACTION_MASK)


@dataclass(frozen=True)
class ActionSlotRecord:
    """Fixed-width action-token slots for masked-diffusion IDM training.

    FDM-1's public IDM description predicts masked action tokens from all frames
    in a non-causal window.  D2E bins can contain multiple keyboard, mouse and
    scroll events, so this record preserves event multiplicity up to the slot
    budget and records overflow explicitly instead of silently pretending that a
    collapsed action set is recipe faithful.
    """

    tokens: tuple[str, ...]
    overflow_tokens: tuple[str, ...]
    source_token_count: int
    max_slots: int

    @property
    def overflow_count(self) -> int:
        return len(self.overflow_tokens)

    @property
    def padded_tokens(self) -> tuple[str, ...]:
        if len(self.tokens) >= self.max_slots:
            return self.tokens[: self.max_slots]
        return self.tokens + (FDM1_ACTION_PAD,) * (self.max_slots - len(self.tokens))


def _clean_button(value: Any) -> str:
    cleaned = "".join(ch for ch in str(value).upper() if ch.isalnum() or ch == "_")
    return cleaned or "UNKNOWN"


def _sign_suffix(class_index: int, *, bins: int = FDM1_MOUSE_AXIS_BINS) -> str:
    zero = bins // 2
    delta = int(class_index) - zero
    if delta == 0:
        return "Z00"
    prefix = "P" if delta > 0 else "N"
    return f"{prefix}{abs(delta):02d}"


def fdm1_mouse_axis_class(
    delta: int | float,
    *,
    screen_extent: int | float,
    bins: int = FDM1_MOUSE_AXIS_BINS,
    exp_base: float = FDM1_MOUSE_EXP_BASE,
) -> int:
    """Map a pixel delta to an inferred FDM-1-style exponential axis bin.

    Public FDM-1 material says mouse movement is split into X/Y, normalized by
    screen width/height, then placed into 49 exponentially-sized bins.  The exact
    private bin edges are not public; this deterministic approximation preserves
    the public shape while recording the approximation in recipe artifacts.
    """

    bins = int(bins)
    if bins < 3 or bins % 2 == 0:
        raise ValueError("FDM-1 mouse axis bins must be an odd integer >= 3")
    value = float(delta)
    if value == 0.0:
        return bins // 2
    extent = max(1.0, float(screen_extent))
    normalized = min(1.0, abs(value) / extent)
    half = bins // 2
    base = max(2.0, float(exp_base))
    magnitude = math.log1p(normalized * (base - 1.0)) / math.log(base)
    bucket = max(1, min(half, int(math.ceil(magnitude * half))))
    return bins // 2 + (bucket if value > 0 else -bucket)


def _suffix_to_class_index(suffix: str, *, bins: int = FDM1_MOUSE_AXIS_BINS) -> int:
    zero = bins // 2
    normalized = str(suffix).upper()
    if normalized in {"Z0", "Z00"}:
        return zero
    if len(normalized) < 2 or normalized[0] not in {"P", "N"}:
        raise ValueError(f"invalid FDM1 mouse-axis suffix: {suffix}")
    magnitude = int(normalized[1:])
    if magnitude < 1 or magnitude > zero:
        raise ValueError(f"FDM1 mouse-axis suffix magnitude out of range: {suffix}")
    return zero + (magnitude if normalized[0] == "P" else -magnitude)


def fdm1_mouse_axis_delta(
    class_index: int,
    *,
    screen_extent: int | float,
    bins: int = FDM1_MOUSE_AXIS_BINS,
    exp_base: float = FDM1_MOUSE_EXP_BASE,
) -> float:
    """Representative pixel delta for an inferred FDM-1-style axis bin."""

    bins = int(bins)
    zero = bins // 2
    offset = int(class_index) - zero
    if offset == 0:
        return 0.0
    half = zero
    bucket = min(half, max(1, abs(offset)))
    base = max(2.0, float(exp_base))
    lower = 0.0 if bucket == 1 else (base ** ((bucket - 1) / half) - 1.0) / (base - 1.0)
    upper = (base ** (bucket / half) - 1.0) / (base - 1.0)
    normalized_midpoint = (lower + upper) / 2.0
    return (1.0 if offset > 0 else -1.0) * normalized_midpoint * max(1.0, float(screen_extent))


def fdm1_mouse_axis_token(axis: str, delta: int | float, *, screen_extent: int | float) -> str:
    normalized_axis = axis.lower()
    if normalized_axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    class_index = fdm1_mouse_axis_class(delta, screen_extent=screen_extent)
    prefix = "FDM1_MOUSE_DX" if normalized_axis == "x" else "FDM1_MOUSE_DY"
    return f"{prefix}_{_sign_suffix(class_index)}"


def _screen_size(row: dict[str, Any], *, default_width: int, default_height: int) -> tuple[int, int]:
    candidates = [row.get("screen"), row.get("frame"), row.get("metadata")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        width = candidate.get("width") or candidate.get("screen_width")
        height = candidate.get("height") or candidate.get("screen_height")
        if width and height:
            return max(1, int(width)), max(1, int(height))
    return int(default_width), int(default_height)


def _tokens_from_events(row: dict[str, Any], *, screen_width: int, screen_height: int) -> list[str]:
    tokens: list[str] = []
    for event in row.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "keyboard":
            action = "PRESS" if event.get("event_type") == "press" else "RELEASE"
            tokens.append(f"KEY_{action}_{_clean_key(str(event.get('key', 'UNKNOWN')))}")
            continue
        if etype == "mouse_button":
            action = "DOWN" if event.get("event_type") == "press" else "UP"
            tokens.append(f"MOUSE_{_clean_button(event.get('button', 'UNKNOWN'))}_{action}")
            continue
        if etype == "mouse_move":
            tokens.append(fdm1_mouse_axis_token("x", event.get("dx", 0), screen_extent=screen_width))
            tokens.append(fdm1_mouse_axis_token("y", event.get("dy", 0), screen_extent=screen_height))
            continue
        if etype == "scroll":
            dy = float(event.get("dy", 0) or 0)
            tokens.append("SCROLL_Z0" if dy == 0 else ("SCROLL_UP" if dy > 0 else "SCROLL_DOWN"))
    return tokens


def _tokens_from_existing_tokens(row: dict[str, Any], *, screen_width: int, screen_height: int) -> list[str]:
    converted: list[str] = []
    for raw in row.get("ground_truth_tokens", []) or []:
        token = str(raw)
        delta_class = token_to_delta_class(token)
        if delta_class is not None and token.startswith("MOUSE_DX_"):
            converted.append(fdm1_mouse_axis_token("x", delta_class, screen_extent=screen_width))
        elif delta_class is not None and token.startswith("MOUSE_DY_"):
            converted.append(fdm1_mouse_axis_token("y", delta_class, screen_extent=screen_height))
        else:
            converted.append(token)
    return converted


def canonical_fdm1_action_tokens(
    row: dict[str, Any],
    *,
    default_width: int = 854,
    default_height: int = 480,
    include_noop: bool = True,
) -> list[str]:
    """Return recipe-shaped action tokens for one D2E bin/window row."""

    width, height = _screen_size(row, default_width=default_width, default_height=default_height)
    tokens = _tokens_from_events(row, screen_width=width, screen_height=height)
    if not tokens:
        tokens = _tokens_from_existing_tokens(row, screen_width=width, screen_height=height)
    tokens = [token for token in tokens if token and token != FDM1_ACTION_PAD]
    if not tokens and include_noop:
        return [FDM1_ACTION_NOOP]
    return tokens


def canonical_action_slot_record(
    row: dict[str, Any],
    *,
    max_slots: int,
    default_width: int = 854,
    default_height: int = 480,
) -> ActionSlotRecord:
    max_slots = max(1, int(max_slots))
    tokens = canonical_fdm1_action_tokens(row, default_width=default_width, default_height=default_height)
    kept = tuple(tokens[:max_slots])
    overflow = tuple(tokens[max_slots:])
    return ActionSlotRecord(tokens=kept, overflow_tokens=overflow, source_token_count=len(tokens), max_slots=max_slots)


def d2e_metric_tokens_from_fdm1_tokens(
    tokens: Iterable[str],
    *,
    screen_width: int = 854,
    screen_height: int = 480,
    drop_noop: bool = True,
) -> list[str]:
    """Convert recipe-shaped FDM1 mouse-bin tokens back to D2E metric tokens.

    The model can train with 49-bin public-recipe mouse tokens while D2E paper
    metrics expect the repo's coarser `MOUSE_DX_*` / `MOUSE_DY_*` token family.
    This adapter is intentionally lossy and auditable: exact D2E metric success
    must be evaluated after this conversion, not on private FDM1 token names.
    """

    converted: list[str] = []
    for raw in tokens:
        token = str(raw)
        if token in SPECIAL_ACTION_TOKENS or (drop_noop and token == FDM1_ACTION_NOOP):
            continue
        if token.startswith("FDM1_MOUSE_DX_"):
            suffix = token.rsplit("_", 1)[-1]
            delta = fdm1_mouse_axis_delta(_suffix_to_class_index(suffix), screen_extent=screen_width)
            converted.append(f"MOUSE_DX_{bin_delta(delta)}")
            continue
        if token.startswith("FDM1_MOUSE_DY_"):
            suffix = token.rsplit("_", 1)[-1]
            delta = fdm1_mouse_axis_delta(_suffix_to_class_index(suffix), screen_extent=screen_height)
            converted.append(f"MOUSE_DY_{bin_delta(delta)}")
            continue
        converted.append(token)
    return converted or ([FDM1_ACTION_NOOP] if not drop_noop else [])


def build_action_vocab(rows: Iterable[dict[str, Any]], *, max_slots: int, min_count: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        record = canonical_action_slot_record(row, max_slots=max_slots)
        for token in record.tokens:
            counts[token] = counts.get(token, 0) + 1
    vocab = [FDM1_ACTION_PAD, FDM1_ACTION_MASK]
    vocab.extend(sorted(token for token, count in counts.items() if count >= int(min_count) and token not in SPECIAL_ACTION_TOKENS))
    return vocab


def corrupt_action_slots(
    slots: Sequence[str],
    *,
    vocab: Sequence[str],
    mask_probability: float,
    random_token_probability: float = 0.0,
    rng: random.Random | None = None,
    force_at_least_one: bool = True,
) -> tuple[list[str], list[bool]]:
    """Corrupt fixed action slots for masked-diffusion IDM training.

    Returns `(corrupted_slots, loss_mask)`.  PAD slots are never loss targets.
    Selected target slots become mask tokens, or occasionally random action
    tokens, matching a BERT/masked-diffusion style denoising objective.
    """

    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")
    if not 0.0 <= random_token_probability <= 1.0:
        raise ValueError("random_token_probability must be in [0, 1]")
    rng = rng or random.Random()
    action_vocab = [token for token in vocab if token not in SPECIAL_ACTION_TOKENS]
    corrupted = list(slots)
    loss_mask = [False for _ in corrupted]
    candidates = [idx for idx, token in enumerate(corrupted) if token != FDM1_ACTION_PAD]
    for idx in candidates:
        if rng.random() <= mask_probability:
            loss_mask[idx] = True
            if action_vocab and rng.random() < random_token_probability:
                corrupted[idx] = rng.choice(action_vocab)
            else:
                corrupted[idx] = FDM1_ACTION_MASK
    if force_at_least_one and candidates and not any(loss_mask):
        idx = rng.choice(candidates)
        loss_mask[idx] = True
        corrupted[idx] = FDM1_ACTION_MASK
    return corrupted, loss_mask


def iterative_unmask_counts(num_masked: int, *, steps: int = 16) -> list[int]:
    """Number of highest-confidence slots to unmask at each denoising step."""

    remaining = max(0, int(num_masked))
    steps = max(1, int(steps))
    counts: list[int] = []
    for step in range(steps):
        steps_left = steps - step
        count = int(math.ceil(remaining / steps_left)) if remaining > 0 else 0
        counts.append(count)
        remaining -= count
    return counts


def select_topk_masked(confidences: Sequence[float], masked: Sequence[bool], *, k: int) -> list[int]:
    """Select the masked slot indices to reveal next by confidence."""

    if len(confidences) != len(masked):
        raise ValueError("confidences and masked must have the same length")
    ranked = sorted(
        ((float(confidences[idx]), idx) for idx, is_masked in enumerate(masked) if is_masked),
        key=lambda item: (-item[0], item[1]),
    )
    return [idx for _, idx in ranked[: max(0, int(k))]]
