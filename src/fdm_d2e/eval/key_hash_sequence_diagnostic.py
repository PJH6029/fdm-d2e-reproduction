from __future__ import annotations

import glob
import json
import math
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_DEFAULT_THRESHOLDS = (0.35, 0.5, 0.65, 0.8)
_MODS = (2, 3, 4, 5, 6, 8, 10, 16, 20, 32)


def _loads(line: str) -> dict[str, Any]:
    payload = orjson.loads(line) if orjson is not None else json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL row must be an object")
    return payload


def _expand_paths(patterns: Sequence[str | Path] | str | Path) -> list[Path]:
    items = [patterns] if isinstance(patterns, (str, Path)) else list(patterns)
    out: list[Path] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        if matches:
            out.extend(Path(match) for match in matches)
            continue
        path = Path(item)
        if path.exists():
            out.append(path)
    return out


def _iter_rows(patterns: Sequence[str | Path] | str | Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    rows = 0
    for path in _expand_paths(patterns):
        with path.open("r", encoding="utf-8", buffering=1024 * 1024) as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows is not None and rows >= max_rows:
                    return
                if not line.strip():
                    continue
                try:
                    yield _loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid JSONL row at {path}:{line_no}") from exc
                rows += 1


def _tokens(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None and key == "ground_truth_tokens":
        value = row.get("target_tokens")
    if value is None and key == "predicted_tokens":
        value = row.get("tokens")
    return [str(token) for token in value] if isinstance(value, list) else []


def _split_tags(row: dict[str, Any]) -> list[str]:
    for key in ("eval_split_tags", "split_tags"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return []


def sequence_bin_index(row: dict[str, Any]) -> int:
    seq = str(row.get("sequence_id", ""))
    match = re.search(r"#(\d+)$", seq)
    if match:
        return int(match.group(1))
    try:
        return int(row.get("timestamp_ns") or 0) // 50_000_000
    except (TypeError, ValueError):
        return 0


def _recording_game(row: dict[str, Any]) -> str:
    rec = str(row.get("recording_id") or row.get("source_recording_key") or "")
    if ":" in rec:
        rec = rec.split(":", 1)[1]
    if "/" in rec:
        return rec.split("/", 1)[0]
    return rec or "unknown"


def _holds(row: dict[str, Any]) -> dict[str, int]:
    value = row.get("prior_key_hold_bins")
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _since(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("prior_since_key_transition_bins") or 0))
    except (TypeError, ValueError):
        return 0


def _bucket(value: int) -> int:
    value = max(0, int(value))
    if value <= 10:
        return value
    if value <= 20:
        return 10 + ((value - 10) // 2)
    if value <= 80:
        return 15 + ((value - 20) // 5)
    return min(40, 27 + ((value - 80) // 20))


def _key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(t) for t in tokens if str(t).startswith("KEY_")]


def _key_codes_from_tokens(tokens: Sequence[str]) -> set[str]:
    codes: set[str] = set()
    for token in tokens:
        text = str(token)
        for prefix in ("KEY_PRESS_", "KEY_RELEASE_", "KEY_DOWN_"):
            if text.startswith(prefix):
                code = text[len(prefix) :]
                if code:
                    codes.add(code)
                break
    return codes


def _non_key_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(t) for t in tokens if not str(t).startswith("KEY_")]


def _quantized(value: float, *, scale: float = 10.0, cap: int = 20) -> int:
    return max(-cap, min(cap, int(round(float(value) * scale))))


def _visual_hash_features(row: dict[str, Any]) -> list[str]:
    frame = row.get("frame") if isinstance(row.get("frame"), dict) else {}
    cur = frame.get("luma16") if isinstance(frame, dict) else None
    nxt = row.get("next_frame_luma16")
    if not isinstance(cur, list) or not isinstance(nxt, list) or len(cur) != 256 or len(nxt) != 256:
        return []
    try:
        cur_f = [float(v) for v in cur]
        nxt_f = [float(v) for v in nxt]
    except (TypeError, ValueError):
        return []
    delta = [b - a for a, b in zip(cur_f, nxt_f)]
    feats = [
        f"lum_mean={_quantized(sum(cur_f) / 256.0, scale=20)}",
        f"next_lum_mean={_quantized(sum(nxt_f) / 256.0, scale=20)}",
        f"delta_mean={_quantized(sum(delta) / 256.0, scale=50)}",
        f"delta_abs={_quantized(sum(abs(v) for v in delta) / 256.0, scale=50)}",
        f"delta_pos_frac={_quantized(sum(1 for v in delta if v > 0.01) / 256.0, scale=10)}",
        f"delta_neg_frac={_quantized(sum(1 for v in delta if v < -0.01) / 256.0, scale=10)}",
    ]
    # 4x4 block transition sketch.  This keeps visual context cheap enough for
    # CPU prefix screening while exposing localized motion/brightness changes.
    for by in range(4):
        for bx in range(4):
            vals = []
            cur_vals = []
            for y in range(by * 4, (by + 1) * 4):
                for x in range(bx * 4, (bx + 1) * 4):
                    idx = y * 16 + x
                    vals.append(delta[idx])
                    cur_vals.append(cur_f[idx])
            mean = sum(vals) / len(vals)
            abs_mean = sum(abs(v) for v in vals) / len(vals)
            cur_mean = sum(cur_vals) / len(cur_vals)
            feats.append(f"block{by}{bx}_cur={_quantized(cur_mean, scale=10)}")
            feats.append(f"block{by}{bx}_d={_quantized(mean, scale=50)}")
            feats.append(f"block{by}{bx}_ad={_quantized(abs_mean, scale=50)}")
    return feats


def _features(row: dict[str, Any], code: str, hold: int, *, include_visual_hash: bool = False) -> list[str]:
    idx = sequence_bin_index(row)
    since = _since(row)
    hbin = _bucket(hold)
    sbin = _bucket(since)
    prev = _tokens(row, "previous_event_tokens")
    prior_codes = sorted(_holds(row))
    held_now = int(code in prior_codes)
    feats = [
        "bias",
        f"code={code}",
        f"game={_recording_game(row)}",
        f"held={held_now}",
        f"code={code}|held={held_now}",
        f"code={code}|hbin={hbin}",
        f"code={code}|sbin={sbin}",
        f"hbin={hbin}|sbin={sbin}",
        f"code={code}|hbin={hbin}|sbin={sbin}",
        f"held_count={min(len(prior_codes), 8)}",
    ]
    for mod in _MODS:
        feats.append(f"phase{mod}={idx % mod}")
        feats.append(f"code={code}|phase{mod}={idx % mod}")
        feats.append(f"code={code}|hmod{mod}={hold % mod}")
        feats.append(f"code={code}|smod{mod}={since % mod}")
    prev_set = set(prev)
    feats.append(f"prev_press_same={int('KEY_PRESS_' + code in prev_set)}")
    feats.append(f"prev_release_same={int('KEY_RELEASE_' + code in prev_set)}")
    feats.append(f"prev_has_any_key={int(any(t.startswith('KEY_') for t in prev))}")
    for token in prev:
        if token.startswith("KEY_"):
            feats.append(f"prev_key={token}")
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            feats.append(f"prev_button={token}")
    for other in prior_codes[:12]:
        if other != code:
            feats.append(f"held_with={other}")
            feats.append(f"code={code}|held_with={other}")
    if include_visual_hash:
        for visual in _visual_hash_features(row):
            feats.append(f"vis={visual}")
            feats.append(f"code={code}|vis={visual}")
    return feats


def _candidate_codes(row: dict[str, Any], key_vocab: Sequence[str] = ()) -> list[str]:
    codes = set(_holds(row))
    codes.update(_key_codes_from_tokens(_tokens(row, "previous_event_tokens")))
    codes.update(_key_codes_from_tokens(_tokens(row, "prior_action_tokens")))
    codes.update(str(code) for code in key_vocab)
    return sorted(code for code in codes if code)


def collect_top_key_codes(
    train_paths: Sequence[str | Path] | str | Path,
    *,
    max_train_rows: int | None = 320_000,
    limit: int = 0,
) -> tuple[list[str], int]:
    if int(limit) <= 0:
        return [], 0
    counts: Counter[str] = Counter()
    rows = 0
    for row in _iter_rows(train_paths, max_rows=max_train_rows):
        rows += 1
        counts.update(_key_codes_from_tokens(_tokens(row, "ground_truth_tokens")))
    return [code for code, _count in counts.most_common(int(limit))], rows


class HashedLogisticKeyModel:
    def __init__(
        self,
        *,
        dim: int = 1 << 18,
        learning_rate: float = 0.05,
        l2: float = 0.0,
        include_visual_hash: bool = False,
        candidate_key_codes: Sequence[str] = (),
    ) -> None:
        self.dim = int(dim)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.include_visual_hash = bool(include_visual_hash)
        self.candidate_key_codes = tuple(str(code) for code in candidate_key_codes)
        self.press_weights = [0.0] * self.dim
        self.release_weights = [0.0] * self.dim
        self.examples = 0
        self.positive_press = 0
        self.positive_release = 0

    def _indices(self, features: Sequence[str]) -> list[int]:
        return [zlib.crc32(feature.encode("utf-8")) % self.dim for feature in features]

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score > 30:
            return 1.0
        if score < -30:
            return 0.0
        return 1.0 / (1.0 + math.exp(-score))

    @staticmethod
    def _score(weights: list[float], indices: Sequence[int]) -> float:
        return sum(weights[idx] for idx in indices)

    def _update(self, weights: list[float], indices: Sequence[int], target: int, class_weight: float) -> None:
        pred = self._sigmoid(self._score(weights, indices))
        grad = (float(target) - pred) * float(class_weight)
        lr = self.learning_rate
        decay = 1.0 - lr * self.l2 if self.l2 else 1.0
        for idx in indices:
            weights[idx] = (weights[idx] * decay) + (lr * grad)

    def observe(self, row: dict[str, Any]) -> None:
        gt = set(_tokens(row, "ground_truth_tokens"))
        holds = _holds(row)
        codes = set(_candidate_codes(row, self.candidate_key_codes))
        # Include current positives during training so non-held press/release
        # examples can teach the binary specialist.  Prediction never sees this.
        codes.update(_key_codes_from_tokens(gt))
        for code in sorted(codes):
            hold = holds.get(code, 0)
            indices = self._indices(_features(row, code, hold, include_visual_hash=self.include_visual_hash))
            press = int(f"KEY_PRESS_{code}" in gt)
            release = int(f"KEY_RELEASE_{code}" in gt)
            self.positive_press += press
            self.positive_release += release
            # Positives are rare, so give them a modest fixed boost without making
            # every held key fire.  This is still a diagnostic, not final training.
            self._update(self.press_weights, indices, press, class_weight=8.0 if press else 1.0)
            self._update(self.release_weights, indices, release, class_weight=4.0 if release else 1.0)
            self.examples += 1

    def predict(self, row: dict[str, Any], *, press_threshold: float, release_threshold: float) -> list[str]:
        out: list[str] = []
        holds = _holds(row)
        for code in _candidate_codes(row, self.candidate_key_codes):
            hold = holds.get(code, 0)
            indices = self._indices(_features(row, code, hold, include_visual_hash=self.include_visual_hash))
            if self._sigmoid(self._score(self.press_weights, indices)) >= float(press_threshold):
                out.append(f"KEY_PRESS_{code}")
            if self._sigmoid(self._score(self.release_weights, indices)) >= float(release_threshold):
                out.append(f"KEY_RELEASE_{code}")
        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for token in out:
            if token not in seen:
                deduped.append(token)
                seen.add(token)
        return deduped


def train_hashed_key_model(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 320_000,
    epochs: int = 1,
    dim: int = 1 << 18,
    learning_rate: float = 0.05,
    include_visual_hash: bool = False,
    candidate_key_count: int = 0,
) -> tuple[HashedLogisticKeyModel, int]:
    key_codes, _vocab_rows = collect_top_key_codes(train_paths, max_train_rows=max_train_rows, limit=candidate_key_count)
    model = HashedLogisticKeyModel(
        dim=dim,
        learning_rate=learning_rate,
        include_visual_hash=include_visual_hash,
        candidate_key_codes=key_codes,
    )
    rows = 0
    for _epoch in range(int(epochs)):
        rows = 0
        for row in _iter_rows(train_paths, max_rows=max_train_rows):
            rows += 1
            model.observe(row)
    return model, rows


def _new_accs(split_tags: Sequence[str]) -> dict[str, _PaperMetricAccumulator]:
    out = {"all": _PaperMetricAccumulator(empty_bins_as_correct=False)}
    for tag in split_tags:
        out[f"eval_split:{tag}"] = _PaperMetricAccumulator(empty_bins_as_correct=False)
    return out


def _metrics(accs: dict[str, _PaperMetricAccumulator]) -> dict[str, Any]:
    return {name: acc.metrics() for name, acc in sorted(accs.items())}


def _summary(group: dict[str, Any]) -> dict[str, Any]:
    pc = group["paper_compatible"]
    strict = group["strict_local"]
    return {
        "keyboard_accuracy": pc["keyboard"].get("key_accuracy"),
        "mouse_button_accuracy": pc["mouse_button"].get("button_accuracy"),
        "pearson_x": pc["mouse_move"].get("pearson_x"),
        "pearson_y": pc["mouse_move"].get("pearson_y"),
        "scale_ratio_x": pc["mouse_move"].get("scale_ratio_x"),
        "scale_ratio_y": pc["mouse_move"].get("scale_ratio_y"),
        "strict_mouse_button_f1": strict["mouse_button"].get("f1"),
        "strict_no_button_fpr": strict["mouse_button"].get("no_button_false_positive_rate"),
    }


def build_key_hash_sequence_diagnostic(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    target_paths: Sequence[str | Path] | str | Path,
    base_prediction_paths: Sequence[str | Path] | str | Path,
    output_prediction_path: str | Path | None = None,
    max_train_rows: int | None = 320_000,
    max_target_rows: int | None = 50_000,
    epochs: int = 1,
    dim: int = 1 << 18,
    learning_rate: float = 0.05,
    include_visual_hash: bool = False,
    candidate_key_count: int = 0,
    press_thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    release_thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    model, train_rows = train_hashed_key_model(
        train_paths=train_paths,
        max_train_rows=max_train_rows,
        epochs=epochs,
        dim=dim,
        learning_rate=learning_rate,
        include_visual_hash=include_visual_hash,
        candidate_key_count=candidate_key_count,
    )
    policies: list[tuple[str, str, float | None, float | None]] = [("base_all", "base", None, None)]
    for press_th in press_thresholds:
        for release_th in release_thresholds:
            policies.append((f"replace_base_keys_press{press_th:g}_release{release_th:g}", "replace", float(press_th), float(release_th)))
            policies.append((f"union_base_keys_press{press_th:g}_release{release_th:g}", "union", float(press_th), float(release_th)))
            policies.append((f"press_only_union_base_keys_press{press_th:g}", "press_union", float(press_th), None))
    # Preserve insertion order but remove duplicate press_only labels emitted for each release threshold.
    seen: set[str] = set()
    unique_policies: list[tuple[str, str, float | None, float | None]] = []
    for spec in policies:
        if spec[0] not in seen:
            unique_policies.append(spec)
            seen.add(spec[0])
    accs = {name: _new_accs(split_tags) for name, _mode, _p, _r in unique_policies}
    alignment = {"sequence_id_mismatches": 0, "missing_base_prediction_rows": 0, "examples": []}
    rows = 0
    pred_handle = None
    if output_prediction_path is not None:
        out_path = Path(output_prediction_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = out_path.open("w", encoding="utf-8")
    base_iter = iter(_iter_rows(base_prediction_paths, max_rows=max_target_rows))
    try:
        for row in _iter_rows(target_paths, max_rows=max_target_rows):
            try:
                base = next(base_iter)
            except StopIteration:
                base = {}
                alignment["missing_base_prediction_rows"] += 1
            rows += 1
            if base.get("sequence_id") is not None and row.get("sequence_id") is not None and base.get("sequence_id") != row.get("sequence_id"):
                alignment["sequence_id_mismatches"] += 1
                if len(alignment["examples"]) < 20:
                    alignment["examples"].append({"prediction_sequence_id": base.get("sequence_id"), "target_sequence_id": row.get("sequence_id")})
            base_tokens = _tokens(base, "predicted_tokens")
            non_key = _non_key_tokens(base_tokens)
            tags = set(_split_tags(row))
            gt = _tokens(row, "ground_truth_tokens")
            best_tokens_for_export: list[str] | None = None
            for name, mode, press_th, release_th in unique_policies:
                if mode == "base":
                    pred_tokens = base_tokens
                else:
                    specialist = model.predict(
                        row,
                        press_threshold=float(press_th),
                        release_threshold=1.1 if release_th is None else float(release_th),
                    )
                    if mode == "press_union":
                        specialist = [token for token in specialist if token.startswith("KEY_PRESS_")]
                    if mode in {"union", "press_union"}:
                        key_tokens = []
                        seen_keys: set[str] = set()
                        for token in _key_tokens(base_tokens) + specialist:
                            if token not in seen_keys:
                                key_tokens.append(token)
                                seen_keys.add(token)
                    else:
                        key_tokens = specialist
                    pred_tokens = non_key + key_tokens
                accs[name]["all"].update(pred_tokens, gt)
                for tag in split_tags:
                    if tag in tags:
                        accs[name][f"eval_split:{tag}"].update(pred_tokens, gt)
                if best_tokens_for_export is None and name != "base_all":
                    best_tokens_for_export = pred_tokens
            if pred_handle is not None:
                pred_handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": best_tokens_for_export or base_tokens}, separators=(",", ":")) + "\n")
    finally:
        if pred_handle is not None:
            pred_handle.close()
    policy_payloads: dict[str, Any] = {}
    for name, group_accs in accs.items():
        groups = _metrics(group_accs)
        policy_payloads[name] = {"groups": groups, "summary": _summary(groups["all"])}
    ranked = sorted(
        ({"policy": name, **payload["summary"]} for name, payload in policy_payloads.items()),
        key=lambda item: (
            item.get("keyboard_accuracy") if item.get("keyboard_accuracy") is not None else -1.0,
            item.get("mouse_button_accuracy") if item.get("mouse_button_accuracy") is not None else -1.0,
            item.get("pearson_x") if item.get("pearson_x") is not None else -1.0,
        ),
        reverse=True,
    )
    return {
        "schema": "g005_key_hash_sequence_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": train_rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "epochs": int(epochs),
        "dim": int(dim),
        "learning_rate": float(learning_rate),
        "include_visual_hash": bool(include_visual_hash),
        "candidate_key_count": int(candidate_key_count),
        "candidate_key_codes": list(model.candidate_key_codes),
        "model_examples": model.examples,
        "model_positive_press": model.positive_press,
        "model_positive_release": model.positive_release,
        "alignment": alignment,
        "policies": policy_payloads,
        "ranked_policies": ranked,
        "output_prediction_path": str(output_prediction_path) if output_prediction_path is not None else None,
        "claim_boundary": "CPU prefix diagnostic only. This is a lightweight learned hashed key specialist composed with an existing base stream, not full-corpus G005 completion evidence.",
    }


def write_key_hash_sequence_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_key_hash_sequence_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
