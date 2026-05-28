from __future__ import annotations

import glob
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from fdm_d2e.eval.paper_idm_metrics import _PaperMetricAccumulator
from fdm_d2e.io_utils import write_json

try:  # pragma: no cover
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

_BUTTONS = ("LEFT", "RIGHT", "MIDDLE")
_DEFAULT_THRESHOLDS = (0.35, 0.5, 0.65, 0.8, 0.9, 0.95)
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
            return [str(item) for item in value]
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


def _button_holds(row: dict[str, Any]) -> dict[str, int]:
    value = row.get("prior_button_hold_bins")
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        code = str(key).upper()
        if code not in _BUTTONS:
            continue
        try:
            out[code] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _since(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("prior_since_button_transition_bins") or 0))
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


def _button_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(t) for t in tokens if str(t).startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))]


def _non_button_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(t) for t in tokens if not str(t).startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_"))]


def _features(row: dict[str, Any], code: str, hold: int) -> list[str]:
    idx = sequence_bin_index(row)
    since = _since(row)
    hbin = _bucket(hold)
    sbin = _bucket(since)
    prev = _tokens(row, "previous_event_tokens")
    prior = _tokens(row, "prior_action_tokens")
    held_codes = sorted(_button_holds(row))
    held_now = int(code in held_codes)
    feats = [
        "bias",
        f"button={code}",
        f"game={_recording_game(row)}",
        f"held={held_now}",
        f"button={code}|held={held_now}",
        f"button={code}|hbin={hbin}",
        f"button={code}|sbin={sbin}",
        f"button={code}|hbin={hbin}|sbin={sbin}",
        f"button_held_count={len(held_codes)}",
    ]
    for mod in _MODS:
        feats.append(f"phase{mod}={idx % mod}")
        feats.append(f"button={code}|phase{mod}={idx % mod}")
        feats.append(f"button={code}|hmod{mod}={hold % mod}")
        feats.append(f"button={code}|smod{mod}={since % mod}")
    for token in prev[:32]:
        if token.startswith(("MOUSE_DX_", "MOUSE_DY_")):
            feats.append(f"prev_move={token}")
            feats.append(f"button={code}|prev_move={token}")
        elif token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            feats.append(f"prev_button={token}")
            feats.append(f"button={code}|prev_button={token}")
        elif token.startswith("KEY_"):
            feats.append(f"prev_key={token}")
    for token in prior[:32]:
        if token.startswith(("MOUSE_LEFT_", "MOUSE_RIGHT_", "MOUSE_MIDDLE_")):
            feats.append(f"prior_button_token={token}")
            feats.append(f"button={code}|prior_button_token={token}")
        elif token.startswith("KEY_DOWN_"):
            feats.append(f"prior_key_down={token}")
    for other in held_codes:
        if other != code:
            feats.append(f"held_with={other}")
            feats.append(f"button={code}|held_with={other}")
    return feats


class HashedLogisticButtonModel:
    def __init__(self, *, dim: int = 1 << 18, learning_rate: float = 0.05, down_weight: float = 8.0, up_weight: float = 8.0) -> None:
        self.dim = int(dim)
        self.learning_rate = float(learning_rate)
        self.down_weight = float(down_weight)
        self.up_weight = float(up_weight)
        self.down_weights = [0.0] * self.dim
        self.up_weights = [0.0] * self.dim
        self.examples = 0
        self.positive_down = 0
        self.positive_up = 0

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
        for idx in indices:
            weights[idx] += self.learning_rate * grad

    def observe(self, row: dict[str, Any]) -> None:
        gt = set(_tokens(row, "ground_truth_tokens"))
        holds = _button_holds(row)
        for code in _BUTTONS:
            indices = self._indices(_features(row, code, holds.get(code, 0)))
            down = int(f"MOUSE_{code}_DOWN" in gt)
            up = int(f"MOUSE_{code}_UP" in gt)
            self.positive_down += down
            self.positive_up += up
            self._update(self.down_weights, indices, down, self.down_weight if down else 1.0)
            self._update(self.up_weights, indices, up, self.up_weight if up else 1.0)
            self.examples += 1

    def predict(self, row: dict[str, Any], *, down_threshold: float, up_threshold: float) -> list[str]:
        out: list[str] = []
        holds = _button_holds(row)
        for code in _BUTTONS:
            indices = self._indices(_features(row, code, holds.get(code, 0)))
            if self._sigmoid(self._score(self.down_weights, indices)) >= float(down_threshold):
                out.append(f"MOUSE_{code}_DOWN")
            if self._sigmoid(self._score(self.up_weights, indices)) >= float(up_threshold):
                out.append(f"MOUSE_{code}_UP")
        return out


def train_hashed_button_model(
    *,
    train_paths: Sequence[str | Path] | str | Path,
    max_train_rows: int | None = 320_000,
    epochs: int = 1,
    dim: int = 1 << 18,
    learning_rate: float = 0.05,
    down_weight: float = 8.0,
    up_weight: float = 8.0,
) -> tuple[HashedLogisticButtonModel, int]:
    model = HashedLogisticButtonModel(dim=dim, learning_rate=learning_rate, down_weight=down_weight, up_weight=up_weight)
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


def build_button_hash_sequence_diagnostic(
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
    down_weight: float = 8.0,
    up_weight: float = 8.0,
    down_thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    up_thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
    split_tags: Sequence[str] = ("temporal", "heldout_recording", "heldout_game"),
) -> dict[str, Any]:
    model, train_rows = train_hashed_button_model(
        train_paths=train_paths,
        max_train_rows=max_train_rows,
        epochs=epochs,
        dim=dim,
        learning_rate=learning_rate,
        down_weight=down_weight,
        up_weight=up_weight,
    )
    specs: list[tuple[str, str, float | None, float | None]] = [("base_all", "base", None, None)]
    for down_th in down_thresholds:
        for up_th in up_thresholds:
            specs.append((f"replace_base_buttons_down{down_th:g}_up{up_th:g}", "replace", float(down_th), float(up_th)))
            specs.append((f"union_base_buttons_down{down_th:g}_up{up_th:g}", "union", float(down_th), float(up_th)))
    accs = {name: _new_accs(split_tags) for name, _mode, _d, _u in specs}
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
            non_button = _non_button_tokens(base_tokens)
            tags = set(_split_tags(row))
            gt = _tokens(row, "ground_truth_tokens")
            export_tokens: list[str] | None = None
            for name, mode, down_th, up_th in specs:
                if mode == "base":
                    pred_tokens = base_tokens
                else:
                    specialist = model.predict(row, down_threshold=float(down_th), up_threshold=float(up_th))
                    if mode == "union":
                        buttons = []
                        seen: set[str] = set()
                        for token in _button_tokens(base_tokens) + specialist:
                            if token not in seen:
                                buttons.append(token)
                                seen.add(token)
                    else:
                        buttons = specialist
                    pred_tokens = non_button + buttons
                accs[name]["all"].update(pred_tokens, gt)
                for tag in split_tags:
                    if tag in tags:
                        accs[name][f"eval_split:{tag}"].update(pred_tokens, gt)
                if export_tokens is None and name != "base_all":
                    export_tokens = pred_tokens
            if pred_handle is not None:
                pred_handle.write(json.dumps({"sequence_id": row.get("sequence_id"), "predicted_tokens": export_tokens or base_tokens}, separators=(",", ":")) + "\n")
    finally:
        if pred_handle is not None:
            pred_handle.close()
    policies: dict[str, Any] = {}
    for name, group_accs in accs.items():
        groups = _metrics(group_accs)
        policies[name] = {"groups": groups, "summary": _summary(groups["all"])}
    ranked = sorted(
        ({"policy": name, **payload["summary"]} for name, payload in policies.items()),
        key=lambda item: (
            item.get("mouse_button_accuracy") if item.get("mouse_button_accuracy") is not None else -1.0,
            -(item.get("strict_no_button_fpr") if item.get("strict_no_button_fpr") is not None else 1.0),
            item.get("strict_mouse_button_f1") if item.get("strict_mouse_button_f1") is not None else -1.0,
        ),
        reverse=True,
    )
    return {
        "schema": "g005_button_hash_sequence_diagnostic.v1",
        "status": "pass",
        "rows": rows,
        "train_rows": train_rows,
        "max_train_rows": max_train_rows,
        "max_target_rows": max_target_rows,
        "epochs": int(epochs),
        "dim": int(dim),
        "learning_rate": float(learning_rate),
        "down_weight": float(down_weight),
        "up_weight": float(up_weight),
        "model_examples": model.examples,
        "model_positive_down": model.positive_down,
        "model_positive_up": model.positive_up,
        "alignment": alignment,
        "policies": policies,
        "ranked_policies": ranked,
        "output_prediction_path": str(output_prediction_path) if output_prediction_path is not None else None,
        "claim_boundary": "CPU prefix diagnostic only. This is a lightweight learned hashed mouse-button specialist composed with an existing base stream, not full-corpus G005 completion evidence.",
    }


def write_button_hash_sequence_diagnostic(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_button_hash_sequence_diagnostic(**kwargs)
    write_json(output_path, payload)
    return payload
