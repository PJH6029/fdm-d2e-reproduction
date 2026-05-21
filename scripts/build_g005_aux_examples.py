#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_d2e.io_utils import sha256_file, stable_hash_json, write_json


DEFAULT_OUTPUT = "artifacts/aux/g005_aux_examples_summary.json"
DEFAULT_ACTION_REGISTRY = "artifacts/aux/g005_aux_action_registry.json"
DEFAULT_NAMESPACE_ROOT = "outputs/aux"
DEFAULT_EXAMPLES_ROOT = "outputs/aux_examples"
DEFAULT_SPLITS = ("train", "val", "test")
SUPPORTED_ADAPTERS = {
    "atari_head_zip_csv_action_adapter",
    "minerl_action_dict_adapter",
    "p_doom_array_record_action_adapter",
}
PDOOM_BREAKOUT_ACTION_MEANINGS = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}


def _path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _action_heads(registry: dict[str, Any], source_ids: list[str] | None) -> list[dict[str, Any]]:
    heads = [row for row in registry.get("action_heads", []) or [] if isinstance(row, dict) and row.get("id")]
    if source_ids:
        wanted = set(source_ids)
        heads = [row for row in heads if str(row.get("id")) in wanted]
        missing = sorted(wanted - {str(row.get("id")) for row in heads})
        if missing:
            raise SystemExit(f"requested source ids are not present in action registry: {', '.join(missing)}")
    return sorted(heads, key=lambda row: str(row.get("id")))


def _split_for_sequence(sequence_id: str, splits: tuple[str, ...]) -> str:
    if set(splits) != {"train", "val", "test"}:
        digest = int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:8], 16)
        return splits[digest % len(splits)]
    bucket = int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _file_report(path: Path, root: Path) -> dict[str, Any]:
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    exists = path.exists() and path.is_file()
    return {
        "path": rel,
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": sha256_file(path) if exists else None,
    }


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _load_atari_action_enums(raw_dir: Path) -> dict[str, str]:
    path = raw_dir / "action_enums.txt"
    if not path.exists() or not path.is_file():
        return {}
    enums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "," in text:
            left, right = text.split(",", 1)
        elif ":" in text:
            left, right = text.split(":", 1)
        else:
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                continue
            left, right = parts
        action_id = str(_maybe_int(left.strip()) if _maybe_int(left.strip()) is not None else left.strip())
        enums[action_id] = right.strip()
    return enums


def _zero_split_counts(splits: tuple[str, ...]) -> dict[str, int]:
    return {split: 0 for split in splits}


def _blocked_source(
    *,
    root: Path,
    source_id: str,
    adapter: str | None,
    raw_dir: Path,
    splits: tuple[str, ...],
    finding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "adapter": adapter,
        "status": "blocked",
        "raw_dir": _rel(root, raw_dir),
        "split_counts": _zero_split_counts(splits),
        "findings": [{**finding, "source_id": source_id}],
    }


def _open_split_handles(out_dir: Path, splits: tuple[str, ...]):
    return {split: _jsonl_writer(out_dir / f"{split}.jsonl") for split in splits}


def _write_example(handles: dict[str, Any], split_counts: dict[str, int], example: dict[str, Any]) -> None:
    handles[example["split"]].write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    split_counts[example["split"]] += 1


def _split_file_report(root: Path, out_dir: Path, split_counts: dict[str, int], splits: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    split_files = {split: _file_report(out_dir / f"{split}.jsonl", root) for split in splits}
    for split, count in split_counts.items():
        split_files[split]["rows"] = count
    return split_files


def _finish_source_row(
    *,
    root: Path,
    source_id: str,
    adapter: str | None,
    raw_dir: Path,
    out_dir: Path,
    split_counts: dict[str, int],
    split_files: dict[str, dict[str, Any]],
    findings: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sum(split_counts.values()) == 0:
        findings.append({"severity": "error", "code": "aux_examples_empty", "source_id": source_id})
    for split, count in split_counts.items():
        if count <= 0:
            findings.append({"severity": "error", "code": "aux_examples_split_empty", "source_id": source_id, "split": split})
    errors = [item for item in findings if item.get("severity") == "error"]
    row = {
        "source_id": source_id,
        "adapter": adapter,
        "status": "pass" if not errors else "blocked",
        "raw_dir": _rel(root, raw_dir),
        "output_namespace": _rel(root, out_dir),
        "split_counts": split_counts,
        "split_files": split_files,
        "max_examples_per_source": extra.pop("max_examples_per_source", None) if extra else None,
        "manifest_fingerprint": stable_hash_json({"source_id": source_id, "split_counts": split_counts, "split_files": split_files}),
        "findings": findings,
    }
    if extra:
        row.update(extra)
    return row


def _matching_tar_member(txt_member: str, members: set[str]) -> str | None:
    candidate = txt_member[:-4] + ".tar.bz2" if txt_member.endswith(".txt") else f"{txt_member}.tar.bz2"
    return candidate if candidate in members else None


def _frame_ref(root: Path, zip_path: Path, tar_member: str | None, sequence_id: str, frame_id: str) -> str:
    zip_rel = str(zip_path.relative_to(root)) if zip_path.is_relative_to(root) else str(zip_path)
    if tar_member:
        return f"nested-archive://{zip_rel}!{tar_member}!{sequence_id}/{frame_id}.png"
    return f"zip-csv://{zip_rel}!{sequence_id}#frame_id={frame_id}"


def _iter_atari_rows(
    *,
    root: Path,
    zip_path: Path,
    source_id: str,
    action_head_namespace: str,
    action_enums: dict[str, str],
    splits: tuple[str, ...],
    max_examples: int | None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        txt_members = sorted(name for name in members if name.endswith(".txt") and not name.endswith("action_enums.txt"))
        for txt_member in txt_members:
            sequence_id = Path(txt_member).with_suffix("").name
            split = _split_for_sequence(sequence_id, splits)
            tar_member = _matching_tar_member(txt_member, members)
            with archive.open(txt_member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""))
                for row_idx, row in enumerate(reader, 1):
                    action_id = _maybe_int(row.get("action"))
                    frame_id = str(row.get("frame_id") or f"{sequence_id}_{row_idx}")
                    action_key = str(action_id) if action_id is not None else str(row.get("action") or "")
                    yield {
                        "source_id": source_id,
                        "source_sequence_id": sequence_id,
                        "frame_or_state_ref": _frame_ref(root, zip_path, tar_member, sequence_id, frame_id),
                        "action": {
                            "type": "atari_discrete",
                            "action_id": action_id,
                            "raw_action": row.get("action"),
                            "action_enum": action_enums.get(action_key),
                        },
                        "action_head_namespace": action_head_namespace,
                        "split": split,
                        "reward": _maybe_int(row.get("unclipped_reward")),
                        "score": _maybe_int(row.get("score")),
                        "duration_ms": _maybe_int(row.get("duration(ms)")),
                        "provenance": {
                            "raw_zip": str(zip_path.relative_to(root)) if zip_path.is_relative_to(root) else str(zip_path),
                            "csv_member": txt_member,
                            "tar_member": tar_member,
                            "frame_id": frame_id,
                            "row_number": row_idx,
                        },
                    }
                    emitted += 1
                    if max_examples is not None and emitted >= max_examples:
                        return


def _build_atari_examples(
    *,
    root: Path,
    head: dict[str, Any],
    namespace_root: Path,
    examples_root: Path,
    splits: tuple[str, ...],
    max_examples: int | None,
    allow_incomplete_raw: bool,
) -> dict[str, Any]:
    source_id = str(head["id"])
    raw_dir = namespace_root / source_id / "raw"
    out_dir = examples_root / source_id
    findings: list[dict[str, Any]] = []
    if not raw_dir.exists() or not raw_dir.is_dir():
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "aux_raw_dir_missing", "path": str(raw_dir)},
        )
    zip_paths = sorted(path for path in raw_dir.glob("*.zip") if path.is_file() and ".part-" not in path.name and ".invalid-" not in path.name)
    if not zip_paths:
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "aux_raw_zip_missing", "path": str(raw_dir)},
        )

    handles = _open_split_handles(out_dir, splits)
    split_counts = _zero_split_counts(splits)
    zip_reports: list[dict[str, Any]] = []
    try:
        action_enums = _load_atari_action_enums(raw_dir)
        if not action_enums:
            findings.append({"severity": "warning", "code": "atari_action_enums_missing", "source_id": source_id})
        for zip_path in zip_paths:
            report = _file_report(zip_path, root)
            try:
                zip_rows = 0
                for example in _iter_atari_rows(
                    root=root,
                    zip_path=zip_path,
                    source_id=source_id,
                    action_head_namespace=str(head.get("namespace") or source_id),
                    action_enums=action_enums,
                    splits=splits,
                    max_examples=None if max_examples is None else max(0, max_examples - sum(split_counts.values())),
                ):
                    _write_example(handles, split_counts, example)
                    zip_rows += 1
                    if max_examples is not None and sum(split_counts.values()) >= max_examples:
                        break
                report["example_rows"] = zip_rows
            except zipfile.BadZipFile as exc:
                severity = "warning" if allow_incomplete_raw else "error"
                findings.append({"severity": severity, "code": "aux_raw_zip_invalid", "source_id": source_id, "path": report["path"], "error": str(exc)})
                report["invalid_zip"] = True
            zip_reports.append(report)
            if max_examples is not None and sum(split_counts.values()) >= max_examples:
                break
    finally:
        for handle in handles.values():
            handle.close()

    split_files = _split_file_report(root, out_dir, split_counts, splits)
    return _finish_source_row(
        root=root,
        source_id=source_id,
        adapter=head.get("adapter"),
        raw_dir=raw_dir,
        out_dir=out_dir,
        split_counts=split_counts,
        split_files=split_files,
        findings=findings,
        extra={"raw_zip_files": zip_reports, "action_enum_count": len(action_enums), "max_examples_per_source": max_examples},
    )


def _iter_minerl_action_payloads(payload: Any, *, member_path: str, sequence_id: str) -> Iterable[tuple[int, Any, dict[str, Any]]]:
    """Yield action payloads from common MineRL JSON trajectory layouts.

    MineRL releases have appeared as trajectory JSON sidecars and API-friendly
    structured dictionaries.  This keeps the adapter source-specific while
    remaining conservative: only explicit action/action-list fields are emitted.
    """

    def walk(value: Any, path: str) -> Iterable[tuple[int, Any, dict[str, Any]]]:
        if isinstance(value, dict):
            if "actions" in value and isinstance(value["actions"], list):
                for idx, action in enumerate(value["actions"]):
                    yield idx, action, {"json_path": f"{path}.actions[{idx}]", "member_path": member_path}
                return
            if "action" in value:
                idx = _maybe_int(value.get("timestep") or value.get("step") or value.get("frame") or value.get("index")) or 0
                yield idx, value["action"], {"json_path": path, "member_path": member_path}
                return
            for key in ("steps", "records", "trajectory", "trajectories", "data", "events", "timesteps"):
                nested = value.get(key)
                if isinstance(nested, (list, dict)):
                    yield from walk(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict) and "action" in item:
                    yield idx, item["action"], {"json_path": f"{path}[{idx}]", "member_path": member_path}
                else:
                    yield from walk(item, f"{path}[{idx}]")

    emitted = False
    for idx, action, provenance in walk(payload, "$"):
        emitted = True
        yield idx, action, provenance
    if not emitted and isinstance(payload, dict) and any(key in payload for key in ("camera", "buttons", "keyboard")):
        yield 0, payload, {"json_path": "$", "member_path": member_path}


def _iter_minerl_zip_rows(
    *,
    root: Path,
    zip_path: Path,
    source_id: str,
    action_head_namespace: str,
    splits: tuple[str, ...],
    max_examples: int | None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    zip_rel = _rel(root, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".json", ".jsonl")) and not name.endswith("/")
        )
        for member in members:
            sequence_id = str(Path(member).with_suffix(""))
            split = _split_for_sequence(f"{zip_path.name}:{sequence_id}", splits)
            with archive.open(member) as raw:
                if member.lower().endswith(".jsonl"):
                    rows = []
                    for line_no, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), 1):
                        text = line.strip()
                        if not text:
                            continue
                        parsed = json.loads(text)
                        rows.append({"line_no": line_no, "action": parsed.get("action") if isinstance(parsed, dict) else parsed})
                    payload: Any = {"actions": [row["action"] for row in rows if row.get("action") is not None]}
                else:
                    payload = json.loads(raw.read().decode("utf-8", errors="replace"))
            for row_idx, action, provenance in _iter_minerl_action_payloads(payload, member_path=member, sequence_id=sequence_id):
                yield {
                    "source_id": source_id,
                    "source_sequence_id": sequence_id,
                    "frame_or_state_ref": f"zip-json://{zip_rel}!{member}#step={row_idx}",
                    "action": {"type": "minecraft_keyboard_mouse", "raw_action": action},
                    "action_head_namespace": action_head_namespace,
                    "split": split,
                    "provenance": {
                        "raw_zip": zip_rel,
                        "json_member": member,
                        "row_number": row_idx,
                        **provenance,
                    },
                }
                emitted += 1
                if max_examples is not None and emitted >= max_examples:
                    return


def _build_minerl_examples(
    *,
    root: Path,
    head: dict[str, Any],
    namespace_root: Path,
    examples_root: Path,
    splits: tuple[str, ...],
    max_examples: int | None,
    allow_incomplete_raw: bool,
) -> dict[str, Any]:
    source_id = str(head["id"])
    raw_dir = namespace_root / source_id / "raw"
    out_dir = examples_root / source_id
    findings: list[dict[str, Any]] = []
    if not raw_dir.exists() or not raw_dir.is_dir():
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "aux_raw_dir_missing", "path": str(raw_dir)},
        )
    zip_paths = sorted(path for path in raw_dir.glob("*.zip") if path.is_file() and ".part-" not in path.name and ".invalid-" not in path.name)
    if not zip_paths:
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "minerl_raw_zip_missing", "path": str(raw_dir)},
        )

    handles = _open_split_handles(out_dir, splits)
    split_counts = _zero_split_counts(splits)
    zip_reports: list[dict[str, Any]] = []
    json_member_count = 0
    try:
        for zip_path in zip_paths:
            report = _file_report(zip_path, root)
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    report["json_member_count"] = len([name for name in archive.namelist() if name.lower().endswith((".json", ".jsonl"))])
                    json_member_count += int(report["json_member_count"])
                rows = 0
                for example in _iter_minerl_zip_rows(
                    root=root,
                    zip_path=zip_path,
                    source_id=source_id,
                    action_head_namespace=str(head.get("namespace") or source_id),
                    splits=splits,
                    max_examples=None if max_examples is None else max(0, max_examples - sum(split_counts.values())),
                ):
                    _write_example(handles, split_counts, example)
                    rows += 1
                    if max_examples is not None and sum(split_counts.values()) >= max_examples:
                        break
                report["example_rows"] = rows
            except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
                severity = "warning" if allow_incomplete_raw else "error"
                findings.append({"severity": severity, "code": "minerl_zip_parse_failed", "source_id": source_id, "path": report["path"], "error": str(exc)})
            zip_reports.append(report)
            if max_examples is not None and sum(split_counts.values()) >= max_examples:
                break
    finally:
        for handle in handles.values():
            handle.close()
    if json_member_count == 0:
        findings.append({"severity": "error", "code": "minerl_action_json_missing", "source_id": source_id})
    split_files = _split_file_report(root, out_dir, split_counts, splits)
    return _finish_source_row(
        root=root,
        source_id=source_id,
        adapter=head.get("adapter"),
        raw_dir=raw_dir,
        out_dir=out_dir,
        split_counts=split_counts,
        split_files=split_files,
        findings=findings,
        extra={"raw_zip_files": zip_reports, "json_member_count": json_member_count, "max_examples_per_source": max_examples},
    )


def _load_array_record_reader():
    try:
        from array_record.python.array_record_module import ArrayRecordReader
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        return None, exc
    return ArrayRecordReader, None


def _flatten_actions(actions: Any) -> list[Any]:
    if actions is None:
        return []
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    if isinstance(actions, tuple):
        actions = list(actions)
    if not isinstance(actions, list):
        return [actions]
    if actions and all(isinstance(item, list) and len(item) == 1 for item in actions):
        return [item[0] for item in actions]
    return actions


def _iter_pdoom_array_record_rows(
    *,
    root: Path,
    array_record_path: Path,
    split: str,
    source_id: str,
    action_head_namespace: str,
    reader_cls: Any,
    max_examples: int | None,
) -> Iterable[dict[str, Any]]:
    emitted = 0
    rel = _rel(root, array_record_path)
    reader = reader_cls(str(array_record_path))
    record_idx = 0
    while True:
        raw = reader.read()
        if raw is None:
            break
        record = pickle.loads(raw)
        seq_len = int(record.get("sequence_length") or 0)
        actions = _flatten_actions(record.get("actions"))
        usable = min(seq_len, len(actions))
        for frame_idx in range(usable):
            action_id = _maybe_int(actions[frame_idx])
            yield {
                "source_id": source_id,
                "source_sequence_id": f"{Path(rel).stem}:record_{record_idx:06d}",
                "frame_or_state_ref": f"array-record://{rel}#record={record_idx}&frame={frame_idx}",
                "action": {
                    "type": "atari_discrete",
                    "action_id": action_id,
                    "raw_action": actions[frame_idx],
                    "action_enum": PDOOM_BREAKOUT_ACTION_MEANINGS.get(action_id) if action_id is not None else None,
                },
                "action_head_namespace": action_head_namespace,
                "split": split,
                "provenance": {
                    "array_record": rel,
                    "record_index": record_idx,
                    "frame_index": frame_idx,
                    "sequence_length": seq_len,
                },
            }
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return
        record_idx += 1


def _split_from_pdoom_path(raw_dir: Path, path: Path, splits: tuple[str, ...]) -> str:
    try:
        first = path.relative_to(raw_dir).parts[0]
    except ValueError:
        first = ""
    return first if first in splits else _split_for_sequence(str(path), splits)


def _build_pdoom_examples(
    *,
    root: Path,
    head: dict[str, Any],
    namespace_root: Path,
    examples_root: Path,
    splits: tuple[str, ...],
    max_examples: int | None,
) -> dict[str, Any]:
    source_id = str(head["id"])
    raw_dir = namespace_root / source_id / "raw"
    out_dir = examples_root / source_id
    findings: list[dict[str, Any]] = []
    if not raw_dir.exists() or not raw_dir.is_dir():
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "aux_raw_dir_missing", "path": str(raw_dir)},
        )
    array_records = sorted(path for path in raw_dir.rglob("*.array_record") if path.is_file() and ".part-" not in path.name and ".invalid-" not in path.name)
    if not array_records:
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={"severity": "error", "code": "pdoom_array_record_files_missing", "path": str(raw_dir)},
        )
    reader_cls, reader_exc = _load_array_record_reader()
    if reader_cls is None:
        return _blocked_source(
            root=root,
            source_id=source_id,
            adapter=head.get("adapter"),
            raw_dir=raw_dir,
            splits=splits,
            finding={
                "severity": "error",
                "code": "array_record_dependency_missing",
                "path": str(raw_dir),
                "error": str(reader_exc),
                "install_hint": "Install the project d2e extra (`uv sync --extra d2e`) or package `array-record` before building p-doom ArrayRecord examples.",
            },
        )

    handles = _open_split_handles(out_dir, splits)
    split_counts = _zero_split_counts(splits)
    file_reports: list[dict[str, Any]] = []
    try:
        for array_record_path in array_records:
            split = _split_from_pdoom_path(raw_dir, array_record_path, splits)
            report = _file_report(array_record_path, root)
            rows = 0
            try:
                for example in _iter_pdoom_array_record_rows(
                    root=root,
                    array_record_path=array_record_path,
                    split=split,
                    source_id=source_id,
                    action_head_namespace=str(head.get("namespace") or source_id),
                    reader_cls=reader_cls,
                    max_examples=None if max_examples is None else max(0, max_examples - sum(split_counts.values())),
                ):
                    _write_example(handles, split_counts, example)
                    rows += 1
                    if max_examples is not None and sum(split_counts.values()) >= max_examples:
                        break
            except (OSError, RuntimeError, pickle.PickleError, ValueError, KeyError) as exc:
                findings.append({"severity": "error", "code": "pdoom_array_record_parse_failed", "source_id": source_id, "path": report["path"], "error": str(exc)})
            report["example_rows"] = rows
            report["split"] = split
            file_reports.append(report)
            if max_examples is not None and sum(split_counts.values()) >= max_examples:
                break
    finally:
        for handle in handles.values():
            handle.close()

    split_files = _split_file_report(root, out_dir, split_counts, splits)
    metadata = _load_json(raw_dir / "metadata.json") if (raw_dir / "metadata.json").exists() else None
    return _finish_source_row(
        root=root,
        source_id=source_id,
        adapter=head.get("adapter"),
        raw_dir=raw_dir,
        out_dir=out_dir,
        split_counts=split_counts,
        split_files=split_files,
        findings=findings,
        extra={
            "array_record_files": file_reports,
            "metadata": {"path": _rel(root, raw_dir / "metadata.json"), "num_actions": (metadata or {}).get("num_actions")} if isinstance(metadata, dict) else None,
            "action_meanings": PDOOM_BREAKOUT_ACTION_MEANINGS,
            "max_examples_per_source": max_examples,
        },
    )


def build_examples(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    registry = _load_json(_path(root, args.action_registry))
    heads = _action_heads(registry, args.source_id)
    namespace_root = _path(root, args.namespace_root)
    examples_root = _path(root, args.examples_root)
    splits = tuple(args.required_splits or DEFAULT_SPLITS)
    findings: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    if registry.get("status") != "pass":
        findings.append({"severity": "error", "code": "action_registry_not_pass", "status": registry.get("status")})
    if not heads:
        findings.append({"severity": "error", "code": "no_action_heads_selected"})

    for head in heads:
        source_id = str(head.get("id"))
        adapter = str(head.get("adapter") or "")
        if adapter == "atari_head_zip_csv_action_adapter":
            row = _build_atari_examples(
                root=root,
                head=head,
                namespace_root=namespace_root,
                examples_root=examples_root,
                splits=splits,
                max_examples=args.max_examples_per_source,
                allow_incomplete_raw=bool(args.allow_incomplete_raw),
            )
            sources.append(row)
            findings.extend({**item, "source_id": item.get("source_id", source_id)} for item in row.get("findings", []))
            continue
        if adapter == "minerl_action_dict_adapter":
            row = _build_minerl_examples(
                root=root,
                head=head,
                namespace_root=namespace_root,
                examples_root=examples_root,
                splits=splits,
                max_examples=args.max_examples_per_source,
                allow_incomplete_raw=bool(args.allow_incomplete_raw),
            )
            sources.append(row)
            findings.extend({**item, "source_id": item.get("source_id", source_id)} for item in row.get("findings", []))
            continue
        if adapter == "p_doom_array_record_action_adapter":
            row = _build_pdoom_examples(
                root=root,
                head=head,
                namespace_root=namespace_root,
                examples_root=examples_root,
                splits=splits,
                max_examples=args.max_examples_per_source,
            )
            sources.append(row)
            findings.extend({**item, "source_id": item.get("source_id", source_id)} for item in row.get("findings", []))
            continue
        findings.append({"severity": "error", "code": "unsupported_aux_example_adapter", "source_id": source_id, "adapter": adapter})
        sources.append(
            {
                "source_id": source_id,
                "adapter": adapter,
                "status": "blocked",
                "split_counts": {split: 0 for split in splits},
                "findings": [{"severity": "error", "code": "unsupported_aux_example_adapter", "adapter": adapter}],
            }
        )

    errors = [item for item in findings if item.get("severity") == "error"]
    payload = {
        "schema": "g005_aux_examples.v1",
        "status": "pass" if not errors else "blocked",
        "root": str(root),
        "action_registry": args.action_registry,
        "namespace_root": str(namespace_root),
        "examples_root": str(examples_root),
        "required_splits": list(splits),
        "selected_source_ids": [str(head.get("id")) for head in heads],
        "supported_adapters": sorted(SUPPORTED_ADAPTERS),
        "sources": sources,
        "total_examples": sum(sum((row.get("split_counts") or {}).values()) for row in sources),
        "findings": findings,
        "error_count": len(errors),
        "claim_boundary": "Source-specific auxiliary example manifests only; they do not train, checkpoint G005, or support D2E+aux quality claims without D2E-only gates, ablation, and final audits.",
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-specific G005 auxiliary example JSONL manifests from materialized raw sources.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--action-registry", default=DEFAULT_ACTION_REGISTRY)
    parser.add_argument("--namespace-root", default=DEFAULT_NAMESPACE_ROOT)
    parser.add_argument("--examples-root", default=DEFAULT_EXAMPLES_ROOT)
    parser.add_argument("--source-id", action="append", help="Restrict to a selected source id; repeatable. Defaults to all registry heads.")
    parser.add_argument("--required-splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-examples-per-source", type=int)
    parser.add_argument("--allow-incomplete-raw", action="store_true", help="Downgrade invalid zip files to warnings while downloads are still in progress.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = build_examples(args)
    write_json(_path(Path(args.root).resolve(), args.output), payload)
    print(f"g005 aux examples: status={payload['status']} examples={payload['total_examples']} output={args.output}")
    return 0 if payload["status"] == "pass" or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
