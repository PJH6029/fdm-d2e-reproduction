from __future__ import annotations

from pathlib import Path
from typing import Any

from fdm_d2e.io_utils import ensure_dir, write_json, write_jsonl
from fdm_d2e.schema import validate_named


KEYS = ['W', 'A', 'S', 'D']


def _pattern_events(pattern: int) -> list[dict[str, Any]]:
    key = KEYS[pattern % len(KEYS)]
    events: list[dict[str, Any]] = [
        {'type': 'keyboard', 'event_type': 'press', 'key': key, 'vk': 87 + pattern},
        {'type': 'mouse_move', 'dx': [2, -2, 6, -6][pattern % 4], 'dy': [0, 3, -3, 8][pattern % 4]},
    ]
    if pattern % 2 == 0:
        events.append({'type': 'mouse_button', 'button': 'left', 'event_type': 'press'})
    else:
        events.append({'type': 'mouse_button', 'button': 'left', 'event_type': 'release'})
    if pattern == 3:
        events.append({'type': 'scroll', 'dy': -1})
    return events


def build_synthetic_records(recording_id: str, num_records: int, fixture_video_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx in range(num_records):
        pattern = idx % 4
        timestamp_ns = idx * 50_000_000  # 50ms bins, matching D2E evaluation convention.
        records.append({
            'schema': 'd2e_record.v1',
            'recording_id': recording_id,
            'sequence_id': f'{recording_id}-{idx:04d}',
            'timestamp_ns': timestamp_ns,
            'frame': {
                'path': fixture_video_path,
                'index': idx,
                'features': [float(pattern), float(idx % 3), float((idx * 7) % 5), 1.0],
            },
            'events': _pattern_events(pattern),
            'source': 'synthetic_d2e_shaped_fixture',
        })
    return records


def write_fixture_pair(output_dir: str | Path, recording_id: str) -> tuple[Path, Path]:
    data_dir = ensure_dir(Path(output_dir) / 'data')
    video_path = data_dir / f'{recording_id}.mkv'
    mcap_path = data_dir / f'{recording_id}.mcap.jsonl'
    video_path.write_text('synthetic D2E-shaped 60fps H.264 video fixture marker; replace with real .mkv for real-data runs\n')
    mcap_path.write_text('')
    return video_path, mcap_path


def prepare_smoke_dataset(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config.get('output_dir', 'outputs'))
    recording_id = str(config.get('recording_id', 'synthetic-d2e-smoke-001'))
    num_records = int(config.get('num_records', 12))
    train_count = int(config.get('train_count', max(1, num_records - 4)))
    video_path, mcap_path = write_fixture_pair(output_dir, recording_id)
    records = build_synthetic_records(recording_id, num_records, str(video_path))
    write_jsonl(mcap_path, [{'timestamp_ns': r['timestamp_ns'], 'events': r['events']} for r in records])
    for idx, rec in enumerate(records):
        rec['split'] = 'train' if idx < train_count else 'heldout'
    train = [r for r in records if r['split'] == 'train']
    heldout = [r for r in records if r['split'] == 'heldout']
    data_dir = ensure_dir(output_dir / 'data')
    manifest = {
        'schema': 'data_manifest.v1',
        'dataset': 'D2E-480p-compatible-smoke-fixture',
        'source_mode': config.get('source_mode', 'synthetic_d2e_shaped_fixture'),
        'hf_repo_id': config.get('hf_repo_id', 'open-world-agents/D2E-480p'),
        'license': 'cc-by-nc-4.0 upstream for D2E; synthetic fixture is generated for smoke tests',
        'source_contract': {
            'paired_video_mcap': True,
            'video_path': str(video_path),
            'mcap_path': str(mcap_path),
            'timestamp_unit': 'nanoseconds',
            'bin_ms': 50,
            'event_topics': ['keyboard', 'mouse/raw', 'mouse/buttons', 'scroll'],
            'official_eval_reference': 'D2E evaluate.py 50ms-bin style metrics',
            'real_sample_download_default': bool(config.get('download_real_sample_by_default', False)),
        },
        'recordings': [{'recording_id': recording_id, 'video': str(video_path), 'mcap': str(mcap_path), 'num_records': num_records}],
        'splits': {'train': len(train), 'heldout': len(heldout)},
        'event_categories': ['keyboard', 'mouse_move', 'mouse_button', 'scroll'],
    }
    validate_named(manifest, 'data_manifest.schema.json')
    write_json(data_dir / 'manifest.json', manifest)
    write_jsonl(data_dir / 'all_records.jsonl', records)
    write_jsonl(data_dir / 'train.jsonl', train)
    write_jsonl(data_dir / 'heldout.jsonl', heldout)
    write_json(data_dir / 'source_contract_smoke.json', manifest['source_contract'])
    return {'manifest': manifest, 'records': records, 'train': train, 'heldout': heldout}
