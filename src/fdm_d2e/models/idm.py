from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def feature_signature(record: dict[str, Any]) -> str:
    features = record.get('frame', {}).get('features') or record.get('frame_features') or []
    if not features:
        return 'empty'
    # The first feature in the synthetic fixture encodes recurring visual state;
    # real D2E integrations can replace this with encoder embeddings.
    return '|'.join(str(round(float(v), 3)) for v in features[:2])


class NearestFeatureIDM:
    def __init__(self) -> None:
        self.by_signature: dict[str, list[str]] = {}
        self.majority: list[str] = ['NOOP']

    def fit(self, records: list[dict[str, Any]]) -> 'NearestFeatureIDM':
        counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
        global_counts: Counter[tuple[str, ...]] = Counter()
        for row in records:
            tokens = tuple(row.get('ground_truth_tokens', [])) or ('NOOP',)
            counts[feature_signature(row)][tokens] += 1
            global_counts[tokens] += 1
        self.by_signature = {sig: list(counter.most_common(1)[0][0]) for sig, counter in counts.items()}
        if global_counts:
            self.majority = list(global_counts.most_common(1)[0][0])
        return self

    def predict(self, record: dict[str, Any]) -> tuple[list[str], float]:
        sig = feature_signature(record)
        if sig in self.by_signature:
            return list(self.by_signature[sig]), 0.95
        return list(self.majority), 0.51
