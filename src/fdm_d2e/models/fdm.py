from __future__ import annotations

from collections import Counter
from typing import Any


class FrequencyFDM:
    """Tiny next-action predictor that consumes IDM pseudo-labels for smoke tests."""

    def __init__(self) -> None:
        self.global_tokens: list[str] = ['NOOP']

    def fit(self, pseudo_labels: list[dict[str, Any]]) -> 'FrequencyFDM':
        counts: Counter[tuple[str, ...]] = Counter()
        for row in pseudo_labels:
            counts[tuple(row.get('predicted_tokens', [])) or ('NOOP',)] += 1
        if counts:
            self.global_tokens = list(counts.most_common(1)[0][0])
        return self

    def predict(self, pseudo_label: dict[str, Any]) -> list[str]:
        # Canonical smoke behavior: pass through the IDM-produced token sequence.
        # This proves dataflow and artifact contracts before optimizing quality.
        return list(pseudo_label.get('predicted_tokens') or self.global_tokens)
