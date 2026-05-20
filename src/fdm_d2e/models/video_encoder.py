from __future__ import annotations

from typing import Iterable


class StatisticalVideoEncoder:
    """Tiny replaceable video encoder baseline for smoke tests.

    It preserves the encoder interface without claiming to reproduce FDM-1's
    private video compression model.
    """

    def __init__(self, feature_dim: int = 4):
        self.feature_dim = feature_dim

    def encode(self, frame_features: Iterable[float]) -> list[float]:
        values = list(float(v) for v in frame_features)
        if not values:
            values = [0.0]
        mean = sum(values) / len(values)
        max_v = max(values)
        min_v = min(values)
        energy = sum(v * v for v in values) / len(values)
        encoded = [mean, max_v, min_v, energy]
        if self.feature_dim <= len(encoded):
            return encoded[: self.feature_dim]
        return encoded + [0.0] * (self.feature_dim - len(encoded))
