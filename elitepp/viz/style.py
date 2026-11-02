"""Shared colors (uint8 RGB) so Rerun recordings and the Viser viewer look the same."""
import numpy as np

GRAY = (150, 150, 150)
ORANGE = (230, 120, 0)
BLUE = (0, 120, 230)

# Per DOR reason code (elitepp.dor.REASONS): static, low ratio, cluster, floating, column, smoothing, ankle
REASON_COLORS = np.array([
    (170, 170, 170),
    (230, 25, 75),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (255, 225, 25),
], dtype=np.uint8)


_RATIO_STOPS = np.array([0.0, 0.5, 1.0])
_RATIO_RGB = np.array([(214, 39, 40), (240, 170, 40), (31, 119, 180)], dtype=np.float32)


def ratio_colors(ratio):
    """Stability ratio in [0, 1] -> red (unstable) .. amber .. blue (stable)."""
    r = np.clip(np.asarray(ratio, dtype=np.float32), 0, 1)
    return np.stack([np.interp(r, _RATIO_STOPS, _RATIO_RGB[:, c]) for c in range(3)], axis=1).astype(np.uint8)
