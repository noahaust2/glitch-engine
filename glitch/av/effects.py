"""Per-frame visual glitch effects."""

from __future__ import annotations

import numpy as np


def rgb_split(
    frame: np.ndarray,
    offset_px: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift R, G, B channels by random pixel offsets."""
    if rng is None:
        rng = np.random.default_rng()
    if offset_px is None:
        offset_px = rng.integers(2, 20)

    result = frame.copy()
    h, w = frame.shape[:2]

    for ch in range(3):
        dx = rng.integers(-offset_px, offset_px + 1)
        dy = rng.integers(-offset_px, offset_px + 1)
        result[:, :, ch] = np.roll(np.roll(frame[:, :, ch], dx, axis=1), dy, axis=0)

    return result


def scan_lines(
    frame: np.ndarray,
    intensity: float = 0.5,
    line_width: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Overlay horizontal scan lines (darken every Nth row)."""
    result = frame.copy().astype(np.float32)
    h = frame.shape[0]

    for y in range(0, h, line_width * 2):
        end = min(y + line_width, h)
        result[y:end] *= (1.0 - intensity)

    return np.clip(result, 0, 255).astype(np.uint8)


def corrupt(
    frame: np.ndarray,
    amount: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero out random rectangular blocks."""
    if rng is None:
        rng = np.random.default_rng()

    result = frame.copy()
    h, w = frame.shape[:2]
    n_blocks = max(1, int(amount * 20))

    for _ in range(n_blocks):
        bw = rng.integers(10, max(11, w // 4))
        bh = rng.integers(5, max(6, h // 8))
        x = rng.integers(0, max(1, w - bw))
        y = rng.integers(0, max(1, h - bh))
        result[y:y + bh, x:x + bw] = 0

    return result


def invert_region(
    frame: np.ndarray,
    amount: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Color-invert random rectangular regions."""
    if rng is None:
        rng = np.random.default_rng()

    result = frame.copy()
    h, w = frame.shape[:2]
    n_regions = max(1, int(amount * 10))

    for _ in range(n_regions):
        rw = rng.integers(20, max(21, w // 3))
        rh = rng.integers(10, max(11, h // 4))
        x = rng.integers(0, max(1, w - rw))
        y = rng.integers(0, max(1, h - rh))
        result[y:y + rh, x:x + rw] = 255 - result[y:y + rh, x:x + rw]

    return result


def posterize(
    frame: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """Reduce color depth to N levels per channel."""
    levels = max(2, levels)
    step = 256 // levels
    return (frame // step * step).astype(np.uint8)


def noise(
    frame: np.ndarray,
    amount: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add random noise to frame."""
    if rng is None:
        rng = np.random.default_rng()

    noise_arr = rng.integers(
        -int(amount * 128), int(amount * 128) + 1,
        size=frame.shape, dtype=np.int16,
    )
    result = frame.astype(np.int16) + noise_arr
    return np.clip(result, 0, 255).astype(np.uint8)
