"""Stem mixing utilities — layer and mixdown."""

import numpy as np
from glitch.core import normalize as peak_normalize, ms_to_samples


def layer(
    stems: list[np.ndarray],
    offsets: list[float],
    gains: list[float],
    sr: int,
) -> np.ndarray:
    """Place stems at given time offsets with given gains, sum them.

    Args:
        stems: List of audio buffers.
        offsets: Time offset in seconds for each stem.
        gains: Linear gain (amplitude multiplier) for each stem.
        sr: Sample rate.

    Returns:
        Mixed audio buffer.
    """
    if not stems:
        return np.zeros(1, dtype=np.float64)

    # Determine channel count
    n_channels = max(
        (s.shape[1] if s.ndim == 2 else 1) for s in stems
    )

    # Calculate total length
    total_length = 0
    for stem, offset in zip(stems, offsets):
        stem_len = len(stem) if stem.ndim == 1 else stem.shape[0]
        end = int(offset * sr) + stem_len
        total_length = max(total_length, end)

    if n_channels == 1:
        output = np.zeros(total_length, dtype=np.float64)
    else:
        output = np.zeros((total_length, n_channels), dtype=np.float64)

    for stem, offset, gain in zip(stems, offsets, gains):
        pos = int(offset * sr)
        stem_len = len(stem) if stem.ndim == 1 else stem.shape[0]

        if pos < 0:
            # Trim start of stem
            trim = -pos
            stem = stem[trim:]
            stem_len -= trim
            pos = 0

        end = min(pos + stem_len, total_length)
        actual_len = end - pos
        seg = stem[:actual_len].copy() * gain

        # Match channels
        if seg.ndim == 1 and n_channels > 1:
            seg = np.column_stack([seg] * n_channels)
        elif seg.ndim == 2 and n_channels == 1:
            seg = seg[:, 0]

        output[pos:end] += seg

    return output


def mixdown(
    stems: list[np.ndarray],
    sr: int,
    do_normalize: bool = True,
) -> np.ndarray:
    """Align and sum all stems, optionally normalize.

    All stems start at time 0. For offset placement, use layer() instead.

    Args:
        stems: List of audio buffers.
        sr: Sample rate.
        do_normalize: Peak-normalize the result.

    Returns:
        Mixed audio buffer.
    """
    if not stems:
        return np.zeros(1, dtype=np.float64)

    n_channels = max(
        (s.shape[1] if s.ndim == 2 else 1) for s in stems
    )

    max_len = max(
        len(s) if s.ndim == 1 else s.shape[0] for s in stems
    )

    if n_channels == 1:
        output = np.zeros(max_len, dtype=np.float64)
    else:
        output = np.zeros((max_len, n_channels), dtype=np.float64)

    for stem in stems:
        stem_len = len(stem) if stem.ndim == 1 else stem.shape[0]
        seg = stem.copy()

        if seg.ndim == 1 and n_channels > 1:
            seg = np.column_stack([seg] * n_channels)
        elif seg.ndim == 2 and n_channels == 1:
            seg = seg[:, 0]

        output[:stem_len] += seg

    if do_normalize:
        output = peak_normalize(output)

    return output
