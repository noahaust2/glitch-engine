"""Stutter and glitch effects — segment-based destruction."""

import numpy as np
import librosa
from glitch.core import ms_to_samples


def _bitcrush(audio: np.ndarray, bits: int) -> np.ndarray:
    """Reduce bit depth of audio signal."""
    bits = max(2, min(16, bits))
    levels = 2 ** bits
    return np.round(audio * levels) / levels


def _downsample(audio: np.ndarray, factor: int) -> np.ndarray:
    """Crude sample-rate reduction by repeating samples."""
    factor = max(1, factor)
    if audio.ndim == 1:
        indices = np.arange(len(audio))
        return audio[(indices // factor) * factor]
    indices = np.arange(audio.shape[0])
    return audio[(indices // factor) * factor]


def glitch(
    audio: np.ndarray,
    sr: int,
    stutter_chance: float = 0.4,
    max_repeats: int = 4,
    reverse_chance: float = 0.2,
    crush_chance: float = 0.1,
    crush_bits: int = 8,
    preserve_length: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Create stutter/glitch effects by detecting transients and manipulating segments.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        stutter_chance: Probability of stuttering a segment (0.0-1.0).
        max_repeats: Maximum repetitions of a stuttered segment.
        reverse_chance: Probability of reversing a repeated segment.
        crush_chance: Probability of bit-crushing a segment.
        crush_bits: Bit depth for crushing (2-16).
        preserve_length: If True, output matches input length.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Work with mono for onset detection
    mono = audio if audio.ndim == 1 else librosa.to_mono(audio.T)

    # Detect onsets
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="samples")

    # Build segment boundaries
    boundaries = [0] + list(onsets) + [len(mono)]
    boundaries = sorted(set(boundaries))

    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        seg = audio[start:end].copy()

        # Decide what to do with this segment
        if rng.random() < stutter_chance:
            repeats = rng.integers(2, max_repeats + 1)
            parts = []
            for _ in range(repeats):
                part = seg.copy()
                if rng.random() < reverse_chance:
                    part = part[::-1]
                if rng.random() < crush_chance:
                    part = _bitcrush(part, crush_bits)
                parts.append(part)
            segments.extend(parts)
        else:
            if rng.random() < crush_chance:
                seg = _bitcrush(seg, crush_bits)
            segments.append(seg)

    if not segments:
        return audio.copy()

    result = np.concatenate(segments)

    if preserve_length:
        target = len(audio) if audio.ndim == 1 else audio.shape[0]
        if len(result) > target:
            result = result[:target]
        elif len(result) < target:
            if result.ndim == 1:
                result = np.pad(result, (0, target - len(result)))
            else:
                result = np.pad(result, ((0, target - len(result)), (0, 0)))

    return result


def chop(
    audio: np.ndarray,
    sr: int,
    slices: int = 16,
    reverse_chance: float = 0.2,
    drop_chance: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Chop audio into N slices, randomly reorder them.

    Like shuffling a deck of cards on the timeline.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        slices: Number of slices.
        reverse_chance: Probability of reversing a slice.
        drop_chance: Probability of dropping a slice (silence).
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    length = len(audio) if audio.ndim == 1 else audio.shape[0]
    slice_len = length // slices

    if slice_len < 1:
        return audio.copy()

    # Cut into slices
    pieces = []
    for i in range(slices):
        start = i * slice_len
        end = start + slice_len if i < slices - 1 else length
        pieces.append(audio[start:end].copy())

    # Shuffle order
    rng.shuffle(pieces)

    # Process each slice
    processed = []
    for piece in pieces:
        if rng.random() < drop_chance:
            # Replace with silence
            processed.append(np.zeros_like(piece))
        elif rng.random() < reverse_chance:
            processed.append(piece[::-1])
        else:
            processed.append(piece)

    return np.concatenate(processed)
