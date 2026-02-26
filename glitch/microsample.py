"""Micro-sampler — the primary instrument.

Every operation is a discrete, non-overlapping slice operation. No overlapping
grains, no crossfaded envelopes. Every slice is a clean cut, designed to map
1:1 to visual cuts in Phase 2.
"""

from __future__ import annotations

import numpy as np
import librosa

from glitch.core import ms_to_samples
from glitch.cutlist import Cut, CutList


def _find_zero_crossing(audio: np.ndarray, pos: int, window: int = 64) -> int:
    """Find nearest zero crossing within a window around pos."""
    mono = audio if audio.ndim == 1 else audio[:, 0]
    length = len(mono)
    if pos <= 0 or pos >= length:
        return pos

    start = max(0, pos - window)
    end = min(length - 1, pos + window)
    segment = mono[start:end]

    if len(segment) < 2:
        return pos

    # Find zero crossings
    signs = np.sign(segment)
    crossings = np.where(np.diff(signs) != 0)[0]

    if len(crossings) == 0:
        return pos

    # Find the crossing nearest to the original position
    crossing_positions = crossings + start
    distances = np.abs(crossing_positions - pos)
    return int(crossing_positions[np.argmin(distances)])


def _compute_boundaries(
    audio: np.ndarray,
    sr: int,
    slice_ms: float | None,
    slice_min_ms: float | None,
    slice_max_ms: float | None,
    mode: str,
    rng: np.random.Generator,
) -> list[int]:
    """Compute slice boundaries based on mode."""
    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    if audio_len == 0:
        return [0]

    if mode == "transients":
        mono = audio if audio.ndim == 1 else librosa.to_mono(audio.T)
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="samples")
        boundaries = [0] + sorted(onsets.tolist()) + [audio_len]
        boundaries = sorted(set(boundaries))
        # Snap to zero crossings
        boundaries = [_find_zero_crossing(audio, b) for b in boundaries]
        boundaries[0] = 0
        boundaries[-1] = audio_len
        return sorted(set(boundaries))

    if mode == "random":
        if slice_min_ms is None:
            slice_min_ms = 20.0
        if slice_max_ms is None:
            slice_max_ms = 100.0
        min_samp = max(1, int(slice_min_ms * sr / 1000))
        max_samp = max(min_samp + 1, int(slice_max_ms * sr / 1000))
        boundaries = [0]
        pos = 0
        while pos < audio_len:
            step = rng.integers(min_samp, max_samp)
            pos += step
            b = min(pos, audio_len)
            b = _find_zero_crossing(audio, b)
            boundaries.append(min(b, audio_len))
        if boundaries[-1] != audio_len:
            boundaries.append(audio_len)
        return sorted(set(boundaries))

    # Default: fixed
    if slice_ms is None:
        slice_ms = 50.0
    step = max(1, int(slice_ms * sr / 1000))
    boundaries = []
    pos = 0
    while pos < audio_len:
        boundaries.append(pos)
        pos += step
    if boundaries[-1] != audio_len:
        boundaries.append(audio_len)
    # Snap to zero crossings (except first and last)
    for i in range(1, len(boundaries) - 1):
        boundaries[i] = _find_zero_crossing(audio, boundaries[i])
    return sorted(set(boundaries))


def microsample(
    audio: np.ndarray,
    sr: int,
    slice_ms: float | None = None,
    slice_min_ms: float | None = None,
    slice_max_ms: float | None = None,
    mode: str = "fixed",
    shuffle_chance: float = 0.3,
    stutter_chance: float = 0.2,
    max_repeats: int = 4,
    reverse_chance: float = 0.1,
    drop_chance: float = 0.1,
    preserve_length: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, CutList]:
    """Micro-sample audio: slice, shuffle, stutter, reverse, drop.

    The primary instrument. Every operation is a discrete, non-overlapping
    slice operation logged to a CutList.

    Args:
        audio: Input audio buffer (float64, [-1.0, 1.0]).
        sr: Sample rate.
        slice_ms: Fixed slice length in ms (default: 50).
        slice_min_ms: Min random slice length (use with mode='random').
        slice_max_ms: Max random slice length (use with mode='random').
        mode: 'fixed', 'transients', or 'random'.
        shuffle_chance: Probability of displacing a slice (0.0-1.0).
        stutter_chance: Probability of repeating a slice (0.0-1.0).
        max_repeats: Maximum stutter repetitions (1-16).
        reverse_chance: Probability of reversing a slice (0.0-1.0).
        drop_chance: Probability of dropping a slice to silence (0.0-1.0).
        preserve_length: If True, output matches input duration.
        seed: Random seed for reproducibility.

    Returns:
        (output audio array, CutList)
    """
    rng = np.random.default_rng(seed)
    actual_seed = seed if seed is not None else int(rng.integers(0, 2**31))

    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    duration_s = audio_len / sr

    # Compute slice boundaries
    boundaries = _compute_boundaries(
        audio, sr, slice_ms, slice_min_ms, slice_max_ms, mode, rng
    )

    # Build slices as (start_sample, end_sample) pairs
    slices = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    slices = [(s, e) for s, e in slices if e > s]

    # Shuffle: displace slices probabilistically
    order = list(range(len(slices)))
    for i in range(len(order)):
        if rng.random() < shuffle_chance:
            j = rng.integers(0, len(order))
            order[i], order[j] = order[j], order[i]

    # Process slices
    parts = []
    cuts = []
    current_dest_s = 0.0

    for idx in order:
        start_samp, end_samp = slices[idx]
        source_start_s = start_samp / sr
        source_end_s = end_samp / sr

        a_slice = audio[start_samp:end_samp].copy()

        # Drop?
        if rng.random() < drop_chance:
            silence = np.zeros_like(a_slice)
            cuts.append(Cut(
                source_start_s=source_start_s,
                source_end_s=source_end_s,
                dest_start_s=current_dest_s,
                repeats=1,
                reversed=False,
                dropped=True,
            ))
            parts.append(silence)
            current_dest_s += len(silence) / sr
            continue

        # Reverse?
        is_reversed = rng.random() < reverse_chance

        # Stutter?
        repeats = 1
        if rng.random() < stutter_chance:
            repeats = rng.integers(2, max_repeats + 1)

        cuts.append(Cut(
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            dest_start_s=current_dest_s,
            repeats=repeats,
            reversed=is_reversed,
            dropped=False,
        ))

        for _ in range(repeats):
            part = a_slice.copy()
            if is_reversed:
                part = part[::-1]
            parts.append(part)
            current_dest_s += len(part) / sr

    # Assemble
    if parts:
        result = np.concatenate(parts)
    else:
        result = np.zeros(1, dtype=np.float64) if audio.ndim == 1 else np.zeros((1, audio.shape[1]), dtype=np.float64)

    # Preserve length
    if preserve_length:
        if len(result) > audio_len:
            result = result[:audio_len]
        elif len(result) < audio_len:
            if result.ndim == 1:
                result = np.pad(result, (0, audio_len - len(result)))
            else:
                result = np.pad(result, ((0, audio_len - len(result)), (0, 0)))

    out_duration = len(result) / sr if result.ndim == 1 else result.shape[0] / sr
    cut_list = CutList(
        cuts=cuts,
        seed=actual_seed,
        source_duration_s=duration_s,
        output_duration_s=out_duration,
    )

    return result, cut_list


def chop(
    audio: np.ndarray,
    sr: int,
    slices: int = 16,
    reverse_chance: float = 0.2,
    drop_chance: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, CutList]:
    """Chop audio into N slices, randomly reorder them.

    Like shuffling a deck of cards on the timeline. Convenience wrapper
    around microsample() with mode='fixed' and shuffle_chance=1.0.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        slices: Number of slices.
        reverse_chance: Probability of reversing each slice.
        drop_chance: Probability of dropping each slice.
        seed: Random seed for reproducibility.

    Returns:
        (output audio array, CutList)
    """
    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    slice_ms = (audio_len / sr * 1000) / slices

    return microsample(
        audio, sr,
        slice_ms=slice_ms,
        mode="fixed",
        shuffle_chance=1.0,
        stutter_chance=0.0,
        reverse_chance=reverse_chance,
        drop_chance=drop_chance,
        preserve_length=True,
        seed=seed,
    )


def chain(
    audio: np.ndarray,
    sr: int,
    operations: list[dict],
) -> tuple[np.ndarray, list[CutList]]:
    """Apply multiple micro-sampling passes in sequence.

    Each operation is a dict of parameters for microsample().
    Returns the final audio and all cut lists.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        operations: List of parameter dicts.

    Returns:
        (final audio, list of CutLists from each pass)
    """
    current = audio
    all_cuts = []

    for params in operations:
        current, cut_list = microsample(current, sr, **params)
        all_cuts.append(cut_list)

    return current, all_cuts
