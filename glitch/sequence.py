"""Probabilistic sequencer and arranger for layering processed stems."""

import numpy as np
from glitch.core import ms_to_samples, fade


def arrange(
    stems: list[np.ndarray],
    sr: int,
    duration_s: float = 30.0,
    density: float = 0.5,
    cluster_factor: float = 0.3,
    max_layers: int = 3,
    fade_ms: float = 10.0,
    seed: int | None = None,
) -> np.ndarray:
    """Arrange stems on a timeline with probabilistic placement.

    Silence gaps are intentional — this produces sparse, textural arrangements.

    Args:
        stems: List of audio buffers (all same sample rate).
        sr: Sample rate.
        duration_s: Total output length in seconds.
        density: 0.0 (very sparse) to 1.0 (packed).
        cluster_factor: Tendency for events to group together.
        max_layers: Maximum simultaneous overlapping stems.
        fade_ms: Crossfade length for events.
        seed: Random seed for reproducibility.
    """
    if not stems:
        return np.zeros(int(duration_s * sr), dtype=np.float64)

    rng = np.random.default_rng(seed)
    out_length = int(duration_s * sr)

    # Determine channel count from stems
    n_channels = max(
        (s.shape[1] if s.ndim == 2 else 1) for s in stems
    )
    if n_channels == 1:
        output = np.zeros(out_length, dtype=np.float64)
        layer_count = np.zeros(out_length, dtype=np.int32)
    else:
        output = np.zeros((out_length, n_channels), dtype=np.float64)
        layer_count = np.zeros(out_length, dtype=np.int32)

    # Calculate how many events to place
    avg_stem_dur = np.mean([len(s) if s.ndim == 1 else s.shape[0] for s in stems])
    total_fill = density * out_length
    n_events = max(1, int(total_fill / avg_stem_dur))

    # Generate cluster centers
    n_clusters = max(1, int(n_events * (1.0 - cluster_factor)))
    centers = rng.uniform(0, out_length, n_clusters)

    for _ in range(n_events):
        stem = stems[rng.integers(0, len(stems))]
        stem_len = len(stem) if stem.ndim == 1 else stem.shape[0]

        # Pick position: near a cluster center or random
        if rng.random() < cluster_factor and len(centers) > 0:
            center = centers[rng.integers(0, len(centers))]
            pos = int(center + rng.normal(0, out_length * 0.05))
        else:
            pos = rng.integers(0, max(1, out_length - stem_len))

        pos = max(0, min(pos, out_length - 1))
        end = min(pos + stem_len, out_length)
        actual_len = end - pos

        # Check layer count
        if np.any(layer_count[pos:end] >= max_layers):
            continue

        # Prepare stem segment
        seg = stem[:actual_len].copy()
        if fade_ms > 0:
            seg = fade(seg, fade_ms, fade_ms, sr)

        # Match channel count
        if seg.ndim == 1 and n_channels > 1:
            seg = np.column_stack([seg] * n_channels)
        elif seg.ndim == 2 and n_channels == 1:
            seg = seg[:, 0]

        output[pos:end] += seg
        layer_count[pos:end] += 1

    return output


def pattern(
    stems: list[np.ndarray],
    sr: int,
    bpm: float = 120.0,
    steps: int = 16,
    bars: int = 4,
    probabilities: list[list[float]] | None = None,
    swing: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Probabilistic step sequencer.

    Define a grid and probabilistically assign stems to steps.

    Args:
        stems: List of audio buffers.
        sr: Sample rate.
        bpm: Tempo in beats per minute.
        steps: Steps per bar (e.g., 16 for 16th notes).
        bars: Number of bars.
        probabilities: Per-stem probability for each step.
            Shape: [n_stems][steps]. If None, random probabilities are generated.
        swing: Swing amount (0.0 = straight, 1.0 = max swing).
        seed: Random seed for reproducibility.
    """
    if not stems:
        return np.zeros(1, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Calculate timing
    beat_duration = 60.0 / bpm  # seconds per beat
    step_duration = beat_duration * 4.0 / steps  # seconds per step
    total_steps = steps * bars
    total_duration = total_steps * step_duration
    out_length = int(total_duration * sr)

    n_channels = max(
        (s.shape[1] if s.ndim == 2 else 1) for s in stems
    )
    if n_channels == 1:
        output = np.zeros(out_length, dtype=np.float64)
    else:
        output = np.zeros((out_length, n_channels), dtype=np.float64)

    # Generate default probabilities
    if probabilities is None:
        probabilities = [
            [rng.uniform(0.0, 0.7) for _ in range(steps)]
            for _ in stems
        ]

    for step_idx in range(total_steps):
        pattern_step = step_idx % steps

        # Calculate position with optional swing
        base_pos = step_idx * step_duration
        if swing > 0 and step_idx % 2 == 1:
            base_pos += step_duration * swing * 0.33
        sample_pos = int(base_pos * sr)

        for stem_idx, stem in enumerate(stems):
            if stem_idx >= len(probabilities):
                prob = 0.5
            else:
                prob = probabilities[stem_idx][pattern_step % len(probabilities[stem_idx])]

            if rng.random() < prob:
                stem_len = len(stem) if stem.ndim == 1 else stem.shape[0]
                end = min(sample_pos + stem_len, out_length)
                actual_len = end - sample_pos

                if actual_len <= 0 or sample_pos < 0:
                    continue

                seg = stem[:actual_len].copy()
                if seg.ndim == 1 and n_channels > 1:
                    seg = np.column_stack([seg] * n_channels)
                elif seg.ndim == 2 and n_channels == 1:
                    seg = seg[:, 0]

                output[sample_pos:end] += seg

    return output
