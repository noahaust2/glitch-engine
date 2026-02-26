"""Granular synthesis and scattering — the primary instrument."""

import numpy as np
from glitch.core import envelope as make_envelope, ms_to_samples


def _pitch_shift_grain(grain: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch-shift a grain via simple resampling (fast for tiny grains)."""
    if semitones == 0 or len(grain) < 2:
        return grain
    ratio = 2.0 ** (semitones / 12.0)
    new_length = max(2, int(round(len(grain) / ratio)))
    x_old = np.linspace(0, 1, len(grain))
    x_new = np.linspace(0, 1, new_length)
    if grain.ndim == 1:
        return np.interp(x_new, x_old, grain)
    # Stereo: interpolate each channel
    return np.column_stack(
        [np.interp(x_new, x_old, grain[:, ch]) for ch in range(grain.shape[1])]
    )


def scatter(
    audio: np.ndarray,
    sr: int,
    grain_min_ms: float = 10.0,
    grain_max_ms: float = 80.0,
    num_grains: int | None = None,
    density: float | None = None,
    pitch_drift: float = 0.0,
    spread: float = 0.5,
    envelope_shape: str = "hann",
    output_duration_s: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Granular scattering — slice input into grains, scatter into output.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        grain_min_ms: Minimum grain length in ms.
        grain_max_ms: Maximum grain length in ms.
        num_grains: Number of grains to place. Mutually exclusive with density.
        density: Grains per second. Mutually exclusive with num_grains.
        pitch_drift: Maximum pitch shift in semitones (bidirectional).
        spread: 0.0 = grains near original position, 1.0 = fully random.
        envelope_shape: Grain envelope type.
        output_duration_s: Output buffer length in seconds (default: same as input).
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    length = len(audio) if audio.ndim == 1 else audio.shape[0]

    if output_duration_s is None:
        out_length = length
    else:
        out_length = int(round(output_duration_s * sr))

    # Determine number of grains
    if num_grains is None and density is None:
        density = 50.0  # sensible default: 50 grains/sec
    if num_grains is None:
        num_grains = int(round(density * out_length / sr))
    num_grains = max(1, num_grains)

    # Build output buffer
    if audio.ndim == 1:
        output = np.zeros(out_length, dtype=np.float64)
    else:
        output = np.zeros((out_length, audio.shape[1]), dtype=np.float64)

    grain_min_samp = ms_to_samples(grain_min_ms, sr)
    grain_max_samp = ms_to_samples(grain_max_ms, sr)
    grain_min_samp = max(2, grain_min_samp)
    grain_max_samp = max(grain_min_samp + 1, grain_max_samp)

    for i in range(num_grains):
        # Random grain length
        g_len = rng.integers(grain_min_samp, grain_max_samp)

        # Pick source position
        max_start = max(0, length - g_len)
        src_pos = rng.integers(0, max(1, max_start + 1))

        # Extract grain
        grain = audio[src_pos : src_pos + g_len].copy()
        if len(grain) < 2:
            continue

        # Apply envelope
        env = make_envelope(len(grain), envelope_shape)
        if grain.ndim == 2:
            env = env[:, np.newaxis]
        grain = grain * env

        # Pitch shift
        if pitch_drift > 0:
            shift = rng.uniform(-pitch_drift, pitch_drift)
            grain = _pitch_shift_grain(grain, shift)

        # Place in output — blend between original position and random
        natural_pos = int(src_pos * out_length / max(1, length))
        random_pos = rng.integers(0, max(1, out_length - len(grain)))
        dest_pos = int(natural_pos * (1.0 - spread) + random_pos * spread)
        dest_pos = max(0, min(dest_pos, out_length - len(grain)))

        # Overlap-add
        end = dest_pos + len(grain)
        if end <= out_length:
            output[dest_pos:end] += grain

    return output


def cloud(
    audio: np.ndarray,
    sr: int,
    grain_min_ms: float = 1.0,
    grain_max_ms: float = 10.0,
    density: float = 500.0,
    pitch_drift: float = 12.0,
    spread: float = 1.0,
    envelope_shape: str = "hann",
    output_duration_s: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Dense granular cloud — textural/ambient variant of scatter.

    Defaults to very short grains, high density, heavy pitch drift.
    Produces Arca-style granular pads and textures.
    """
    return scatter(
        audio,
        sr,
        grain_min_ms=grain_min_ms,
        grain_max_ms=grain_max_ms,
        density=density,
        pitch_drift=pitch_drift,
        spread=spread,
        envelope_shape=envelope_shape,
        output_duration_s=output_duration_s,
        seed=seed,
    )
