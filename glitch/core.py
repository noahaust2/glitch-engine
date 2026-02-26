"""Shared utilities: load/save audio, normalize, fade, envelope."""

import numpy as np
import soundfile as sf


def load(path: str) -> tuple[np.ndarray, int]:
    """Load any common audio format, return float64 array + sample rate.

    Always preserves original sample rate. Mono files return shape (N,),
    stereo return shape (N, 2). All values normalized to [-1.0, 1.0].
    """
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    return audio, sr


def save(path: str, audio: np.ndarray, sr: int) -> None:
    """Write audio to wav or flac (inferred from extension)."""
    # Clip to prevent distortion on write
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak normalize to [-1.0, 1.0]."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def fade(audio: np.ndarray, in_ms: float, out_ms: float, sr: int) -> np.ndarray:
    """Apply fade in/out to audio buffer."""
    result = audio.copy()
    in_samples = ms_to_samples(in_ms, sr)
    out_samples = ms_to_samples(out_ms, sr)

    length = len(result) if result.ndim == 1 else result.shape[0]

    if in_samples > 0:
        in_samples = min(in_samples, length)
        fade_in = np.linspace(0.0, 1.0, in_samples)
        if result.ndim == 2:
            fade_in = fade_in[:, np.newaxis]
        result[:in_samples] *= fade_in

    if out_samples > 0:
        out_samples = min(out_samples, length)
        fade_out = np.linspace(1.0, 0.0, out_samples)
        if result.ndim == 2:
            fade_out = fade_out[:, np.newaxis]
        result[-out_samples:] *= fade_out

    return result


def envelope(length: int, shape: str = "hann") -> np.ndarray:
    """Generate a grain envelope of the given length.

    Shapes: 'hann', 'triangle', 'tukey', 'random'
    """
    if length <= 0:
        return np.array([], dtype=np.float64)

    if shape == "hann":
        return np.hanning(length)
    elif shape == "triangle":
        return np.bartlett(length)
    elif shape == "tukey":
        from scipy.signal import windows
        return windows.tukey(length, alpha=0.5)
    elif shape == "random":
        env = np.random.rand(length)
        # Smooth it a bit and force zero at endpoints
        from scipy.ndimage import uniform_filter1d
        env = uniform_filter1d(env, size=max(3, length // 10))
        env[0] = 0.0
        env[-1] = 0.0
        return env / (np.max(env) if np.max(env) > 0 else 1.0)
    else:
        raise ValueError(f"Unknown envelope shape: {shape}")


def ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to sample count."""
    return int(round(ms * sr / 1000.0))


def samples_to_ms(samples: int, sr: int) -> float:
    """Convert sample count to milliseconds."""
    return samples * 1000.0 / sr
