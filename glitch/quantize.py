"""BPM detection and time-stretch quantization."""

import numpy as np
import librosa


def detect_bpm(audio: np.ndarray, sr: int) -> float:
    """Detect the BPM of an audio buffer.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.

    Returns:
        Estimated BPM as a float.
    """
    mono = audio if audio.ndim == 1 else librosa.to_mono(audio.T)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    # librosa may return an array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)


def quantize(
    audio: np.ndarray,
    sr: int,
    target_bpm: float,
    source_bpm: float | None = None,
) -> np.ndarray:
    """Time-stretch audio to match a target BPM.

    Uses pyrubberband if available, falls back to librosa phase vocoder.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        target_bpm: Desired BPM.
        source_bpm: Current BPM (auto-detected if None).

    Returns:
        Time-stretched audio buffer.
    """
    if source_bpm is None:
        source_bpm = detect_bpm(audio, sr)

    if source_bpm <= 0 or target_bpm <= 0:
        return audio.copy()

    rate = target_bpm / source_bpm

    # If rate is close to 1.0, skip processing
    if abs(rate - 1.0) < 0.01:
        return audio.copy()

    # Try pyrubberband first (needs both the Python package and rubberband CLI)
    try:
        import pyrubberband as pyrb
        if audio.ndim == 1:
            return pyrb.time_stretch(audio, sr=sr, rate=rate)
        channels = [
            pyrb.time_stretch(audio[:, ch], sr=sr, rate=rate)
            for ch in range(audio.shape[1])
        ]
        min_len = min(len(ch) for ch in channels)
        return np.column_stack([ch[:min_len] for ch in channels])
    except (ImportError, RuntimeError, OSError):
        # RuntimeError: pyrubberband installed but rubberband CLI binary missing
        # OSError: system-level file not found
        pass

    # Fallback to librosa phase vocoder
    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=rate)

    channels = [
        librosa.effects.time_stretch(audio[:, ch], rate=rate)
        for ch in range(audio.shape[1])
    ]
    min_len = min(len(ch) for ch in channels)
    return np.column_stack([ch[:min_len] for ch in channels])
