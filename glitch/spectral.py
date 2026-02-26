"""FFT-based spectral processing — metallic, underwater, ghostly textures."""

import numpy as np
from scipy.ndimage import uniform_filter1d


def smear(
    audio: np.ndarray,
    sr: int,
    phase_randomize: float = 0.5,
    blur: float = 0.3,
    freeze_chance: float = 0.1,
    frame_size: int = 2048,
    hop_size: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    """FFT-based spectral smearing.

    Windows the audio into overlapping frames, manipulates spectral data,
    and reconstructs via overlap-add.

    Args:
        audio: Input audio buffer (mono or stereo).
        sr: Sample rate.
        phase_randomize: 0.0 = original phase, 1.0 = fully random phase.
        blur: 0.0 = no blur, 1.0 = heavy Gaussian blur on magnitude spectrum.
        freeze_chance: Probability of freezing a frame (reusing previous frame's spectrum).
        frame_size: FFT window size.
        hop_size: Hop size for overlap-add.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Handle stereo by processing each channel
    if audio.ndim == 2:
        channels = [
            smear(audio[:, ch], sr, phase_randomize, blur, freeze_chance,
                  frame_size, hop_size, seed=seed)
            for ch in range(audio.shape[1])
        ]
        min_len = min(len(ch) for ch in channels)
        return np.column_stack([ch[:min_len] for ch in channels])

    # Mono processing
    input_length = len(audio)
    window = np.hanning(frame_size)
    # Enough frames to cover all input samples
    n_frames = max(1, int(np.ceil((input_length - frame_size) / hop_size)) + 1)
    out_length = max(input_length, (n_frames - 1) * hop_size + frame_size)
    output = np.zeros(out_length, dtype=np.float64)
    window_sum = np.zeros(out_length, dtype=np.float64)

    prev_spectrum = None

    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frame = np.zeros(frame_size, dtype=np.float64)
        actual_end = min(end, len(audio))
        frame[: actual_end - start] = audio[start:actual_end]
        frame *= window

        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Spectral freeze
        if prev_spectrum is not None and rng.random() < freeze_chance:
            spectrum = prev_spectrum.copy()
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

        # Phase randomization
        if phase_randomize > 0:
            random_phase = rng.uniform(-np.pi, np.pi, len(phase))
            phase = phase * (1.0 - phase_randomize) + random_phase * phase_randomize

        # Magnitude blur
        if blur > 0:
            blur_width = max(1, int(blur * len(magnitude) * 0.1))
            magnitude = uniform_filter1d(magnitude, size=blur_width)

        # Reconstruct
        spectrum = magnitude * np.exp(1j * phase)
        prev_spectrum = spectrum.copy()

        reconstructed = np.fft.irfft(spectrum, n=frame_size)
        reconstructed *= window

        output[start:start + frame_size] += reconstructed
        window_sum[start:start + frame_size] += window ** 2

    # Normalize by window sum (avoid division by zero)
    mask = window_sum > 1e-8
    output[mask] /= window_sum[mask]

    # Trim to original length
    return output[: len(audio)]


def mangle(
    audio: np.ndarray,
    sr: int,
    bin_swap_chance: float = 0.3,
    zero_band_chance: float = 0.2,
    noise_amount: float = 0.5,
    frame_size: int = 2048,
    hop_size: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    """Aggressive spectral destruction.

    Randomly swaps FFT bins between frames, zeros out frequency bands,
    and multiplies magnitudes by random noise.

    Args:
        audio: Input audio buffer.
        sr: Sample rate.
        bin_swap_chance: Probability of swapping bins between adjacent frames.
        zero_band_chance: Probability of zeroing a frequency band.
        noise_amount: Amount of random magnitude noise (0.0-1.0).
        frame_size: FFT window size.
        hop_size: Hop size for overlap-add.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if audio.ndim == 2:
        channels = [
            mangle(audio[:, ch], sr, bin_swap_chance, zero_band_chance,
                   noise_amount, frame_size, hop_size, seed=seed)
            for ch in range(audio.shape[1])
        ]
        min_len = min(len(ch) for ch in channels)
        return np.column_stack([ch[:min_len] for ch in channels])

    input_length = len(audio)
    window = np.hanning(frame_size)
    n_frames = max(1, int(np.ceil((input_length - frame_size) / hop_size)) + 1)
    out_length = max(input_length, (n_frames - 1) * hop_size + frame_size)
    output = np.zeros(out_length, dtype=np.float64)
    window_sum = np.zeros(out_length, dtype=np.float64)

    # Pre-compute all spectra for bin swapping
    spectra = []
    for i in range(n_frames):
        start = i * hop_size
        frame = np.zeros(frame_size, dtype=np.float64)
        actual_end = min(start + frame_size, len(audio))
        frame[: actual_end - start] = audio[start:actual_end]
        frame *= window
        spectra.append(np.fft.rfft(frame))

    n_bins = len(spectra[0])

    for i in range(n_frames):
        spectrum = spectra[i].copy()
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Swap bins with adjacent frame
        if i > 0 and rng.random() < bin_swap_chance:
            other = spectra[i - 1]
            swap_mask = rng.random(n_bins) < 0.5
            magnitude[swap_mask] = np.abs(other[swap_mask])

        # Zero out random frequency bands
        if rng.random() < zero_band_chance:
            band_start = rng.integers(0, n_bins)
            band_width = rng.integers(1, max(2, n_bins // 8))
            band_end = min(band_start + band_width, n_bins)
            magnitude[band_start:band_end] = 0.0

        # Multiply magnitudes by random noise
        if noise_amount > 0:
            noise = 1.0 + (rng.random(n_bins) - 0.5) * 2.0 * noise_amount
            magnitude *= np.maximum(noise, 0.0)

        spectrum = magnitude * np.exp(1j * phase)
        reconstructed = np.fft.irfft(spectrum, n=frame_size)
        reconstructed *= window

        start = i * hop_size
        output[start:start + frame_size] += reconstructed
        window_sum[start:start + frame_size] += window ** 2

    mask = window_sum > 1e-8
    output[mask] /= window_sum[mask]

    return output[: len(audio)]
