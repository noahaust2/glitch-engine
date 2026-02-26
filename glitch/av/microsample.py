"""Micro-sampler: synchronized audiovisual cut list generation and application."""

from __future__ import annotations

import numpy as np
import librosa

from glitch.av.core import (
    AVClip, Cut, CutList, LazyFrameList,
    audio_to_frame_index, frame_to_audio_index, frames_for_duration,
)
from glitch.av import effects as fx


def _materialize_frames(clip: AVClip) -> list[np.ndarray]:
    """Ensure frames are a real list (not LazyFrameList) for random access writes."""
    if isinstance(clip.frames, LazyFrameList):
        return clip.frames.materialize()
    return list(clip.frames)


def _extract_audio_slice(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract audio slice, clamped to bounds."""
    length = len(audio) if audio.ndim == 1 else audio.shape[0]
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    return audio[start:end].copy()


def _extract_frame_slice(frames: list[np.ndarray], start: int, end: int) -> list[np.ndarray]:
    """Extract frame slice, clamped to bounds."""
    start = max(0, min(start, len(frames)))
    end = max(start, min(end, len(frames)))
    return [f.copy() for f in frames[start:end]]


def _black_frame(resolution: tuple[int, int]) -> np.ndarray:
    """Create a black frame at the given resolution."""
    w, h = resolution
    return np.zeros((h, w, 3), dtype=np.uint8)


def microsample(
    clip: AVClip,
    slice_ms: float | None = None,
    slice_min_ms: float | None = None,
    slice_max_ms: float | None = None,
    mode: str = "transients",
    shuffle_chance: float = 0.3,
    stutter_chance: float = 0.2,
    max_repeats: int = 4,
    reverse_chance: float = 0.15,
    drop_chance: float = 0.05,
    preserve_length: bool = True,
    effects: list[str] | None = None,
    effect_chance: float = 0.3,
    seed: int | None = None,
) -> tuple[AVClip, CutList]:
    """Micro-sample an AVClip with synchronized audio+video cuts.

    Args:
        clip: Input AVClip.
        slice_ms: Fixed slice length (mutually exclusive with min/max).
        slice_min_ms: Minimum random slice length.
        slice_max_ms: Maximum random slice length.
        mode: 'fixed', 'transients', or 'random'.
        shuffle_chance: Probability of displacing a slice.
        stutter_chance: Probability of repeating a slice.
        max_repeats: Maximum stutter repetitions.
        reverse_chance: Probability of reversing a slice.
        drop_chance: Probability of dropping a slice (silence + black).
        preserve_length: If True, output matches input duration.
        effects: List of visual effect names to randomly apply.
        effect_chance: Probability of applying effects to a frame.
        seed: Random seed for reproducibility.

    Returns:
        (output AVClip, CutList)
    """
    rng = np.random.default_rng(seed)
    actual_seed = seed if seed is not None else int(rng.integers(0, 2**31))

    frames = _materialize_frames(clip)
    audio = clip.audio
    sr = clip.sr
    fps = clip.fps
    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    duration_s = audio_len / sr

    # Determine slice boundaries
    boundaries = _compute_boundaries(
        audio, sr, slice_ms, slice_min_ms, slice_max_ms, mode, rng
    )

    # Build slices as (start_sample, end_sample) pairs
    slices = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    slices = [(s, e) for s, e in slices if e > s]

    # Process slices
    out_audio_parts = []
    out_frame_parts = []
    cuts = []
    current_dest_s = 0.0

    # Shuffled indices for position displacement
    shuffled_order = list(range(len(slices)))
    for i in range(len(shuffled_order)):
        if rng.random() < shuffle_chance:
            j = rng.integers(0, len(shuffled_order))
            shuffled_order[i], shuffled_order[j] = shuffled_order[j], shuffled_order[i]

    for idx in shuffled_order:
        start_samp, end_samp = slices[idx]
        source_start_s = start_samp / sr
        source_end_s = end_samp / sr

        # Extract audio slice
        a_slice = _extract_audio_slice(audio, start_samp, end_samp)

        # Extract corresponding frame slice
        f_start = audio_to_frame_index(start_samp, sr, fps)
        f_end = audio_to_frame_index(end_samp, sr, fps)
        f_end = max(f_start + 1, f_end)
        f_slice = _extract_frame_slice(frames, f_start, f_end)

        # Decide: drop?
        dropped = rng.random() < drop_chance
        if dropped:
            silence = np.zeros_like(a_slice)
            black_frames = [_black_frame(clip.resolution)] * len(f_slice)

            cuts.append(Cut(
                source_start_s=source_start_s,
                source_end_s=source_end_s,
                dest_start_s=current_dest_s,
                repeats=1,
                reversed=False,
                dropped=True,
            ))
            out_audio_parts.append(silence)
            out_frame_parts.extend(black_frames)
            current_dest_s += len(silence) / sr
            continue

        # Decide: reverse?
        is_reversed = rng.random() < reverse_chance

        # Decide: stutter?
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
            a_part = a_slice.copy()
            f_part = [f.copy() for f in f_slice]

            if is_reversed:
                a_part = a_part[::-1]
                f_part = f_part[::-1]

            # Apply visual effects
            if effects and f_part:
                f_part = _apply_effects(f_part, effects, effect_chance, rng)

            out_audio_parts.append(a_part)
            out_frame_parts.extend(f_part)
            current_dest_s += len(a_part) / sr

    # Assemble output
    if out_audio_parts:
        out_audio = np.concatenate(out_audio_parts)
    else:
        out_audio = np.zeros(1, dtype=np.float64)

    # Handle preserve_length
    if preserve_length:
        if len(out_audio) > audio_len:
            out_audio = out_audio[:audio_len]
        elif len(out_audio) < audio_len:
            if out_audio.ndim == 1:
                out_audio = np.pad(out_audio, (0, audio_len - len(out_audio)))
            else:
                out_audio = np.pad(out_audio, ((0, audio_len - len(out_audio)), (0, 0)))

        target_frames = frames_for_duration(duration_s, fps)
        if len(out_frame_parts) > target_frames:
            out_frame_parts = out_frame_parts[:target_frames]
        elif len(out_frame_parts) < target_frames:
            black = _black_frame(clip.resolution)
            out_frame_parts.extend([black] * (target_frames - len(out_frame_parts)))

    out_duration = len(out_audio) / sr if out_audio.ndim == 1 else out_audio.shape[0] / sr
    cut_list = CutList(
        cuts=cuts,
        seed=actual_seed,
        source_duration_s=duration_s,
        output_duration_s=out_duration,
    )

    return AVClip(
        audio=out_audio,
        sr=sr,
        frames=out_frame_parts,
        fps=fps,
        resolution=clip.resolution,
    ), cut_list


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
            boundaries.append(min(pos, audio_len))
        if boundaries[-1] != audio_len:
            boundaries.append(audio_len)
        return boundaries

    # Default: fixed
    if slice_ms is None:
        slice_ms = 50.0
    step = max(1, int(slice_ms * sr / 1000))
    boundaries = list(range(0, audio_len, step))
    if boundaries[-1] != audio_len:
        boundaries.append(audio_len)
    return boundaries


def _apply_effects(
    frames: list[np.ndarray],
    effect_names: list[str],
    chance: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply visual effects to frames probabilistically."""
    effect_map = {
        "rgb_split": lambda f: fx.rgb_split(f, rng=rng),
        "scan_lines": lambda f: fx.scan_lines(f, rng=rng),
        "corrupt": lambda f: fx.corrupt(f, rng=rng),
        "invert_region": lambda f: fx.invert_region(f, rng=rng),
        "posterize": lambda f: fx.posterize(f),
        "noise": lambda f: fx.noise(f, rng=rng),
    }

    result = []
    for frame in frames:
        if rng.random() < chance:
            for name in effect_names:
                if name in effect_map:
                    frame = effect_map[name](frame)
        result.append(frame)
    return result


def apply_cut_list(clip: AVClip, cut_list: CutList) -> AVClip:
    """Re-apply a previously generated cut list to a clip.

    This lets you reuse a set of cuts on different material.
    """
    frames = _materialize_frames(clip)
    audio = clip.audio
    sr = clip.sr
    fps = clip.fps
    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]

    out_audio_parts = []
    out_frame_parts = []

    for cut in cut_list.cuts:
        start_samp = int(cut.source_start_s * sr)
        end_samp = int(cut.source_end_s * sr)

        if cut.dropped:
            silence_len = end_samp - start_samp
            silence = np.zeros(silence_len, dtype=np.float64)
            if audio.ndim == 2:
                silence = np.zeros((silence_len, audio.shape[1]), dtype=np.float64)
            n_black = max(1, audio_to_frame_index(end_samp, sr, fps) - audio_to_frame_index(start_samp, sr, fps))
            black = _black_frame(clip.resolution)

            out_audio_parts.append(silence)
            out_frame_parts.extend([black] * n_black)
            continue

        a_slice = _extract_audio_slice(audio, start_samp, end_samp)
        f_start = audio_to_frame_index(start_samp, sr, fps)
        f_end = max(f_start + 1, audio_to_frame_index(end_samp, sr, fps))
        f_slice = _extract_frame_slice(frames, f_start, f_end)

        for _ in range(cut.repeats):
            a_part = a_slice.copy()
            f_part = [f.copy() for f in f_slice]

            if cut.reversed:
                a_part = a_part[::-1]
                f_part = f_part[::-1]

            out_audio_parts.append(a_part)
            out_frame_parts.extend(f_part)

    out_audio = np.concatenate(out_audio_parts) if out_audio_parts else np.zeros(1, dtype=np.float64)

    return AVClip(
        audio=out_audio,
        sr=sr,
        frames=out_frame_parts,
        fps=fps,
        resolution=clip.resolution,
    )


def chain(
    clip: AVClip,
    operations: list[dict],
) -> tuple[AVClip, list[CutList]]:
    """Apply multiple micro-sampling passes in sequence.

    Args:
        clip: Input AVClip.
        operations: List of parameter dicts for microsample().

    Returns:
        (final AVClip, list of CutLists from each pass)
    """
    current = clip
    all_cuts = []

    for params in operations:
        current, cut_list = microsample(current, **params)
        all_cuts.append(cut_list)

    return current, all_cuts
