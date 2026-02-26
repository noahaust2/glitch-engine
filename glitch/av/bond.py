"""Bonding: fuse audio + visual media into a unified AVClip."""

from __future__ import annotations

import numpy as np
import librosa

from glitch.core import load as load_audio
from glitch.av.core import (
    AVClip, LazyFrameList,
    load_image, load_images, load_video, fit_to_resolution,
    detect_media_type, frames_for_duration, audio_to_frame_index,
)


def bond(
    audio_path: str,
    visual_path: str,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
) -> AVClip:
    """Bond audio + visual into a single AVClip.

    Auto-detects visual type:
        - Single image file -> still mode
        - Directory of images -> sequence mode (images switch at transients)
        - Video file -> video mode (trim/loop to match audio)

    Args:
        audio_path: Path to audio file.
        visual_path: Path to image, directory of images, or video file.
        resolution: Output resolution (width, height).
        fps: Output frame rate.

    Returns:
        Bonded AVClip.
    """
    audio, sr = load_audio(audio_path)
    media_type = detect_media_type(visual_path)

    if media_type == "still":
        image = load_image(visual_path, resolution)
        return _bond_still(audio, sr, image, resolution, fps)
    elif media_type == "sequence":
        images = load_images(visual_path, resolution)
        if not images:
            raise ValueError(f"No images found in {visual_path}")
        return _bond_sequence(audio, sr, images, resolution, fps)
    elif media_type == "video":
        return _bond_video(audio, sr, visual_path, resolution, fps)
    else:
        raise ValueError(f"Unknown media type: {media_type}")


def _bond_still(
    audio: np.ndarray,
    sr: int,
    image: np.ndarray,
    resolution: tuple[int, int],
    fps: int,
) -> AVClip:
    """Bond a still image: hold for the full audio duration."""
    duration_s = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
    n_frames = frames_for_duration(duration_s, fps)
    frames = LazyFrameList(image, n_frames)
    return AVClip(audio=audio, sr=sr, frames=frames, fps=fps, resolution=resolution)


def _bond_sequence(
    audio: np.ndarray,
    sr: int,
    images: list[np.ndarray],
    resolution: tuple[int, int],
    fps: int,
) -> AVClip:
    """Bond an image sequence: images change at musically meaningful transients."""
    mono = audio if audio.ndim == 1 else librosa.to_mono(audio.T)
    audio_len = len(mono)
    duration_s = audio_len / sr
    n_frames = frames_for_duration(duration_s, fps)
    n_images = len(images)

    if n_images == 1:
        frames = LazyFrameList(images[0], n_frames)
        return AVClip(audio=audio, sr=sr, frames=frames, fps=fps, resolution=resolution)

    # Detect onsets for transition points
    n_transitions = n_images - 1
    onset_frames_librosa = librosa.onset.onset_detect(y=mono, sr=sr, units="samples")

    if len(onset_frames_librosa) >= n_transitions:
        # Pick the N-1 most prominent onsets (use onset strength)
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onset_times = librosa.onset.onset_detect(
            y=mono, sr=sr, units="time"
        )
        onset_samples = librosa.onset.onset_detect(y=mono, sr=sr, units="samples")

        # Get strengths at onset positions
        onset_frame_indices = librosa.onset.onset_detect(y=mono, sr=sr)
        strengths = []
        for idx in onset_frame_indices:
            if idx < len(onset_env):
                strengths.append(onset_env[idx])
            else:
                strengths.append(0.0)

        # Sort by strength, take top N-1
        paired = list(zip(onset_samples, strengths))
        paired.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([s for s, _ in paired[:n_transitions]])
    else:
        # Fewer transients than needed: subdivide the largest segments evenly
        boundaries = [0] + sorted(onset_frames_librosa.tolist()) + [audio_len]
        while len(boundaries) - 1 < n_images:
            # Find largest segment and split it
            max_gap = 0
            max_idx = 0
            for i in range(len(boundaries) - 1):
                gap = boundaries[i + 1] - boundaries[i]
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            mid = (boundaries[max_idx] + boundaries[max_idx + 1]) // 2
            boundaries.insert(max_idx + 1, mid)
        # Use interior boundaries as transitions
        selected = boundaries[1:n_images]

    # Build frame list: each image fills frames between transitions
    transition_frames = [audio_to_frame_index(s, sr, fps) for s in selected]
    transition_frames = [0] + transition_frames + [n_frames]

    frames = []
    for i in range(n_images):
        start_f = transition_frames[i]
        end_f = transition_frames[i + 1]
        count = max(1, end_f - start_f)
        frames.extend([images[i]] * count)

    # Ensure exact frame count
    while len(frames) < n_frames:
        frames.append(images[-1])
    frames = frames[:n_frames]

    return AVClip(audio=audio, sr=sr, frames=frames, fps=fps, resolution=resolution)


def _bond_video(
    audio: np.ndarray,
    sr: int,
    video_path: str,
    resolution: tuple[int, int],
    fps: int,
) -> AVClip:
    """Bond a video clip: trim/loop video frames to match audio duration."""
    video_frames, _, _ = load_video(video_path, resolution, fps)

    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    audio_duration = audio_len / sr
    n_frames_needed = frames_for_duration(audio_duration, fps)

    if not video_frames:
        # Empty video: black frames
        black = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        frames = [black] * n_frames_needed
    elif len(video_frames) >= n_frames_needed:
        # Video longer or same: trim
        frames = video_frames[:n_frames_needed]
    else:
        # Video shorter: loop
        frames = []
        while len(frames) < n_frames_needed:
            remaining = n_frames_needed - len(frames)
            frames.extend(video_frames[:remaining])

    return AVClip(audio=audio, sr=sr, frames=frames, fps=fps, resolution=resolution)
