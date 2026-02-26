"""AVClip dataclass and shared AV utilities.

Cut and CutList are imported from glitch.cutlist (the shared data model
used by both audio-only and AV micro-sampling).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Re-export from shared module so existing imports still work
from glitch.cutlist import Cut, CutList


@dataclass
class AVClip:
    """Fundamental audiovisual unit — audio buffer + synchronized frame sequence."""
    audio: np.ndarray          # float64, [-1.0, 1.0]
    sr: int                    # sample rate
    frames: list[np.ndarray] | LazyFrameList  # (H, W, 3) uint8 arrays
    fps: int                   # frame rate
    resolution: tuple[int, int]  # (width, height)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sr if self.audio.ndim == 1 else self.audio.shape[0] / self.sr

    @property
    def n_frames(self) -> int:
        return len(self.frames)


class LazyFrameList:
    """Memory-efficient frame list for repeated frames (e.g. still images).

    Behaves like a list but stores one frame and repeats it.
    """

    def __init__(self, frame: np.ndarray, count: int):
        self._frame = frame
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._count)
            return [self._frame.copy() for _ in range(start, stop, step or 1)]
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._count})")
        return self._frame.copy()

    def __iter__(self):
        for _ in range(self._count):
            yield self._frame.copy()

    def materialize(self) -> list[np.ndarray]:
        """Expand to actual list of frames (for operations needing random write)."""
        return [self._frame.copy() for _ in range(self._count)]


# --- Utility functions ---

def load_image(path: str, resolution: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Load image via Pillow, resize/crop to target resolution, return (H, W, 3) uint8."""
    img = Image.open(path).convert("RGB")
    return fit_to_resolution(np.array(img), resolution, mode="cover")


def load_images(directory: str, resolution: tuple[int, int] = (1920, 1080)) -> list[np.ndarray]:
    """Load sorted image sequence from a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    files = sorted(
        f for f in Path(directory).iterdir()
        if f.suffix.lower() in exts
    )
    return [load_image(str(f), resolution) for f in files]


def load_video(
    path: str,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
) -> tuple[list[np.ndarray], np.ndarray | None, int]:
    """Load video with moviepy, return (frames, audio_array_or_None, sr)."""
    from moviepy import VideoFileClip

    clip = VideoFileClip(path)

    # Extract frames at target fps
    frames = []
    for t in np.arange(0, clip.duration, 1.0 / fps):
        frame = clip.get_frame(t)
        frame = fit_to_resolution(frame.astype(np.uint8), resolution)
        frames.append(frame)

    # Extract audio if present
    audio = None
    sr = 44100
    if clip.audio is not None:
        sr = clip.audio.fps
        audio_data = clip.audio.to_soundarray(fps=sr)
        audio = audio_data.astype(np.float64)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]

    clip.close()
    return frames, audio, sr


def fit_to_resolution(
    image: np.ndarray,
    resolution: tuple[int, int],
    mode: str = "cover",
) -> np.ndarray:
    """Resize image to target resolution (width, height).

    Modes:
        'cover': crop to fill (no black bars)
        'contain': letterbox with black
        'stretch': distort to fit
    """
    target_w, target_h = resolution
    h, w = image.shape[:2]

    if h == target_h and w == target_w:
        return image

    img = Image.fromarray(image)

    if mode == "stretch":
        return np.array(img.resize((target_w, target_h), Image.LANCZOS))

    elif mode == "cover":
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        # Center crop
        y_off = (new_h - target_h) // 2
        x_off = (new_w - target_w) // 2
        return arr[y_off:y_off + target_h, x_off:x_off + target_w]

    elif mode == "contain":
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        result[y_off:y_off + new_h, x_off:x_off + new_w] = np.array(img)
        return result

    else:
        raise ValueError(f"Unknown fit mode: {mode}")


def frames_for_duration(duration_s: float, fps: int) -> int:
    """How many frames needed for a given duration."""
    return max(1, int(round(duration_s * fps)))


def detect_media_type(path: str) -> str:
    """Detect whether path is a still image, image sequence directory, or video file."""
    p = Path(path)
    if p.is_dir():
        return "sequence"

    ext = p.suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".gif"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    if ext in image_exts:
        return "still"
    elif ext in video_exts:
        return "video"
    else:
        raise ValueError(f"Unknown media type for extension: {ext}")


def audio_to_frame_index(sample_index: int, sr: int, fps: int) -> int:
    """Convert audio sample position to frame index."""
    time_s = sample_index / sr
    return int(time_s * fps)


def frame_to_audio_index(frame_index: int, fps: int, sr: int) -> int:
    """Convert frame position to audio sample position."""
    time_s = frame_index / fps
    return int(time_s * sr)
