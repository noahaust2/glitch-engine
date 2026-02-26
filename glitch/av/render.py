"""Render AVClip(s) to final .mp4 files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from glitch.av.core import AVClip, LazyFrameList, frames_for_duration
from glitch.core import normalize as peak_normalize


def render(
    clip: AVClip,
    output_path: str,
    codec: str = "libx264",
    crf: int = 18,
    audio_bitrate: str = "320k",
) -> None:
    """Render a single AVClip to an .mp4 file.

    Args:
        clip: The AVClip to render.
        output_path: Output file path (.mp4).
        codec: Video codec.
        crf: Constant rate factor (quality).
        audio_bitrate: Audio bitrate.
    """
    from moviepy import VideoClip, AudioClip

    frames = clip.frames if not isinstance(clip.frames, LazyFrameList) else clip.frames
    fps = clip.fps
    duration = clip.duration_s

    def make_frame(t):
        idx = int(t * fps)
        idx = min(idx, len(frames) - 1)
        idx = max(0, idx)
        return frames[idx]

    video = VideoClip(make_frame, duration=duration).with_fps(fps)

    # Attach audio
    audio = clip.audio.copy()
    sr = clip.sr

    # Ensure audio is stereo for moviepy
    if audio.ndim == 1:
        audio_stereo = np.column_stack([audio, audio])
    else:
        audio_stereo = audio

    def make_audio_frame(t):
        # t can be a float or an array of floats
        if isinstance(t, np.ndarray):
            indices = (t * sr).astype(int)
            indices = np.clip(indices, 0, len(audio_stereo) - 1)
            return audio_stereo[indices]
        else:
            idx = int(t * sr)
            idx = max(0, min(idx, len(audio_stereo) - 1))
            return audio_stereo[idx]

    audio_clip = AudioClip(make_audio_frame, duration=duration, fps=sr)
    video = video.with_audio(audio_clip)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    video.write_videofile(
        output_path,
        codec=codec,
        audio_codec="aac",
        audio_bitrate=audio_bitrate,
        fps=fps,
        logger=None,
        ffmpeg_params=["-crf", str(crf)],
    )


def composite(
    clips: list[tuple[AVClip, float]],
    output_path: str,
    duration_s: float | None = None,
    background: str = "black",
    resolution: tuple[int, int] | None = None,
    fps: int | None = None,
    codec: str = "libx264",
    crf: int = 18,
    audio_bitrate: str = "320k",
) -> None:
    """Composite multiple AVClips into one video.

    Args:
        clips: List of (AVClip, offset_s) tuples.
        output_path: Output file path.
        duration_s: Total output duration (default: end of last clip).
        background: Background color for empty regions.
        resolution: Output resolution (default: from first clip).
        fps: Output frame rate (default: from first clip).
        codec: Video codec.
        crf: Quality factor.
        audio_bitrate: Audio bitrate.
    """
    if not clips:
        raise ValueError("No clips to composite")

    first_clip = clips[0][0]
    if resolution is None:
        resolution = first_clip.resolution
    if fps is None:
        fps = first_clip.fps

    # Calculate duration
    if duration_s is None:
        duration_s = max(offset + c.duration_s for c, offset in clips)

    # Build composited frames
    n_frames = frames_for_duration(duration_s, fps)
    w, h = resolution

    if background == "black":
        bg_color = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        bg_color = np.zeros((h, w, 3), dtype=np.uint8)

    composited_frames = [bg_color.copy() for _ in range(n_frames)]

    # Layer clips (later entries render on top)
    for clip, offset in clips:
        start_frame = int(offset * fps)
        clip_frames = clip.frames if not isinstance(clip.frames, LazyFrameList) else clip.frames
        for i, frame in enumerate(clip_frames):
            target_idx = start_frame + i
            if 0 <= target_idx < n_frames:
                from glitch.av.core import fit_to_resolution
                resized = fit_to_resolution(frame, resolution)
                composited_frames[target_idx] = resized

    # Mix audio
    from glitch.mix import layer as audio_layer

    stems = []
    offsets = []
    gains = []
    sr = first_clip.sr

    for clip, offset in clips:
        stems.append(clip.audio)
        offsets.append(offset)
        gains.append(1.0)

    mixed_audio = audio_layer(stems, offsets, gains, sr)
    mixed_audio = peak_normalize(mixed_audio)

    # Trim/pad audio to match video duration
    target_samples = int(duration_s * sr)
    if len(mixed_audio) > target_samples:
        mixed_audio = mixed_audio[:target_samples]
    elif len(mixed_audio) < target_samples:
        if mixed_audio.ndim == 1:
            mixed_audio = np.pad(mixed_audio, (0, target_samples - len(mixed_audio)))
        else:
            mixed_audio = np.pad(mixed_audio, ((0, target_samples - len(mixed_audio)), (0, 0)))

    # Create temp AVClip and render
    composited = AVClip(
        audio=mixed_audio,
        sr=sr,
        frames=composited_frames,
        fps=fps,
        resolution=resolution,
    )
    render(composited, output_path, codec=codec, crf=crf, audio_bitrate=audio_bitrate)


def composite_from_manifest(
    manifest_path: str,
    output_path: str,
    resolution: tuple[int, int] | None = None,
    fps: int | None = None,
    global_microsample: dict | None = None,
    preview_s: float | None = None,
) -> None:
    """Load a JSON manifest, bond all pairs, optionally micro-sample, and render.

    Args:
        manifest_path: Path to manifest JSON file.
        output_path: Output .mp4 path.
        resolution: Override resolution (default: from manifest).
        fps: Override fps (default: from manifest).
        global_microsample: Global micro-sample params applied to all pairs.
        preview_s: If set, only render first N seconds.
    """
    from glitch.av.bond import bond
    from glitch.av.microsample import microsample

    with open(manifest_path) as f:
        manifest = json.load(f)

    base_dir = str(Path(manifest_path).parent)
    res = tuple(manifest.get("resolution", [1920, 1080]))
    manifest_fps = manifest.get("fps", 30)

    if resolution is not None:
        res = resolution
    if fps is not None:
        manifest_fps = fps

    clips = []
    for pair in manifest.get("pairs", []):
        audio_path = os.path.join(base_dir, pair["audio"])
        visual_path = os.path.join(base_dir, pair["visual"])

        if not os.path.exists(audio_path):
            print(f"Warning: skipping missing audio {audio_path}")
            continue
        if not os.path.exists(visual_path):
            print(f"Warning: skipping missing visual {visual_path}")
            continue

        clip = bond(audio_path, visual_path, resolution=res, fps=manifest_fps)

        # Apply micro-sampling
        ms_params = pair.get("microsample")
        if ms_params is None and global_microsample:
            ms_params = global_microsample
        if ms_params:
            # Remove 'effects' from params to handle separately
            ms_params = dict(ms_params)
            clip, _ = microsample(clip, **ms_params)

        offset = pair.get("offset", 0.0)
        gain_db = pair.get("gain_db", 0.0)
        if gain_db != 0.0:
            gain = 10.0 ** (gain_db / 20.0)
            clip.audio = clip.audio * gain

        clips.append((clip, offset))

    if not clips:
        raise ValueError("No valid pairs found in manifest")

    duration = preview_s if preview_s else None
    composite(clips, output_path, duration_s=duration, resolution=res, fps=manifest_fps)


def composite_from_folder(
    folder_path: str,
    output_path: str,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    global_microsample: dict | None = None,
    preview_s: float | None = None,
) -> None:
    """Convenience: auto-generate manifest from folder structure and composite.

    Expected structure:
        folder/
          name1/
            sample.wav
            visual.png (or visual.mp4, or numbered images)
          name2/
            sample.wav
            001.png, 002.png, ...
    """
    from glitch.av.bond import bond
    from glitch.av.microsample import microsample

    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    clips = []
    offset = 0.0

    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue

        # Find audio
        audio_path = None
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            candidate = subdir / f"sample{ext}"
            if candidate.exists():
                audio_path = str(candidate)
                break

        if audio_path is None:
            print(f"Warning: no sample audio in {subdir}, skipping")
            continue

        # Find visual
        visual_path = None
        # Check for single visual file
        for name in ["visual.png", "visual.jpg", "visual.jpeg", "visual.mp4", "visual.mov"]:
            candidate = subdir / name
            if candidate.exists():
                visual_path = str(candidate)
                break

        # Check for numbered image sequence
        if visual_path is None:
            img_files = sorted(
                f for f in subdir.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                and f.stem not in {"visual"}
            )
            if img_files:
                visual_path = str(subdir)

        if visual_path is None:
            print(f"Warning: no visual media in {subdir}, skipping")
            continue

        clip = bond(audio_path, visual_path, resolution=resolution, fps=fps)

        if global_microsample:
            clip, _ = microsample(clip, **global_microsample)

        clips.append((clip, offset))
        offset += clip.duration_s

    if not clips:
        raise ValueError(f"No valid pairs found in {folder_path}")

    duration = preview_s if preview_s else None
    composite(clips, output_path, duration_s=duration, resolution=resolution, fps=fps)
