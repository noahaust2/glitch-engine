"""Tests for av.bond module."""

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from glitch.av.bond import bond, _bond_still, _bond_sequence
from glitch.av.core import AVClip, LazyFrameList, frames_for_duration


@pytest.fixture
def audio_file(mono_audio, sr, tmp_path):
    path = str(tmp_path / "sample.wav")
    sf.write(path, mono_audio, sr)
    return path


@pytest.fixture
def image_file(tmp_path):
    img = Image.fromarray(np.full((100, 160, 3), 200, dtype=np.uint8))
    path = str(tmp_path / "visual.png")
    img.save(path)
    return path


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(4):
        img = Image.fromarray(np.full((100, 160, 3), i * 50, dtype=np.uint8))
        img.save(str(d / f"{i:03d}.png"))
    return str(d)


class TestBondStill:
    def test_basic(self, audio_file, image_file):
        clip = bond(audio_file, image_file, resolution=(160, 100), fps=10)
        assert isinstance(clip, AVClip)
        assert isinstance(clip.frames, LazyFrameList)
        assert clip.resolution == (160, 100)
        assert clip.fps == 10
        assert clip.n_frames == frames_for_duration(clip.duration_s, 10)

    def test_frame_shape(self, audio_file, image_file):
        clip = bond(audio_file, image_file, resolution=(160, 100), fps=10)
        frame = clip.frames[0]
        assert frame.shape == (100, 160, 3)


class TestBondSequence:
    def test_basic(self, audio_file, image_dir):
        clip = bond(audio_file, image_dir, resolution=(160, 100), fps=10)
        assert isinstance(clip, AVClip)
        # Should have real list, not lazy (images change)
        assert not isinstance(clip.frames, LazyFrameList)
        assert clip.n_frames == frames_for_duration(clip.duration_s, 10)

    def test_images_change(self, audio_file, image_dir):
        clip = bond(audio_file, image_dir, resolution=(160, 100), fps=10)
        # Different images should appear at different times
        first_frame = clip.frames[0]
        last_frame = clip.frames[-1]
        # With 4 images over 1 second, they shouldn't all be identical
        assert not np.array_equal(first_frame, last_frame)


class TestBondInternal:
    def test_bond_still_lazy(self, mono_audio, sr):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        clip = _bond_still(mono_audio, sr, img, (100, 100), 30)
        assert isinstance(clip.frames, LazyFrameList)
        assert len(clip.frames) == frames_for_duration(len(mono_audio) / sr, 30)

    def test_bond_sequence_single_image(self, mono_audio, sr):
        images = [np.full((100, 100, 3), 128, dtype=np.uint8)]
        clip = _bond_sequence(mono_audio, sr, images, (100, 100), 30)
        assert isinstance(clip.frames, LazyFrameList)
