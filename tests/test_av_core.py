"""Tests for av.core module."""

import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from glitch.av.core import (
    AVClip, Cut, CutList, LazyFrameList,
    load_image, load_images, fit_to_resolution,
    frames_for_duration, detect_media_type,
    audio_to_frame_index, frame_to_audio_index,
)


@pytest.fixture
def sample_frame():
    """A 100x100 red test frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # red
    return frame


@pytest.fixture
def sample_avclip(mono_audio, sr, sample_frame):
    fps = 30
    n_frames = frames_for_duration(len(mono_audio) / sr, fps)
    frames = LazyFrameList(sample_frame, n_frames)
    return AVClip(
        audio=mono_audio, sr=sr, frames=frames,
        fps=fps, resolution=(100, 100),
    )


class TestLazyFrameList:
    def test_len(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 10)
        assert len(lfl) == 10

    def test_getitem(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 5)
        f = lfl[0]
        assert f.shape == sample_frame.shape
        np.testing.assert_array_equal(f, sample_frame)

    def test_getitem_negative(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 5)
        f = lfl[-1]
        np.testing.assert_array_equal(f, sample_frame)

    def test_getitem_oob(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 3)
        with pytest.raises(IndexError):
            lfl[5]

    def test_slice(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 10)
        sliced = lfl[2:5]
        assert len(sliced) == 3

    def test_iter(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 3)
        items = list(lfl)
        assert len(items) == 3

    def test_materialize(self, sample_frame):
        lfl = LazyFrameList(sample_frame, 5)
        materialized = lfl.materialize()
        assert isinstance(materialized, list)
        assert len(materialized) == 5
        # Modifying one shouldn't affect others (independent copies)
        materialized[0][0, 0, 0] = 42
        assert materialized[1][0, 0, 0] != 42


class TestAVClip:
    def test_duration(self, sample_avclip, sr):
        expected = 44100 / sr  # 1 second
        assert sample_avclip.duration_s == pytest.approx(expected)

    def test_n_frames(self, sample_avclip):
        assert sample_avclip.n_frames == 30  # 1 sec at 30fps


class TestCutCutList:
    def test_cut_roundtrip(self):
        cut = Cut(0.0, 0.5, 1.0, repeats=3, reversed=True, dropped=False)
        d = cut.to_dict()
        restored = Cut.from_dict(d)
        assert restored.source_start_s == 0.0
        assert restored.repeats == 3
        assert restored.reversed is True

    def test_cutlist_save_load(self, tmp_path):
        cl = CutList(
            cuts=[Cut(0.0, 0.1, 0.0), Cut(0.1, 0.2, 0.1, repeats=2, reversed=True)],
            seed=42,
            source_duration_s=1.0,
            output_duration_s=1.2,
        )
        path = str(tmp_path / "cuts.json")
        cl.save(path)
        loaded = CutList.load(path)
        assert len(loaded.cuts) == 2
        assert loaded.seed == 42
        assert loaded.cuts[1].reversed is True


class TestFitToResolution:
    def test_cover(self, sample_frame):
        result = fit_to_resolution(sample_frame, (50, 50), mode="cover")
        assert result.shape == (50, 50, 3)

    def test_contain(self, sample_frame):
        result = fit_to_resolution(sample_frame, (200, 100), mode="contain")
        assert result.shape == (100, 200, 3)

    def test_stretch(self, sample_frame):
        result = fit_to_resolution(sample_frame, (200, 50), mode="stretch")
        assert result.shape == (50, 200, 3)

    def test_same_size(self, sample_frame):
        result = fit_to_resolution(sample_frame, (100, 100))
        np.testing.assert_array_equal(result, sample_frame)


class TestLoadImage:
    def test_load(self, tmp_path):
        img = Image.fromarray(np.full((50, 80, 3), 128, dtype=np.uint8))
        path = str(tmp_path / "test.png")
        img.save(path)
        result = load_image(path, (100, 100))
        assert result.shape == (100, 100, 3)

    def test_load_images(self, tmp_path):
        for i in range(3):
            img = Image.fromarray(np.full((50, 50, 3), i * 80, dtype=np.uint8))
            img.save(str(tmp_path / f"{i:03d}.png"))
        result = load_images(str(tmp_path), (100, 100))
        assert len(result) == 3
        assert result[0].shape == (100, 100, 3)


class TestMediaDetection:
    def test_still(self, tmp_path):
        (tmp_path / "test.png").touch()
        assert detect_media_type(str(tmp_path / "test.png")) == "still"

    def test_video(self, tmp_path):
        (tmp_path / "test.mp4").touch()
        assert detect_media_type(str(tmp_path / "test.mp4")) == "video"

    def test_sequence(self, tmp_path):
        subdir = tmp_path / "frames"
        subdir.mkdir()
        assert detect_media_type(str(subdir)) == "sequence"


class TestConversions:
    def test_frames_for_duration(self):
        assert frames_for_duration(1.0, 30) == 30
        assert frames_for_duration(0.5, 24) == 12

    def test_audio_to_frame(self):
        # At 44100 sr and 30 fps, sample 44100 = frame 30
        assert audio_to_frame_index(44100, 44100, 30) == 30

    def test_frame_to_audio(self):
        assert frame_to_audio_index(30, 30, 44100) == 44100
