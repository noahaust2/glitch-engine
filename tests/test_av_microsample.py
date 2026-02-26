"""Tests for av.microsample module."""

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from glitch.av.core import AVClip, CutList, LazyFrameList, frames_for_duration
from glitch.av.bond import bond
from glitch.av.microsample import microsample, apply_cut_list, chain


@pytest.fixture
def simple_clip(mono_audio, sr):
    """AVClip with a simple red frame."""
    fps = 10
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    n_frames = frames_for_duration(len(mono_audio) / sr, fps)
    return AVClip(
        audio=mono_audio, sr=sr,
        frames=LazyFrameList(frame, n_frames),
        fps=fps, resolution=(100, 100),
    )


class TestMicrosample:
    def test_basic(self, simple_clip):
        result, cut_list = microsample(simple_clip, seed=42)
        assert isinstance(result, AVClip)
        assert isinstance(cut_list, CutList)
        assert len(cut_list.cuts) > 0

    def test_preserve_length(self, simple_clip):
        result, _ = microsample(simple_clip, preserve_length=True, seed=42)
        orig_len = len(simple_clip.audio)
        assert len(result.audio) == orig_len

    def test_reproducibility(self, simple_clip):
        r1, cl1 = microsample(simple_clip, seed=42)
        r2, cl2 = microsample(simple_clip, seed=42)
        np.testing.assert_array_equal(r1.audio, r2.audio)
        assert len(cl1.cuts) == len(cl2.cuts)

    def test_different_seeds(self, simple_clip):
        r1, _ = microsample(simple_clip, seed=1)
        r2, _ = microsample(simple_clip, seed=2)
        # Different seeds should produce different results
        assert not np.array_equal(r1.audio, r2.audio)

    def test_fixed_mode(self, simple_clip):
        result, cl = microsample(
            simple_clip, slice_ms=50, mode="fixed", seed=42
        )
        assert len(cl.cuts) > 0

    def test_random_mode(self, simple_clip):
        result, cl = microsample(
            simple_clip, mode="random",
            slice_min_ms=20, slice_max_ms=100, seed=42
        )
        assert len(cl.cuts) > 0

    def test_transients_mode(self, simple_clip):
        result, cl = microsample(
            simple_clip, mode="transients", seed=42
        )
        assert len(cl.cuts) > 0

    def test_high_shuffle(self, simple_clip):
        result, _ = microsample(simple_clip, shuffle_chance=1.0, seed=42)
        assert len(result.audio) > 0

    def test_high_stutter(self, simple_clip):
        result, _ = microsample(
            simple_clip, stutter_chance=0.8, max_repeats=4,
            preserve_length=True, seed=42
        )
        assert len(result.audio) > 0

    def test_with_drops(self, simple_clip):
        result, _ = microsample(
            simple_clip, drop_chance=0.5, preserve_length=True, seed=42
        )
        assert len(result.audio) > 0

    def test_with_effects(self, simple_clip):
        result, _ = microsample(
            simple_clip,
            effects=["rgb_split", "scan_lines"],
            effect_chance=0.5,
            seed=42,
        )
        assert len(result.frames) > 0

    def test_cutlist_serialization(self, simple_clip, tmp_path):
        _, cl = microsample(simple_clip, seed=42)
        path = str(tmp_path / "cuts.json")
        cl.save(path)
        loaded = CutList.load(path)
        assert len(loaded.cuts) == len(cl.cuts)
        assert loaded.seed == cl.seed


class TestApplyCutList:
    def test_basic(self, simple_clip):
        _, cut_list = microsample(simple_clip, seed=42)
        result = apply_cut_list(simple_clip, cut_list)
        assert isinstance(result, AVClip)
        assert len(result.audio) > 0
        assert len(result.frames) > 0


class TestChain:
    def test_basic(self, simple_clip):
        ops = [
            {"slice_ms": 100, "shuffle_chance": 0.3, "seed": 1},
            {"slice_ms": 50, "stutter_chance": 0.5, "seed": 2},
        ]
        result, cut_lists = chain(simple_clip, ops)
        assert isinstance(result, AVClip)
        assert len(cut_lists) == 2
        assert all(isinstance(cl, CutList) for cl in cut_lists)
