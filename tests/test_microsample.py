"""Tests for microsample module."""

import numpy as np
import pytest

from glitch.microsample import microsample, chop, chain
from glitch.cutlist import Cut, CutList


class TestMicrosample:
    def test_basic(self, mono_audio, sr):
        result, cl = microsample(mono_audio, sr, seed=42)
        assert isinstance(result, np.ndarray)
        assert isinstance(cl, CutList)
        assert len(cl.cuts) > 0

    def test_reproducibility(self, mono_audio, sr):
        r1, cl1 = microsample(mono_audio, sr, seed=42)
        r2, cl2 = microsample(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)
        assert len(cl1.cuts) == len(cl2.cuts)

    def test_different_seeds(self, mono_audio, sr):
        r1, _ = microsample(mono_audio, sr, seed=1)
        r2, _ = microsample(mono_audio, sr, seed=2)
        assert not np.array_equal(r1, r2)

    def test_preserve_length(self, mono_audio, sr):
        result, _ = microsample(mono_audio, sr, preserve_length=True, seed=42)
        assert len(result) == len(mono_audio)

    def test_fixed_mode(self, mono_audio, sr):
        result, cl = microsample(mono_audio, sr, slice_ms=30, mode="fixed", seed=42)
        assert len(cl.cuts) > 0

    def test_random_mode(self, mono_audio, sr):
        result, cl = microsample(
            mono_audio, sr, mode="random",
            slice_min_ms=20, slice_max_ms=80, seed=42
        )
        assert len(cl.cuts) > 0

    def test_transients_mode(self, mono_audio, sr):
        result, cl = microsample(mono_audio, sr, mode="transients", seed=42)
        assert len(cl.cuts) > 0

    def test_high_shuffle(self, mono_audio, sr):
        result, _ = microsample(mono_audio, sr, shuffle_chance=1.0, seed=42)
        assert len(result) > 0

    def test_high_stutter(self, mono_audio, sr):
        result, _ = microsample(
            mono_audio, sr, stutter_chance=0.8, max_repeats=4,
            preserve_length=True, seed=42
        )
        assert len(result) > 0

    def test_with_drops(self, mono_audio, sr):
        result, _ = microsample(
            mono_audio, sr, drop_chance=0.5, preserve_length=True, seed=42
        )
        assert len(result) > 0

    def test_stereo(self, stereo_audio, sr):
        result, cl = microsample(stereo_audio, sr, preserve_length=True, seed=42)
        assert result.ndim == 2
        assert result.shape[0] == stereo_audio.shape[0]

    def test_cutlist_metadata(self, mono_audio, sr):
        _, cl = microsample(mono_audio, sr, seed=42)
        assert cl.seed == 42
        assert cl.source_duration_s > 0

    def test_cutlist_serialization(self, mono_audio, sr, tmp_path):
        _, cl = microsample(mono_audio, sr, seed=42)
        path = str(tmp_path / "cuts.json")
        cl.save(path)
        loaded = CutList.load(path)
        assert len(loaded.cuts) == len(cl.cuts)
        assert loaded.seed == cl.seed


class TestChop:
    def test_basic(self, mono_audio, sr):
        result, cl = chop(mono_audio, sr, slices=8, seed=42)
        assert len(result) == len(mono_audio)  # preserve_length=True by default
        assert len(cl.cuts) > 0

    def test_reproducibility(self, mono_audio, sr):
        r1, _ = chop(mono_audio, sr, seed=42)
        r2, _ = chop(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_with_drops(self, mono_audio, sr):
        result, _ = chop(mono_audio, sr, drop_chance=0.5, seed=42)
        assert len(result) > 0

    def test_with_reversal(self, mono_audio, sr):
        result, _ = chop(mono_audio, sr, reverse_chance=1.0, seed=42)
        assert len(result) > 0

    def test_returns_cutlist(self, mono_audio, sr):
        _, cl = chop(mono_audio, sr, seed=42)
        assert isinstance(cl, CutList)


class TestChain:
    def test_basic(self, mono_audio, sr):
        ops = [
            {"slice_ms": 100, "shuffle_chance": 0.3, "seed": 1},
            {"slice_ms": 50, "stutter_chance": 0.5, "seed": 2},
        ]
        result, cut_lists = chain(mono_audio, sr, ops)
        assert isinstance(result, np.ndarray)
        assert len(cut_lists) == 2
        assert all(isinstance(cl, CutList) for cl in cut_lists)

    def test_single_op(self, mono_audio, sr):
        ops = [{"slice_ms": 50, "shuffle_chance": 0.5, "seed": 42}]
        result, cut_lists = chain(mono_audio, sr, ops)
        assert len(cut_lists) == 1
