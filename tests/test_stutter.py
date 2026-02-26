"""Tests for stutter module."""

import numpy as np
import pytest
from glitch.stutter import glitch, chop


class TestGlitch:
    def test_basic(self, mono_audio, sr):
        result = glitch(mono_audio, sr, seed=42)
        assert isinstance(result, np.ndarray)

    def test_preserve_length(self, mono_audio, sr):
        result = glitch(mono_audio, sr, preserve_length=True, seed=42)
        assert len(result) == len(mono_audio)

    def test_reproducibility(self, mono_audio, sr):
        r1 = glitch(mono_audio, sr, seed=42)
        r2 = glitch(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_high_stutter(self, mono_audio, sr):
        result = glitch(mono_audio, sr, stutter_chance=1.0, max_repeats=8, seed=42)
        assert len(result) > 0

    def test_with_crush(self, mono_audio, sr):
        result = glitch(mono_audio, sr, crush_chance=0.5, crush_bits=4, seed=42)
        assert len(result) > 0

    def test_stereo(self, stereo_audio, sr):
        result = glitch(stereo_audio, sr, preserve_length=True, seed=42)
        assert result.ndim == 2
        assert result.shape[0] == stereo_audio.shape[0]


class TestChop:
    def test_basic(self, mono_audio, sr):
        result = chop(mono_audio, sr, slices=8, seed=42)
        assert len(result) == len(mono_audio)

    def test_reproducibility(self, mono_audio, sr):
        r1 = chop(mono_audio, sr, seed=42)
        r2 = chop(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_with_drops(self, mono_audio, sr):
        result = chop(mono_audio, sr, drop_chance=0.5, seed=42)
        assert len(result) > 0

    def test_with_reversal(self, mono_audio, sr):
        result = chop(mono_audio, sr, reverse_chance=1.0, seed=42)
        assert len(result) > 0
