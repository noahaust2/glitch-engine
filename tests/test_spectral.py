"""Tests for spectral module."""

import numpy as np
import pytest
from glitch.spectral import smear, mangle


class TestSmear:
    def test_basic(self, mono_audio, sr):
        result = smear(mono_audio, sr, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mono_audio)

    def test_full_phase_randomize(self, mono_audio, sr):
        result = smear(mono_audio, sr, phase_randomize=1.0, seed=42)
        assert not np.array_equal(result, mono_audio)

    def test_heavy_blur(self, mono_audio, sr):
        result = smear(mono_audio, sr, blur=1.0, seed=42)
        assert len(result) > 0

    def test_freeze(self, mono_audio, sr):
        result = smear(mono_audio, sr, freeze_chance=0.8, seed=42)
        assert len(result) > 0

    def test_reproducibility(self, mono_audio, sr):
        r1 = smear(mono_audio, sr, seed=42)
        r2 = smear(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_stereo(self, stereo_audio, sr):
        result = smear(stereo_audio, sr, seed=42)
        assert result.ndim == 2
        assert result.shape[1] == 2


class TestMangle:
    def test_basic(self, mono_audio, sr):
        result = mangle(mono_audio, sr, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mono_audio)

    def test_aggressive(self, mono_audio, sr):
        result = mangle(mono_audio, sr, bin_swap_chance=1.0,
                        zero_band_chance=0.5, noise_amount=1.0, seed=42)
        assert len(result) > 0

    def test_reproducibility(self, mono_audio, sr):
        r1 = mangle(mono_audio, sr, seed=42)
        r2 = mangle(mono_audio, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)
