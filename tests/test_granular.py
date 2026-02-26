"""Tests for granular module."""

import numpy as np
import pytest
from glitch.granular import scatter, cloud


class TestScatter:
    def test_basic_output(self, mono_audio, sr):
        result = scatter(mono_audio, sr, num_grains=50, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mono_audio)

    def test_output_duration(self, mono_audio, sr):
        result = scatter(mono_audio, sr, output_duration_s=2.0, seed=42)
        expected = int(2.0 * sr)
        assert len(result) == expected

    def test_reproducibility(self, mono_audio, sr):
        r1 = scatter(mono_audio, sr, seed=123)
        r2 = scatter(mono_audio, sr, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self, mono_audio, sr):
        r1 = scatter(mono_audio, sr, seed=1)
        r2 = scatter(mono_audio, sr, seed=2)
        assert not np.array_equal(r1, r2)

    def test_pitch_drift(self, mono_audio, sr):
        result = scatter(mono_audio, sr, pitch_drift=6.0, num_grains=50, seed=42)
        assert len(result) > 0

    def test_spread_zero(self, mono_audio, sr):
        result = scatter(mono_audio, sr, spread=0.0, num_grains=50, seed=42)
        assert len(result) == len(mono_audio)

    def test_spread_one(self, mono_audio, sr):
        result = scatter(mono_audio, sr, spread=1.0, num_grains=50, seed=42)
        assert len(result) == len(mono_audio)

    def test_stereo(self, stereo_audio, sr):
        result = scatter(stereo_audio, sr, num_grains=50, seed=42)
        assert result.ndim == 2
        assert result.shape[1] == 2


class TestCloud:
    def test_basic_output(self, mono_audio, sr):
        result = cloud(mono_audio, sr, density=100, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mono_audio)

    def test_dense(self, short_audio, sr):
        result = cloud(short_audio, sr, density=500, pitch_drift=12, seed=42)
        assert len(result) > 0
