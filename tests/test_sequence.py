"""Tests for sequence module."""

import numpy as np
import pytest
from glitch.sequence import arrange, pattern


class TestArrange:
    def test_basic(self, short_audio, sr):
        result = arrange([short_audio], sr, duration_s=1.0, seed=42)
        expected_len = int(1.0 * sr)
        assert len(result) == expected_len

    def test_multiple_stems(self, short_audio, sr):
        stems = [short_audio, short_audio * 0.5, short_audio * 0.3]
        result = arrange(stems, sr, duration_s=2.0, density=0.5, seed=42)
        assert len(result) == int(2.0 * sr)

    def test_empty_stems(self, sr):
        result = arrange([], sr, duration_s=1.0)
        assert len(result) == sr

    def test_reproducibility(self, short_audio, sr):
        stems = [short_audio]
        r1 = arrange(stems, sr, seed=42)
        r2 = arrange(stems, sr, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_sparse_vs_dense(self, short_audio, sr):
        stems = [short_audio]
        sparse = arrange(stems, sr, density=0.1, duration_s=2.0, seed=42)
        dense = arrange(stems, sr, density=0.9, duration_s=2.0, seed=42)
        # Dense should have more non-zero samples
        assert np.count_nonzero(dense) > np.count_nonzero(sparse)


class TestPattern:
    def test_basic(self, short_audio, sr):
        result = pattern([short_audio], sr, bpm=120, steps=16, bars=2, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_multiple_stems(self, short_audio, sr):
        stems = [short_audio, short_audio * 0.5]
        result = pattern(stems, sr, bpm=140, steps=8, bars=4, seed=42)
        assert len(result) > 0

    def test_swing(self, short_audio, sr):
        straight = pattern([short_audio], sr, swing=0.0, seed=42)
        swung = pattern([short_audio], sr, swing=0.5, seed=42)
        # Should produce different placement
        assert not np.array_equal(straight, swung)

    def test_empty_stems(self, sr):
        result = pattern([], sr)
        assert len(result) > 0
