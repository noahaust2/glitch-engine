"""Tests for mix module."""

import numpy as np
import pytest
from glitch.mix import layer, mixdown


class TestLayer:
    def test_basic(self, short_audio, sr):
        result = layer([short_audio], [0.0], [1.0], sr)
        np.testing.assert_allclose(result, short_audio, atol=1e-10)

    def test_offset(self, short_audio, sr):
        result = layer([short_audio], [1.0], [1.0], sr)
        # First second should be silence
        assert np.all(result[: sr] == 0.0)
        # Audio starts at 1 second
        expected_start = sr
        np.testing.assert_allclose(
            result[expected_start : expected_start + len(short_audio)],
            short_audio, atol=1e-10
        )

    def test_gain(self, short_audio, sr):
        result = layer([short_audio], [0.0], [0.5], sr)
        np.testing.assert_allclose(result[:len(short_audio)], short_audio * 0.5, atol=1e-10)

    def test_multiple(self, short_audio, sr):
        stems = [short_audio, short_audio]
        result = layer(stems, [0.0, 0.0], [0.5, 0.5], sr)
        np.testing.assert_allclose(result[:len(short_audio)], short_audio, atol=1e-10)

    def test_empty(self):
        result = layer([], [], [], 44100)
        assert len(result) > 0


class TestMixdown:
    def test_basic(self, short_audio, sr):
        result = mixdown([short_audio], sr, do_normalize=False)
        np.testing.assert_allclose(result[:len(short_audio)], short_audio, atol=1e-10)

    def test_normalize(self, short_audio, sr):
        result = mixdown([short_audio * 0.1], sr, do_normalize=True)
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)

    def test_multiple_stems(self, short_audio, sr):
        stems = [short_audio, short_audio * 0.5]
        result = mixdown(stems, sr, do_normalize=False)
        expected = short_audio + short_audio * 0.5
        np.testing.assert_allclose(result[:len(short_audio)], expected, atol=1e-10)

    def test_different_lengths(self, sr):
        short = np.ones(1000)
        long = np.ones(2000) * 0.5
        result = mixdown([short, long], sr, do_normalize=False)
        assert len(result) == 2000
        # First 1000 samples: 1.0 + 0.5 = 1.5
        np.testing.assert_allclose(result[:1000], 1.5, atol=1e-10)
        # Last 1000 samples: only 0.5
        np.testing.assert_allclose(result[1000:], 0.5, atol=1e-10)
