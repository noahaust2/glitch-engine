"""Tests for av.effects module."""

import numpy as np
import pytest

from glitch.av.effects import (
    rgb_split, scan_lines, corrupt, invert_region, posterize, noise,
)


@pytest.fixture
def frame():
    """A 100x100 gradient test frame."""
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    f[:, :, 0] = np.arange(100, dtype=np.uint8).reshape(1, 100)  # R gradient
    f[:, :, 1] = 128
    f[:, :, 2] = 64
    return f


class TestRGBSplit:
    def test_basic(self, frame):
        result = rgb_split(frame, offset_px=5)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        # Should differ from original (channels shifted)
        assert not np.array_equal(result, frame)

    def test_with_rng(self, frame):
        rng = np.random.default_rng(42)
        r1 = rgb_split(frame, rng=rng)
        rng2 = np.random.default_rng(42)
        r2 = rgb_split(frame, rng=rng2)
        np.testing.assert_array_equal(r1, r2)


class TestScanLines:
    def test_basic(self, frame):
        result = scan_lines(frame, intensity=0.5, line_width=2)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_intensity(self, frame):
        light = scan_lines(frame, intensity=0.1)
        heavy = scan_lines(frame, intensity=0.9)
        # Heavy should be darker overall
        assert np.mean(heavy) < np.mean(light)


class TestCorrupt:
    def test_basic(self, frame):
        result = corrupt(frame, amount=0.2)
        assert result.shape == frame.shape
        # Should have some zeroed-out regions
        assert np.sum(result == 0) > np.sum(frame == 0)

    def test_amount(self, frame):
        low = corrupt(frame, amount=0.01, rng=np.random.default_rng(42))
        high = corrupt(frame, amount=0.5, rng=np.random.default_rng(42))
        # More corruption = more zeros
        assert np.sum(high == 0) >= np.sum(low == 0)


class TestInvertRegion:
    def test_basic(self, frame):
        result = invert_region(frame, amount=0.2)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


class TestPosterize:
    def test_basic(self, frame):
        result = posterize(frame, levels=4)
        assert result.shape == frame.shape
        # Should have fewer unique values
        assert len(np.unique(result)) <= len(np.unique(frame))

    def test_extreme(self, frame):
        result = posterize(frame, levels=2)
        unique = np.unique(result)
        assert len(unique) <= 3  # 0, 128 (approx)


class TestNoise:
    def test_basic(self, frame):
        result = noise(frame, amount=0.2)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, frame)

    def test_amount(self, frame):
        low = noise(frame, amount=0.01, rng=np.random.default_rng(42))
        high = noise(frame, amount=0.5, rng=np.random.default_rng(42))
        # Higher noise = more difference from original
        diff_low = np.mean(np.abs(low.astype(int) - frame.astype(int)))
        diff_high = np.mean(np.abs(high.astype(int) - frame.astype(int)))
        assert diff_high > diff_low
