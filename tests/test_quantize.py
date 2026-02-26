"""Tests for quantize module."""

import numpy as np
import pytest
from glitch.quantize import detect_bpm, quantize


class TestDetectBPM:
    def test_returns_float(self, mono_audio, sr):
        bpm = detect_bpm(mono_audio, sr)
        assert isinstance(bpm, float)
        assert bpm > 0

    def test_stereo(self, stereo_audio, sr):
        bpm = detect_bpm(stereo_audio, sr)
        assert isinstance(bpm, float)
        assert bpm > 0


class TestQuantize:
    def test_basic(self, mono_audio, sr):
        result = quantize(mono_audio, sr, target_bpm=140)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_same_bpm(self, mono_audio, sr):
        bpm = detect_bpm(mono_audio, sr)
        result = quantize(mono_audio, sr, target_bpm=bpm, source_bpm=bpm)
        # Should return copy unchanged
        np.testing.assert_array_equal(result, mono_audio)

    def test_with_source_bpm(self, mono_audio, sr):
        result = quantize(mono_audio, sr, target_bpm=160, source_bpm=120)
        assert len(result) > 0
        # Faster tempo = shorter audio
        assert len(result) < len(mono_audio)
