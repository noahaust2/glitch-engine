"""Tests for core module."""

import numpy as np
import pytest
from glitch.core import load, save, normalize, fade, envelope, ms_to_samples, samples_to_ms


class TestLoadSave:
    def test_roundtrip(self, mono_audio, sr, tmp_path):
        path = str(tmp_path / "roundtrip.wav")
        save(path, mono_audio, sr)
        loaded, loaded_sr = load(path)
        assert loaded_sr == sr
        assert loaded.shape == mono_audio.shape
        # WAV default subtype is float32, so roundtrip has float32-level precision
        np.testing.assert_allclose(loaded, mono_audio, atol=1e-4)

    def test_roundtrip_stereo(self, stereo_audio, sr, tmp_path):
        path = str(tmp_path / "roundtrip_stereo.wav")
        save(path, stereo_audio, sr)
        loaded, loaded_sr = load(path)
        assert loaded_sr == sr
        assert loaded.shape == stereo_audio.shape

    def test_save_clips(self, sr, tmp_path):
        path = str(tmp_path / "clipped.wav")
        audio = np.array([2.0, -2.0, 0.5])
        save(path, audio, sr)
        loaded, _ = load(path)
        assert np.max(loaded) <= 1.0
        assert np.min(loaded) >= -1.0


class TestNormalize:
    def test_peak_normalize(self):
        audio = np.array([0.5, -0.3, 0.1])
        result = normalize(audio)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_silent_audio(self):
        audio = np.zeros(100)
        result = normalize(audio)
        np.testing.assert_array_equal(result, audio)


class TestFade:
    def test_fade_in(self, sr):
        audio = np.ones(sr)  # 1 second
        result = fade(audio, in_ms=100, out_ms=0, sr=sr)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(1.0)

    def test_fade_out(self, sr):
        audio = np.ones(sr)
        result = fade(audio, in_ms=0, out_ms=100, sr=sr)
        assert result[0] == pytest.approx(1.0)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    def test_fade_stereo(self, sr):
        audio = np.ones((sr, 2))
        result = fade(audio, in_ms=50, out_ms=50, sr=sr)
        assert result.shape == audio.shape
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)


class TestEnvelope:
    @pytest.mark.parametrize("shape", ["hann", "triangle", "tukey", "random"])
    def test_shapes(self, shape):
        env = envelope(256, shape)
        assert len(env) == 256
        assert env[0] == pytest.approx(0.0, abs=0.05)
        assert env[-1] == pytest.approx(0.0, abs=0.05)

    def test_zero_length(self):
        env = envelope(0)
        assert len(env) == 0

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            envelope(100, "invalid")


class TestConversions:
    def test_ms_to_samples(self):
        assert ms_to_samples(1000, 44100) == 44100
        assert ms_to_samples(500, 44100) == 22050

    def test_samples_to_ms(self):
        assert samples_to_ms(44100, 44100) == pytest.approx(1000.0)
        assert samples_to_ms(22050, 44100) == pytest.approx(500.0)
