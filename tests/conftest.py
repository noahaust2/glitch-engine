"""Shared test fixtures — generate synthetic audio for testing."""

import numpy as np
import pytest
import os
import tempfile
import soundfile as sf


@pytest.fixture
def sr():
    return 44100


@pytest.fixture
def mono_audio(sr):
    """1 second of synthetic audio: sine wave + some transients."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add some transients (clicks)
    for pos in [0.1, 0.3, 0.5, 0.7, 0.9]:
        idx = int(pos * sr)
        click_len = min(200, sr - idx)
        audio[idx : idx + click_len] += 0.8 * np.exp(
            -np.linspace(0, 10, click_len)
        )
    return np.clip(audio, -1.0, 1.0)


@pytest.fixture
def stereo_audio(mono_audio):
    """Stereo version of mono_audio."""
    return np.column_stack([mono_audio, mono_audio * 0.8])


@pytest.fixture
def short_audio(sr):
    """Short 0.1 second audio for fast tests."""
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def wav_file(mono_audio, sr, tmp_path):
    """Write mono audio to a temp wav file."""
    path = str(tmp_path / "test.wav")
    sf.write(path, mono_audio, sr)
    return path


@pytest.fixture
def stereo_wav_file(stereo_audio, sr, tmp_path):
    """Write stereo audio to a temp wav file."""
    path = str(tmp_path / "test_stereo.wav")
    sf.write(path, stereo_audio, sr)
    return path
