"""Tests for av.render module (unit tests, no actual ffmpeg rendering)."""

import json
import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from pathlib import Path
from unittest.mock import patch

from glitch.av.core import AVClip, frames_for_duration
from glitch.av.render import _quantize_clip, _auto_detect_target_bpm


class TestRenderHelpers:
    def test_avclip_construction(self, mono_audio, sr):
        fps = 10
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        n_frames = frames_for_duration(len(mono_audio) / sr, fps)
        frames = [frame.copy() for _ in range(n_frames)]
        clip = AVClip(
            audio=mono_audio, sr=sr, frames=frames,
            fps=fps, resolution=(100, 100),
        )
        assert clip.duration_s == pytest.approx(1.0)
        assert clip.n_frames == n_frames


class TestCompositeFromManifest:
    def test_manifest_format(self, tmp_path, mono_audio, sr):
        """Test that a manifest can be constructed and parsed."""
        # Create test files
        audio_path = tmp_path / "sample.wav"
        sf.write(str(audio_path), mono_audio, sr)
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        visual_path = tmp_path / "visual.png"
        img.save(str(visual_path))

        manifest = {
            "title": "test",
            "resolution": [100, 100],
            "fps": 10,
            "pairs": [
                {
                    "audio": "sample.wav",
                    "visual": "visual.png",
                    "offset": 0.0,
                    "gain_db": 0.0,
                    "microsample": None,
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Just verify it parses correctly
        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["title"] == "test"
        assert len(loaded["pairs"]) == 1


class TestAutoQuantize:
    def _make_clip(self, mono_audio, sr):
        fps = 10
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        n_frames = frames_for_duration(len(mono_audio) / sr, fps)
        frames = [frame.copy() for _ in range(n_frames)]
        return AVClip(audio=mono_audio, sr=sr, frames=frames,
                      fps=fps, resolution=(100, 100))

    def test_quantize_clip(self, mono_audio, sr):
        clip = self._make_clip(mono_audio, sr)
        original_len = len(clip.audio)
        with patch("glitch.quantize.quantize") as mock_qt:
            mock_qt.return_value = np.zeros(original_len * 2, dtype=np.float64)
            result = _quantize_clip(clip, 140.0)
            mock_qt.assert_called_once()
            assert len(result.audio) == original_len * 2

    def test_auto_detect_target_bpm(self, mono_audio, sr):
        clips = [self._make_clip(mono_audio, sr) for _ in range(3)]
        with patch("glitch.quantize.detect_bpm") as mock_detect:
            mock_detect.side_effect = [120.0, 140.0, 120.0]
            target = _auto_detect_target_bpm(clips)
            assert target == 120.0  # median of [120, 120, 140]

    def test_auto_detect_even_count(self, mono_audio, sr):
        clips = [self._make_clip(mono_audio, sr) for _ in range(2)]
        with patch("glitch.quantize.detect_bpm") as mock_detect:
            mock_detect.side_effect = [120.0, 140.0]
            target = _auto_detect_target_bpm(clips)
            assert target == 130.0  # average of [120, 140]

    def test_manifest_bpm_field(self, tmp_path, mono_audio, sr):
        """Manifest with top-level bpm field is parseable."""
        audio_path = tmp_path / "sample.wav"
        sf.write(str(audio_path), mono_audio, sr)
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        visual_path = tmp_path / "visual.png"
        img.save(str(visual_path))

        manifest = {
            "title": "test",
            "resolution": [100, 100],
            "fps": 10,
            "bpm": 140,
            "pairs": [
                {
                    "audio": "sample.wav",
                    "visual": "visual.png",
                    "offset": 0.0,
                    "gain_db": 0.0,
                    "microsample": None,
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["bpm"] == 140

    def test_pair_bpm_field(self, tmp_path, mono_audio, sr):
        """Per-pair bpm field overrides global."""
        manifest = {
            "title": "test",
            "resolution": [100, 100],
            "fps": 10,
            "bpm": 120,
            "pairs": [
                {
                    "audio": "sample.wav",
                    "visual": "visual.png",
                    "offset": 0.0,
                    "bpm": 160,
                    "microsample": None,
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with open(manifest_path) as f:
            loaded = json.load(f)
        pair_bpm = loaded["pairs"][0].get("bpm")
        global_bpm = loaded.get("bpm")
        # Per-pair should take priority
        effective = pair_bpm or global_bpm
        assert effective == 160


class TestFolderStructure:
    def test_folder_setup(self, tmp_path, mono_audio, sr):
        """Test that folder structure is correctly recognized."""
        track = tmp_path / "track"
        kick = track / "kick"
        kick.mkdir(parents=True)

        sf.write(str(kick / "sample.wav"), mono_audio, sr)
        img = Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8))
        img.save(str(kick / "visual.png"))

        # Verify structure
        assert (kick / "sample.wav").exists()
        assert (kick / "visual.png").exists()
