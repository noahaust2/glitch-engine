"""Tests for cutlist module."""

import numpy as np
import pytest

from glitch.cutlist import Cut, CutList, apply_cut_list


class TestCut:
    def test_to_dict(self):
        cut = Cut(source_start_s=0.1, source_end_s=0.2, dest_start_s=0.3,
                  repeats=2, reversed=True, dropped=False)
        d = cut.to_dict()
        assert d["source_start_s"] == 0.1
        assert d["repeats"] == 2
        assert d["reversed"] is True

    def test_from_dict(self):
        d = {"source_start_s": 0.0, "source_end_s": 0.05,
             "dest_start_s": 0.0, "repeats": 1, "reversed": False, "dropped": False}
        cut = Cut.from_dict(d)
        assert cut.source_end_s == 0.05

    def test_native_types(self):
        """Ensure to_dict produces JSON-serializable native types."""
        import json
        cut = Cut(source_start_s=np.float64(0.1), source_end_s=np.float64(0.2),
                  dest_start_s=np.float64(0.0), repeats=np.int64(3),
                  reversed=np.bool_(True), dropped=np.bool_(False))
        d = cut.to_dict()
        json.dumps(d)  # Should not raise


class TestCutList:
    def test_save_load(self, tmp_path):
        cuts = [
            Cut(0.0, 0.05, 0.0, 1, False, False),
            Cut(0.05, 0.1, 0.05, 2, True, False),
        ]
        cl = CutList(cuts=cuts, seed=42, source_duration_s=1.0, output_duration_s=1.1)
        path = str(tmp_path / "cuts.json")
        cl.save(path)
        loaded = CutList.load(path)
        assert len(loaded.cuts) == 2
        assert loaded.seed == 42
        assert loaded.cuts[1].reversed is True
        assert loaded.cuts[1].repeats == 2


class TestApplyCutList:
    def test_basic(self, mono_audio, sr):
        cuts = [
            Cut(0.0, 0.1, 0.0, 1, False, False),
            Cut(0.1, 0.2, 0.1, 1, False, False),
        ]
        cl = CutList(cuts=cuts, seed=0, source_duration_s=1.0, output_duration_s=0.2)
        result = apply_cut_list(mono_audio, sr, cl)
        expected_len = int(0.1 * sr) + int(0.1 * sr)
        assert abs(len(result) - expected_len) <= 1

    def test_reversed(self, mono_audio, sr):
        cuts = [Cut(0.0, 0.1, 0.0, 1, True, False)]
        cl = CutList(cuts=cuts, seed=0, source_duration_s=1.0, output_duration_s=0.1)
        result = apply_cut_list(mono_audio, sr, cl)
        # Reversed should differ from the original slice
        orig = mono_audio[:int(0.1 * sr)]
        assert not np.array_equal(result[:len(orig)], orig)

    def test_dropped(self, mono_audio, sr):
        cuts = [Cut(0.0, 0.1, 0.0, 1, False, True)]
        cl = CutList(cuts=cuts, seed=0, source_duration_s=1.0, output_duration_s=0.1)
        result = apply_cut_list(mono_audio, sr, cl)
        assert np.all(result == 0.0)

    def test_stutter(self, mono_audio, sr):
        cuts = [Cut(0.0, 0.05, 0.0, 3, False, False)]
        cl = CutList(cuts=cuts, seed=0, source_duration_s=1.0, output_duration_s=0.15)
        result = apply_cut_list(mono_audio, sr, cl)
        # 3 repeats of 0.05s = 0.15s
        expected_len = int(0.05 * sr) * 3
        assert abs(len(result) - expected_len) <= 1
