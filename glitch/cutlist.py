"""CutList data model — the central data structure of the entire system.

The cut list logs every decision the micro-sampler makes. In Phase 2,
the same cut list is applied to video frames, so it must be a complete,
unambiguous record of operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Cut:
    """A single micro-sampling operation record."""
    source_start_s: float      # where this slice came from in the original
    source_end_s: float        # end of source slice
    dest_start_s: float        # where it was placed in the output
    repeats: int = 1           # how many times it was repeated (1 = no repeat)
    reversed: bool = False     # whether the slice was reversed
    dropped: bool = False      # whether it was replaced with silence

    def to_dict(self) -> dict:
        return {
            "source_start_s": float(self.source_start_s),
            "source_end_s": float(self.source_end_s),
            "dest_start_s": float(self.dest_start_s),
            "repeats": int(self.repeats),
            "reversed": bool(self.reversed),
            "dropped": bool(self.dropped),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cut:
        return cls(**d)


@dataclass
class CutList:
    """Complete log of all micro-sampling operations.

    This is the single source of truth for both audio and video processing.
    The same cut list applied to the same material always produces the same output.
    """
    cuts: list[Cut] = field(default_factory=list)
    seed: int = 0
    source_duration_s: float = 0.0
    output_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cuts": [c.to_dict() for c in self.cuts],
            "seed": self.seed,
            "source_duration_s": self.source_duration_s,
            "output_duration_s": self.output_duration_s,
        }

    def save(self, path: str) -> None:
        """Serialize to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> CutList:
        return cls(
            cuts=[Cut.from_dict(c) for c in d["cuts"]],
            seed=d.get("seed", 0),
            source_duration_s=d.get("source_duration_s", 0.0),
            output_duration_s=d.get("output_duration_s", 0.0),
        )

    @classmethod
    def load(cls, path: str) -> CutList:
        """Deserialize from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def apply_cut_list(audio, sr: int, cut_list: CutList):
    """Re-apply a previously generated cut list to audio.

    This is essential for Phase 2: the same cut list gets applied to both
    audio and video.

    Args:
        audio: Input audio buffer (numpy array).
        sr: Sample rate.
        cut_list: Previously generated CutList.

    Returns:
        Processed audio array.
    """
    import numpy as np

    audio_len = len(audio) if audio.ndim == 1 else audio.shape[0]
    parts = []

    for cut in cut_list.cuts:
        start_samp = int(cut.source_start_s * sr)
        end_samp = int(cut.source_end_s * sr)
        start_samp = max(0, min(start_samp, audio_len))
        end_samp = max(start_samp, min(end_samp, audio_len))

        if cut.dropped:
            silence_len = end_samp - start_samp
            if audio.ndim == 1:
                parts.append(np.zeros(silence_len, dtype=np.float64))
            else:
                parts.append(np.zeros((silence_len, audio.shape[1]), dtype=np.float64))
            continue

        a_slice = audio[start_samp:end_samp].copy()
        for _ in range(cut.repeats):
            part = a_slice.copy()
            if cut.reversed:
                part = part[::-1]
            parts.append(part)

    if not parts:
        return audio.copy()

    return np.concatenate(parts)
