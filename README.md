# glitch-engine

Procedural sampling & glitch toolkit for IDM audiovisual production. Think Arca, Aphex Twin, Autechre.

No DAWs, no GUIs — scripts that take audio and visuals in and spit processed audiovisual output out. Every function is probabilistic: re-running produces different results.

## Installation

```bash
cd glitch-engine
pip install -e .

# Optional: install pyrubberband for high-quality time-stretching
pip install pyrubberband
```

Requires Python 3.10+, ffmpeg (system dependency for video rendering).

## Phase 1: Audio Processing

```bash
# Granular scatter — break audio into grains and scatter them
glitch granular input.wav -grains 200 -spread 0.8 -pitch 4 -o scattered.wav

# Dense granular cloud — textural ambient pad
glitch cloud input.wav -density 500 -pitch 12 -o cloud.wav

# Stutter glitch — repeat segments at transient points
glitch stutter input.wav -chance 0.6 -repeats 4 -o stuttered.wav

# Chop and shuffle — slice and rearrange timeline
glitch chop input.wav -slices 16 -o chopped.wav

# Spectral smear — ghostly underwater textures
glitch spectral input.wav -blur 0.5 -phase 0.8 -o smeared.wav

# Spectral mangle — full spectral destruction
glitch mangle input.wav -swap 0.5 -noise 0.8 -o mangled.wav

# BPM detection
glitch quantize input.wav --detect-only -o dummy.wav

# Time-stretch to target BPM
glitch quantize input.wav -bpm 140 -o stretched.wav

# Probabilistic arrangement
glitch sequence stem1.wav stem2.wav stem3.wav -duration 60 -density 0.5 -o arranged.wav

# Step sequencer
glitch pattern kick.wav snare.wav hat.wav -bpm 140 -steps 16 -bars 8 -o beat.wav

# Mix stems
glitch mix stem1.wav stem2.wav -o mixed.wav
```

## Phase 2: Visual Engine

Bond audio with visuals, micro-sample both in perfect sync, render to video.

```bash
# Bond audio + still image, render to video
glitch render sample.wav visual.png -o output.mp4

# Bond audio + image sequence (images switch at transients)
glitch render sample.wav frames/ -o output.mp4

# Bond audio + video clip
glitch render sample.wav clip.mp4 -o output.mp4

# Bond + micro-sample + render (synchronized audio+video glitch)
glitch render sample.wav frames/ -o output.mp4 \
  --slice 40 --shuffle 0.6 --stutter 0.3 --reverse 0.2

# Compose full track from JSON manifest
glitch compose manifest.json -o track.mp4

# Compose from folder structure
glitch compose track_folder/ -o track.mp4

# Compose with global micro-sample settings
glitch compose track_folder/ -o track.mp4 --shuffle 0.4 --stutter 0.2

# Preview first 10 seconds
glitch compose manifest.json -o preview.mp4 --preview 10

# Generate manifest template from folder
glitch manifest track_folder/ -o manifest.json

# Export cut list for reuse
glitch cutlist sample.wav frames/ --slice 30 --shuffle 0.5 -o cuts.json

# Apply saved cut list to new material
glitch apply cuts.json new_sample.wav new_frames/ -o output.mp4
```

Every command accepts `--seed N` for reproducibility.

## Full Workflow

```bash
# Phase 1: process audio
glitch granular break.wav -grains 300 -spread 0.9 -o processed_break.wav
glitch stutter vocal.wav -chance 0.7 -o processed_vocal.wav

# Phase 2: bond with visuals and micro-sample
glitch render processed_break.wav frames/ -o break.mp4 \
  --shuffle 0.5 --stutter 0.3 --reverse 0.2
glitch render processed_vocal.wav portrait.png -o vocal.mp4

# Compose everything
glitch compose my_track/ -o final.mp4
```

## Folder Structure Convention

```
track/
  kick/
    sample.wav
    visual.png           # single image -> bonded as still
  break/
    sample.wav
    001.png              # image sequence -> bonded at transients
    002.png
    003.png
  texture/
    sample.wav
    visual.mp4           # video clip -> bonded directly
```

## Python API

```python
import glitch
from glitch.av import bond, microsample, render

# Audio processing
audio, sr = glitch.load("input.wav")
scattered = glitch.scatter(audio, sr, num_grains=200, spread=0.8, pitch_drift=4)
stuttered = glitch.stutter_glitch(scattered, sr, stutter_chance=0.6, max_repeats=4)
glitch.save("processed.wav", stuttered, sr)

# AV bonding and micro-sampling
clip = bond("processed.wav", "frames/", resolution=(1920, 1080), fps=30)
result, cut_list = microsample(clip, shuffle_chance=0.5, stutter_chance=0.3, seed=42)
render(result, "output.mp4")

# Save cut list for reuse
cut_list.save("cuts.json")
```

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `core` | `load`, `save`, `normalize`, `fade`, `envelope` | Shared I/O and utilities |
| `granular` | `scatter`, `cloud` | Granular synthesis and scattering |
| `stutter` | `glitch`, `chop` | Stutter/glitch and timeline shuffling |
| `spectral` | `smear`, `mangle` | FFT-based spectral processing |
| `quantize` | `detect_bpm`, `quantize` | BPM detection and time-stretching |
| `sequence` | `arrange`, `pattern` | Probabilistic sequencing and step patterns |
| `mix` | `layer`, `mixdown` | Stem mixing and layering |
| `av.core` | `AVClip`, `CutList`, `LazyFrameList` | AV data model and utilities |
| `av.bond` | `bond` | Fuse audio + visual into AVClip |
| `av.microsample` | `microsample`, `apply_cut_list`, `chain` | Synchronized AV micro-sampling |
| `av.render` | `render`, `composite` | Render AVClips to .mp4 |
| `av.effects` | `rgb_split`, `scan_lines`, `corrupt`, `posterize`, `noise` | Per-frame visual glitch effects |

## Design Principles

1. **Pure functions** — audio/video in, audio/video out. No side effects.
2. **Sensible defaults** — every parameter has a default that sounds/looks interesting.
3. **Composable** — chain any function's output into any other.
4. **Probabilistic** — different seed = different result, same aesthetic.
5. **Audio-video lockstep** — every cut applies identically to both streams via cut lists.
6. **Minimal dependencies** — NumPy, SciPy, librosa, soundfile, moviepy, Pillow.
