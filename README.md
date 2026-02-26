# glitch-engine

Procedural micro-sampling & glitch toolkit for IDM audiovisual production. Think Arca, Aphex Twin, Autechre.

No DAWs, no GUIs — scripts that take audio and visuals in and spit processed audiovisual output out. Every function is probabilistic: re-running produces different results.

The core paradigm is **micro-sampling**: slicing audio into tiny pieces and rearranging, repeating, reversing, and dropping them. Every operation is a clean, discrete cut on a timeline — no overlapping grains, no pitch shifting, no envelopes blending together. This matters because every audio operation also applies identically to synchronized video via cut lists.

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
# Micro-sample: slice, shuffle, stutter, reverse, drop
glitch microsample input.wav --slice 40 --shuffle 0.6 --stutter 0.3 --reverse 0.2 -o out.wav

# Micro-sample and export cut list (for Phase 2 video sync)
glitch microsample input.wav --slice 30 --shuffle 0.5 -o out.wav --cutlist cuts.json

# Apply a saved cut list to different audio
glitch apply cuts.json other.wav -o out.wav

# Quick chop-and-shuffle (shortcut for microsample with 100% shuffle)
glitch chop input.wav --slices 32 --reverse 0.3 -o out.wav

# Spectral smear — ghostly underwater textures (audio-only, no cut list)
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

# Chain multiple micro-sampling passes
glitch chain input.wav \
  --op microsample --slice 50 --shuffle 0.4 \
  --op microsample --stutter 0.6 --max-repeats 8 \
  -o out.wav

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

# Compose from folder structure (auto-quantizes all clips to dominant BPM)
glitch compose track_folder/ -o track.mp4

# Compose with explicit target BPM (all clips stretched to 140)
glitch compose track_folder/ -o track.mp4 --bpm 140

# Compose with global micro-sample settings
glitch compose track_folder/ -o track.mp4 --shuffle 0.4 --stutter 0.2

# Preview first 10 seconds
glitch compose manifest.json -o preview.mp4 --preview 10

# Generate manifest template from folder
glitch manifest track_folder/ -o manifest.json

# Export cut list from AV material
glitch cutlist sample.wav frames/ --slice 30 --shuffle 0.5 -o cuts.json

# Apply saved cut list to new AV material
glitch apply cuts.json new_sample.wav new_frames/ -o output.mp4
```

Every command accepts `--seed N` for reproducibility.

## Full Workflow

```bash
# Phase 1: process audio (micro-sample, spectral)
glitch microsample break.wav --slice 40 --shuffle 0.6 --stutter 0.3 -o processed_break.wav
glitch spectral vocal.wav -blur 0.5 -phase 0.8 -o processed_vocal.wav

# Phase 2: bond with visuals and micro-sample the combined result
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

# Audio processing — micro-sample with cut list
audio, sr = glitch.load("input.wav")
processed, cut_list = glitch.microsample(audio, sr, shuffle_chance=0.6, stutter_chance=0.3, seed=42)
glitch.save("processed.wav", processed, sr)
cut_list.save("cuts.json")  # Save for Phase 2 video sync

# Spectral processing (audio-only, no cut list)
from glitch.spectral import smear
smeared = smear(audio, sr, blur=0.5, phase_randomize=0.8)

# AV bonding and micro-sampling
clip = bond("processed.wav", "frames/", resolution=(1920, 1080), fps=30)
result, av_cut_list = microsample(clip, shuffle_chance=0.5, stutter_chance=0.3, seed=42)
render(result, "output.mp4")
```

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `core` | `load`, `save`, `normalize`, `fade`, `envelope` | Shared I/O and utilities |
| `cutlist` | `Cut`, `CutList`, `apply_cut_list` | Cut list data model and serialization |
| `microsample` | `microsample`, `chop`, `chain` | Audio micro-sampling with cut list output |
| `spectral` | `smear`, `mangle` | FFT-based spectral processing (audio-only) |
| `quantize` | `detect_bpm`, `quantize` | BPM detection and time-stretching |
| `sequence` | `arrange`, `pattern` | Probabilistic sequencing and step patterns |
| `mix` | `layer`, `mixdown` | Stem mixing and layering |
| `av.core` | `AVClip`, `LazyFrameList` | AV data model and utilities |
| `av.bond` | `bond` | Fuse audio + visual into AVClip |
| `av.microsample` | `microsample`, `apply_cut_list`, `chain` | Synchronized AV micro-sampling |
| `av.render` | `render`, `composite` | Render AVClips to .mp4 |
| `av.effects` | `rgb_split`, `scan_lines`, `corrupt`, `posterize`, `noise` | Per-frame visual glitch effects |

## Design Principles

1. **Cut list is the source of truth** — every micro-sampling operation is fully described by its cut list. The same cut list applied to the same material always produces the same output. In Phase 2, the same cut list applies to video.
2. **Discrete, non-overlapping operations** — no overlapping grains, no crossfaded envelopes. Every slice is a clean cut. Essential for Phase 2 video sync.
3. **Pure functions** — audio/video in, audio/video out. No side effects.
4. **Sensible defaults** — every parameter has a default that sounds/looks interesting.
5. **Composable** — chain any function's output into any other.
6. **Probabilistic** — different seed = different result, same aesthetic.
7. **Audio-video lockstep** — every cut applies identically to both streams via cut lists.
8. **Minimal dependencies** — NumPy, SciPy, librosa, soundfile, moviepy, Pillow.
