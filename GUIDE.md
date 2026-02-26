# glitch-engine User Guide

## What This Is

A command-line toolkit for making glitchy, generative IDM music and visuals. You feed it audio files (and optionally images/video), and it spits out processed, destroyed, rearranged versions. Everything is randomized — run the same command twice and you get two different (but aesthetically related) results.

There are two layers:
1. **Audio processing** — micro-sampling (slice/shuffle/stutter/reverse/drop), spectral mangling, sequencing
2. **Visual engine** — bond audio to visuals, then micro-sample both in perfect sync

The core paradigm is **micro-sampling**: slicing audio into tiny pieces and rearranging them. Every operation is a discrete, non-overlapping cut. This is critical because every audio cut also applies identically to synchronized video via cut lists.

---

## Setup

```bash
cd glitch-engine
pip install -e .
```

You now have a `glitch` command available everywhere. You also need **ffmpeg** installed on your system for video rendering (the `render` and `compose` commands).

---

## Audio Commands

### `glitch microsample` — The Primary Instrument

Slices your audio into tiny pieces and processes each slice with probabilistic decisions: shuffle (move to random position), stutter (repeat N times), reverse, or drop (replace with silence). Every decision is logged to a **cut list** that can be saved and reused.

```bash
# Basic: 40ms slices, moderate shuffle and stutter
glitch microsample drums.wav --slice 40 --shuffle 0.5 --stutter 0.3 -o out.wav

# Heavy glitch: high shuffle, lots of stuttering
glitch microsample vocals.wav --slice 30 --shuffle 0.8 --stutter 0.6 --max-repeats 8 -o destroyed.wav

# Transient-based slicing (cuts at drum hits)
glitch microsample breakbeat.wav --mode transients --shuffle 0.6 --stutter 0.3 -o glitched.wav

# Random slice lengths
glitch microsample texture.wav --mode random --slice-min 20 --slice-max 100 --shuffle 0.5 -o varied.wav

# Export the cut list for reuse in Phase 2 (video sync)
glitch microsample drums.wav --slice 40 --shuffle 0.6 -o out.wav --cutlist cuts.json
```

**Key parameters:**
- `--slice N` — fixed slice length in milliseconds (default: 50)
- `--slice-min` / `--slice-max` — random slice length range (use with `--mode random`)
- `--mode fixed|transients|random` — how to determine cut points (default: transients)
- `--shuffle 0.0-1.0` — probability of moving a slice to a random position
- `--stutter 0.0-1.0` — probability of repeating a slice
- `--max-repeats N` — maximum stutter repetitions (default: 4)
- `--reverse 0.0-1.0` — probability of reversing a slice
- `--drop 0.0-1.0` — probability of dropping a slice (silence)
- `--cutlist path.json` — export cut list for Phase 2 video sync

### `glitch chop` — Chop & Shuffle

Convenience shortcut: evenly slices the audio into N pieces and shuffles them like a deck of cards. Some slices can be reversed or dropped.

```bash
glitch chop breakbeat.wav -slices 16 -o shuffled.wav
glitch chop vocals.wav -slices 32 --reverse 0.3 --drop 0.2 -o chopped.wav
```

### `glitch apply` — Apply Cut Lists

Apply a previously saved cut list to different audio. The same sequence of cuts (which slices, how many repeats, reversed or not) gets applied to new material.

```bash
glitch apply cuts.json different_drums.wav -o recut.wav
```

### `glitch chain` — Chain Multiple Passes

Apply multiple micro-sampling passes in one command. Each `--op` starts a new pass with its own parameters.

```bash
glitch chain input.wav \
  --op microsample --slice 50 --shuffle 0.4 \
  --op microsample --stutter 0.6 --max-repeats 8 \
  -o out.wav
```

### `glitch spectral` — Spectral Smear

FFT-based processing. Randomizes phase, blurs the frequency spectrum, freezes frames. Produces metallic, underwater, ghostly textures.

```bash
# Ghostly wash
glitch spectral piano.wav -phase 0.8 -blur 0.5 -o ghost.wav

# Spectral freeze (frames repeat)
glitch spectral drums.wav -freeze 0.5 -phase 0.3 -o frozen.wav

# Full underwater
glitch spectral vocals.wav -phase 1.0 -blur 0.8 -freeze 0.3 -o underwater.wav
```

**Key parameters:**
- `-phase 0.0-1.0` — how much to randomize the phase spectrum (0 = original, 1 = fully random)
- `-blur 0.0-1.0` — how much to smooth the magnitude spectrum
- `-freeze 0.0-1.0` — probability of reusing the previous frame's spectrum

### `glitch mangle` — Spectral Destruction

More aggressive than `spectral`. Swaps frequency bins between frames, zeros out random bands, multiplies by noise.

```bash
glitch mangle synth.wav -swap 0.5 -zero 0.3 -noise 0.8 -o mangled.wav
```

### `glitch quantize` — BPM Detection & Time-Stretch

```bash
# Just detect BPM
glitch quantize breakbeat.wav --detect-only -o dummy.wav

# Stretch to target BPM
glitch quantize breakbeat.wav -bpm 140 -o at_140.wav

# Specify source BPM (skip auto-detection)
glitch quantize sample.wav -bpm 160 -source-bpm 120 -o faster.wav
```

### `glitch sequence` — Probabilistic Arranger

Takes multiple audio files and arranges them on a timeline with probabilistic placement. Creates sparse, textural compositions with intentional silence.

```bash
glitch sequence kick.wav snare.wav hat.wav texture.wav \
  -duration 60 -density 0.5 -cluster 0.3 -layers 3 -o arrangement.wav
```

**Key parameters:**
- `-duration N` — output length in seconds
- `-density 0.0-1.0` — how much of the timeline is filled
- `-cluster 0.0-1.0` — tendency for events to bunch together
- `-layers N` — max simultaneous overlapping stems

### `glitch pattern` — Step Sequencer

Probabilistic step sequencer. Assigns stems to a grid and triggers them by probability.

```bash
glitch pattern kick.wav snare.wav hat.wav \
  -bpm 140 -steps 16 -bars 8 -swing 0.2 -o beat.wav
```

### `glitch mix` — Mix Stems

Simple: sum multiple audio files together and normalize.

```bash
glitch mix drums.wav bass.wav melody.wav -o mixdown.wav
```

---

## Chaining

Every audio command takes a file in and writes a file out. Chain them:

```bash
# Step 1: Micro-sample a breakbeat
glitch microsample break.wav --slice 40 --shuffle 0.9 --stutter 0.4 --seed 42 -o step1.wav

# Step 2: Spectral smear the result
glitch spectral step1.wav -blur 0.6 -phase 0.9 -freeze 0.3 -o step2.wav

# Step 3: Micro-sample again (double-glitch)
glitch microsample step2.wav --slice 20 --stutter 0.7 --max-repeats 8 -o final.wav
```

Or use the `chain` command for multiple passes in one go:

```bash
glitch chain break.wav \
  --op microsample --slice 40 --shuffle 0.9 --stutter 0.4 \
  --op microsample --slice 20 --stutter 0.7 --max-repeats 8 \
  -o final.wav
```

Or use the Python API for tighter control:

```python
from glitch import load, save, normalize, microsample
from glitch.spectral import smear

audio, sr = load("break.wav")
audio, cl1 = microsample(audio, sr, slice_ms=40, shuffle_chance=0.9, stutter_chance=0.4, seed=42)
audio = smear(audio, sr, blur=0.6, phase_randomize=0.9, freeze_chance=0.3)
audio, cl2 = microsample(audio, sr, slice_ms=20, stutter_chance=0.7, max_repeats=8)
save("final.wav", normalize(audio), sr)
```

---

## Visual Commands

### The Concept: Bond → Destroy

Every visual starts by being **bonded** to audio. After bonding, audio and video are a single unit. All processing applies identically to both.

Three bonding modes (auto-detected):
1. **Still image** → the image is held for the entire audio duration
2. **Image sequence** (a folder of images) → images switch at musically meaningful transient points
3. **Video clip** → trimmed/looped to match the audio length

### `glitch render` — Bond + Render

The simplest AV command. Bond audio to a visual and render to .mp4.

```bash
# Still image (image held for full duration)
glitch render sample.wav photo.png -o output.mp4

# Image sequence (images switch at transients)
glitch render sample.wav frames_folder/ -o output.mp4

# Video clip (trimmed/looped to match audio)
glitch render sample.wav clip.mp4 -o output.mp4

# Custom resolution and framerate
glitch render sample.wav photo.png --resolution 1280x720 --fps 24 -o output.mp4

# Quantize audio to target BPM before rendering
glitch render sample.wav photo.png --bpm 140 -o output.mp4
```

### `glitch render` with Micro-Sampling

Add `--slice`, `--shuffle`, `--stutter`, `--reverse`, `--drop` flags to micro-sample the bonded clip. Both audio and video are cut identically.

```bash
# Mild glitch
glitch render sample.wav frames/ -o output.mp4 \
  --slice 100 --shuffle 0.3 --stutter 0.2

# Heavy destruction
glitch render sample.wav frames/ -o output.mp4 \
  --slice 30 --shuffle 0.8 --stutter 0.5 --reverse 0.4 --drop 0.1

# With visual effects (rgb split, scan lines, etc.)
glitch render sample.wav photo.png -o output.mp4 \
  --slice 50 --shuffle 0.5 --effects rgb_split scan_lines --effect-chance 0.4

# Transient-based slicing (cuts at drum hits)
glitch render drums.wav frames/ -o output.mp4 \
  --mode transients --shuffle 0.6 --stutter 0.3

# Random slice lengths
glitch render texture.wav frames/ -o output.mp4 \
  --mode random --slice-min 20 --slice-max 150 --shuffle 0.5
```

**Micro-sample parameters:**
- `--slice N` — fixed slice length in milliseconds
- `--slice-min` / `--slice-max` — random slice length range (use with `--mode random`)
- `--mode fixed|transients|random` — how to determine cut points (default: transients)
- `--shuffle 0.0-1.0` — probability of moving a slice to a random position
- `--stutter 0.0-1.0` — probability of repeating a slice
- `--max-repeats N` — maximum repetitions
- `--reverse 0.0-1.0` — probability of reversing a slice
- `--drop 0.0-1.0` — probability of replacing a slice with silence + black
- `--effects [names]` — visual effects: `rgb_split`, `scan_lines`, `corrupt`, `invert_region`, `posterize`, `noise`
- `--effect-chance 0.0-1.0` — probability of applying effects to each frame

### `glitch compose` — Multi-Clip Composition

Compose a full track from multiple audio+visual pairs using a manifest or folder.

**From a folder:**

Set up your folder like this:
```
my_track/
  kick/
    sample.wav
    visual.png
  break/
    sample.wav
    001.png
    002.png
    003.png
  texture/
    sample.wav
    visual.mp4
```

Then:
```bash
# Auto-quantizes all clips to the dominant BPM
glitch compose my_track/ -o track.mp4

# Force all clips to 140 BPM
glitch compose my_track/ -o track.mp4 --bpm 140

# With global micro-sampling applied to everything
glitch compose my_track/ -o track.mp4 --shuffle 0.4 --stutter 0.2

# Preview first 10 seconds
glitch compose my_track/ -o preview.mp4 --preview 10
```

**From a JSON manifest** (more control):

```bash
# Generate a manifest template from your folder
glitch manifest my_track/ -o manifest.json

# Edit manifest.json to set offsets, gains, per-clip microsample settings
# Then render:
glitch compose manifest.json -o track.mp4
```

Manifest format:
```json
{
  "title": "my_track",
  "resolution": [1920, 1080],
  "fps": 30,
  "bpm": 140,
  "pairs": [
    {
      "audio": "kick/sample.wav",
      "visual": "kick/visual.png",
      "offset": 0.0,
      "gain_db": 0.0,
      "bpm": null,
      "microsample": null
    },
    {
      "audio": "break/sample.wav",
      "visual": "break/",
      "offset": 2.5,
      "bpm": 160,
      "microsample": {
        "slice_ms": 40,
        "shuffle_chance": 0.6,
        "stutter_chance": 0.3,
        "max_repeats": 4,
        "reverse_chance": 0.2,
        "seed": 42
      }
    }
  ]
}
```

- `bpm` (top-level) — quantize all clips to this BPM (auto-detected if omitted)
- `bpm` (per-pair) — override target BPM for this specific clip
- `offset` — when this clip starts in seconds
- `gain_db` — volume adjustment in dB (0 = no change)
- `microsample` — per-clip micro-sample settings (null = no processing)

BPM priority: per-pair `bpm` > manifest-level `bpm` > CLI `--bpm` > auto-detect

### `glitch cutlist` — Export Cut Lists

Generate a cut list without rendering. Useful for finding a good set of cuts, saving them, and reusing them.

```bash
glitch cutlist drums.wav frames/ --slice 30 --shuffle 0.5 --seed 42 -o cuts.json
```

### `glitch apply` — Reuse Cut Lists

Apply a saved cut list to new material:

```bash
glitch apply cuts.json different_drums.wav different_frames/ -o output.mp4
```

This applies the exact same sequence of cuts (which slices, how many repeats, reversed or not) but to completely different source material.

---

## Reproducibility

Every command accepts `--seed N` (or `-seed N` for audio commands). Same seed = same result. Omit the seed for a random result each time.

```bash
# These two produce identical output:
glitch microsample drums.wav --slice 40 --shuffle 0.5 --seed 42 -o a.wav
glitch microsample drums.wav --slice 40 --shuffle 0.5 --seed 42 -o b.wav

# This produces something different:
glitch microsample drums.wav --slice 40 --shuffle 0.5 --seed 99 -o c.wav

# This is random every time:
glitch microsample drums.wav --slice 40 --shuffle 0.5 -o random.wav
```

---

## Complete Workflow Example

Process audio first, then bond with visuals:

```bash
# 1. Process the audio (Phase 1) — micro-sample and save cut list
glitch microsample raw_break.wav --slice 40 --shuffle 0.9 --stutter 0.4 --seed 42 \
  -o processed.wav --cutlist cuts.json

# 2. Spectral processing (audio-only, no video equivalent)
glitch spectral processed.wav -blur 0.6 -phase 0.8 --seed 42 -o smeared.wav

# 3. Bond with visuals and micro-sample again (Phase 2)
glitch render smeared.wav my_frames/ -o video.mp4 \
  --slice 40 --shuffle 0.5 --stutter 0.3 --reverse 0.2 --seed 42 \
  --effects rgb_split corrupt --effect-chance 0.3

# Double-glitch: the audio was micro-sampled in Phase 1, then the
# combined audiovisual gets micro-sampled again in Phase 2
```

---

## Visual Effects Reference

These are applied per-frame during micro-sampling when you use `--effects`:

| Effect | What it does |
|--------|-------------|
| `rgb_split` | Shifts R, G, B channels by random pixel offsets (chromatic aberration) |
| `scan_lines` | Darkens alternating horizontal rows (CRT look) |
| `corrupt` | Zeros out random rectangular blocks (data corruption) |
| `invert_region` | Color-inverts random rectangles |
| `posterize` | Reduces color depth (flat, graphic look) |
| `noise` | Adds random noise to pixel values |

Combine multiple: `--effects rgb_split scan_lines corrupt`

---

## Python API Quick Reference

```python
# Audio micro-sampling
from glitch import load, save, normalize, microsample, chop, chain
from glitch.cutlist import Cut, CutList, apply_cut_list
from glitch.spectral import smear, mangle
from glitch.quantize import detect_bpm, quantize
from glitch.sequence import arrange, pattern
from glitch.mix import layer, mixdown

# AV
from glitch.av import bond, microsample as av_microsample, apply_cut_list as av_apply, chain as av_chain, render, composite
from glitch.av.core import AVClip
from glitch.av.effects import rgb_split, scan_lines, corrupt, posterize, noise
```

Audio micro-sampling signature: `microsample(audio, sr, **params) -> (audio_array, CutList)`
Spectral functions return only audio (no cut list): `smear(audio, sr, **params) -> audio_array`
AV functions work with `AVClip` objects: `microsample(clip, **params) -> (AVClip, CutList)`
