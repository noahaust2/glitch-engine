# glitch-engine

A command-line toolkit for making glitchy IDM music and visuals. Think Arca, Aphex Twin, Autechre.

No DAWs, no GUIs — you run commands that take audio (and optionally images/video) in and spit glitched output out. Everything is randomized: run the same command twice and you get two different results.

## How It Works (The Big Idea)

The core concept is **micro-sampling**: your audio gets sliced into tiny pieces (at drum hits by default), then those pieces get shuffled around, repeated, reversed, or dropped. It's like cutting up a tape with scissors and rearranging the pieces.

The key thing that makes this different from granular synthesis: every cut is clean and discrete. No crossfading, no overlapping. This means the exact same cuts can be applied to video too — audio and video get chopped in perfect sync.

## Installation

```bash
cd glitch-engine
pip install -e .
```

You need Python 3.10+ and **ffmpeg** installed on your system (for video rendering).

## Quick Start

```bash
# Take a drum loop, slice it at the transients (drum hits), shuffle the pieces around
glitch microsample drums.wav --shuffle 0.6 -o glitched.wav

# Same thing but also stutter (repeat) some hits and reverse others
glitch microsample drums.wav --shuffle 0.6 --stutter 0.3 --reverse 0.2 -o glitched.wav
```

That's it. You now have a glitched drum loop.

## What Can I Feed It?

**Audio loops work best.** The auto-quantize feature (BPM matching) detects tempo and time-stretches loops so they play in sync — this only works with loops that have a detectable BPM. One-shot hits (a single kick, a single snare) don't have a tempo, so BPM detection won't do anything useful with them.

That said, you can absolutely micro-sample one-shot hits — the slicing/shuffling/stuttering works on any audio. It just won't auto-quantize them.

For the `compose` workflow (building a full track from multiple clips), **think of each clip as a loop or a musical phrase**. The tool plays them one after another, time-stretched to a common BPM.

## Making a Track (The Main Workflow)

This is the end-to-end workflow for composing a full audiovisual track.

### Step 1: Set up your folder

Create a folder for your track. Inside it, make a subfolder for each clip (each musical element). The subfolders are played in alphabetical order, one after another.

```
my_track/
  01_intro/
    sample.wav        <-- your audio loop (MUST be named "sample")
    visual.png        <-- a still image (MUST be named "visual")
  02_drums/
    sample.wav
    visual.mp4        <-- or a video clip (named "visual")
  03_break/
    sample.wav
    001.png           <-- or numbered images (NOT named "visual")
    002.png           <-- these switch at drum hits automatically
    003.png
  04_outro/
    sample.wav
    visual.jpg
```

**The naming matters:**
- Audio: must be called `sample.wav` (or `sample.flac`, `sample.mp3`, `sample.ogg`)
- Visual, option A: a single file called `visual.png` / `visual.jpg` / `visual.mp4` / `visual.mov`
- Visual, option B: numbered images like `001.png`, `002.png` — these get treated as a sequence that switches at audio transients (drum hits)

If a subfolder is missing `sample.wav`, it gets skipped.

### Step 2: Compose

```bash
glitch compose my_track/ -o track.mp4
```

This will:
1. Load each subfolder as a clip
2. Auto-detect the BPM of every loop
3. Time-stretch them all to match (so they play in sync)
4. Play them one after another
5. Render to a video file

Want to force a specific BPM instead of auto-detecting?

```bash
glitch compose my_track/ -o track.mp4 --bpm 140
```

Want to glitch everything while composing?

```bash
glitch compose my_track/ -o track.mp4 --shuffle 0.4 --stutter 0.2
```

Preview just the first 10 seconds:

```bash
glitch compose my_track/ -o preview.mp4 --preview 10
```

### Step 3 (Optional): Fine-tune with a manifest

If you want more control (custom offsets, per-clip settings, volume adjustments), generate a manifest file:

```bash
glitch manifest my_track/ -o manifest.json
```

Edit `manifest.json` in a text editor, then compose from it:

```bash
glitch compose manifest.json -o track.mp4
```

The manifest lets you set:
- `offset` — when each clip starts (in seconds), so clips can overlap
- `gain_db` — volume per clip
- `bpm` — per-clip BPM override
- `microsample` — per-clip glitch settings

## Individual Commands

### Micro-sample (the main tool)

```bash
# Slice at transients, shuffle pieces around
glitch microsample drums.wav --shuffle 0.6 -o out.wav

# Go harder: more shuffle, stutter some hits, reverse others
glitch microsample drums.wav --shuffle 0.8 --stutter 0.5 --reverse 0.3 -o destroyed.wav

# Use fixed-size slices instead of transients
glitch microsample texture.wav --mode fixed --slice 40 --shuffle 0.5 -o out.wav

# Random slice sizes
glitch microsample texture.wav --mode random --slice-min 20 --slice-max 100 --shuffle 0.5 -o out.wav

# Save the cut list (so you can apply the same cuts to video later)
glitch microsample drums.wav --shuffle 0.6 -o out.wav --cutlist cuts.json
```

### Render (one audio + one visual = one video)

```bash
# Audio + still image
glitch render loop.wav photo.png -o output.mp4

# Audio + image sequence (images switch at drum hits)
glitch render loop.wav frames/ -o output.mp4

# Audio + video clip
glitch render loop.wav footage.mp4 -o output.mp4

# Render with glitch processing (audio and video get cut in sync)
glitch render loop.wav frames/ -o output.mp4 --shuffle 0.6 --stutter 0.3
```

### Other audio tools

```bash
# Chop into 16 equal pieces and shuffle like a deck of cards
glitch chop drums.wav --slices 16 -o shuffled.wav

# Spectral smear (ghostly, underwater textures)
glitch spectral vocals.wav -blur 0.5 -phase 0.8 -o ghost.wav

# Spectral destruction
glitch mangle synth.wav -swap 0.5 -noise 0.8 -o mangled.wav

# Detect BPM
glitch quantize loop.wav --detect-only -o dummy.wav

# Time-stretch to target BPM
glitch quantize loop.wav -bpm 140 -o stretched.wav

# Probabilistic arrangement (scatter clips across a timeline)
glitch sequence stem1.wav stem2.wav -duration 60 -density 0.5 -o arranged.wav

# Step sequencer
glitch pattern kick.wav snare.wav hat.wav -bpm 140 -steps 16 -bars 8 -o beat.wav

# Mix stems together
glitch mix drums.wav bass.wav melody.wav -o mixed.wav

# Chain multiple glitch passes
glitch chain input.wav \
  --op microsample --slice 50 --shuffle 0.4 \
  --op microsample --stutter 0.6 --max-repeats 8 \
  -o out.wav
```

## Reproducibility

Every command accepts `--seed N`. Same seed = same result. Leave it off for random results each time.

```bash
glitch microsample drums.wav --shuffle 0.5 --seed 42 -o a.wav   # always the same
glitch microsample drums.wav --shuffle 0.5 -o b.wav              # different every time
```

## Auto-Quantize (BPM Matching)

When you use `glitch compose`, the tool automatically time-stretches all your loops to a common BPM so they play in sync. **This only happens in memory — your original files are never modified.**

How it decides the target BPM:
1. If you pass `--bpm 140`, everything gets stretched to 140
2. If the manifest has a `"bpm"` field, that's used
3. If neither, it detects the BPM of every clip and uses the median

This only makes sense with loops. One-shot samples don't have a detectable BPM.

## Python API

```python
import glitch
from glitch.av import bond, microsample, render

# Audio micro-sampling
audio, sr = glitch.load("drums.wav")
processed, cut_list = glitch.microsample(audio, sr, shuffle_chance=0.6, seed=42)
glitch.save("glitched.wav", processed, sr)

# AV: bond audio to visual, micro-sample both, render
clip = bond("drums.wav", "frames/", resolution=(1920, 1080), fps=30)
result, cut_list = microsample(clip, shuffle_chance=0.5, stutter_chance=0.3)
render(result, "output.mp4")
```

## Modules

| Module | What it does |
|--------|-------------|
| `core` | Load/save audio, normalize, fade |
| `cutlist` | Cut list data model (shared between audio and video) |
| `microsample` | Slice and rearrange audio |
| `spectral` | FFT-based spectral processing |
| `quantize` | BPM detection and time-stretching |
| `sequence` | Probabilistic arrangement |
| `mix` | Stem mixing |
| `av.core` | AVClip data model |
| `av.bond` | Fuse audio + visual into one unit |
| `av.microsample` | Slice and rearrange audio+video in sync |
| `av.render` | Render to .mp4 |
| `av.effects` | Per-frame visual glitch effects (rgb split, scan lines, etc.) |
