# glitch-engine User Guide

## What Is This?

A command-line tool for making glitchy electronic music and visuals. You give it audio files and images/video, it chops them up, rearranges the pieces, and spits out glitched results. Every run is different — it's a generative instrument, not a precise editor.

## What Does It Actually Do?

The main thing: **micro-sampling**. It slices your audio into tiny pieces (by default, it cuts at the drum hits / transients), then randomly:
- **Shuffles** them (moves pieces to different positions)
- **Stutters** them (repeats a piece 2-8 times, like a CD skip)
- **Reverses** some pieces
- **Drops** some pieces (replaces with silence)

If you give it video or images too, the video gets cut at the exact same points as the audio. Audio and video always stay in sync.

## What Should I Feed It?

**Audio loops work best** for the compose workflow. When you're building a track from multiple clips, the tool detects each loop's BPM and time-stretches them to match. This only works if the audio has a detectable tempo — so loops and phrases, not one-shot hits.

You can still micro-sample a one-shot hit (it'll slice and stutter it), but the auto-quantize BPM matching won't do anything useful with a single kick drum sample.

**Think of it like this:** each clip you give it should be a musical phrase or loop. A 4-bar drum pattern. An 8-bar synth line. A vocal phrase. The tool sequences these one after another and time-stretches them to a common BPM.

If you want a 4/4 kick pattern, you'd use the **step sequencer** to build one first:

```bash
# Build a kick pattern: 140 BPM, 16 steps per bar, 8 bars long
glitch pattern kick.wav -bpm 140 -steps 16 -bars 8 -o kick_pattern.wav
```

Then use that `kick_pattern.wav` as one of your clips in the compose workflow.

---

## Setup

```bash
cd glitch-engine
pip install -e .
```

You now have a `glitch` command. You also need **ffmpeg** installed for video rendering.

---

## The Full Workflow: Making a Track

Here's what an actual session looks like, start to finish.

### 1. Gather your raw materials

You need audio files. These can be:
- Drum loops you recorded or sampled
- Synth loops from a DAW export
- Field recordings, vocal snippets, whatever

If you want video output, you also need visuals — images, image sequences, or video clips.

### 2. (Optional) Process individual sounds first

Before composing, you might want to glitch individual sounds:

```bash
# Chop up a breakbeat
glitch microsample amen_break.wav --shuffle 0.8 --stutter 0.4 -o chopped_break.wav

# Make a ghostly pad from a vocal sample
glitch spectral vocal.wav -blur 0.7 -phase 0.9 -o ghost_vocal.wav

# Build a kick pattern from a single kick hit
glitch pattern kick.wav -bpm 140 -steps 16 -bars 8 -o kick_loop.wav

# Build a full beat from individual hits
glitch pattern kick.wav snare.wav hat.wav -bpm 140 -steps 16 -bars 8 -o beat.wav
```

### 3. Set up your track folder

Create a folder. Inside it, make one subfolder per clip. Each subfolder needs:
- An audio file named `sample.wav` (or `.flac`, `.mp3`, `.ogg`)
- A visual file (optional, for video output)

The subfolders play in alphabetical order — name them `01_kick`, `02_break`, etc. to control the sequence.

```
my_track/
  01_kick/
    sample.wav              <-- your kick loop or pattern
    visual.png              <-- a still image (must be named "visual")
  02_break/
    sample.wav              <-- a chopped breakbeat
    001.png                 <-- numbered images switch at drum hits
    002.png
    003.png
  03_texture/
    sample.wav              <-- a synth texture or pad
    visual.mp4              <-- a video clip (must be named "visual")
```

**Naming rules:**
- Audio MUST be called `sample` (with any common audio extension)
- For a still image or video: name it `visual.png`, `visual.jpg`, `visual.mp4`, or `visual.mov`
- For an image sequence: use numbered filenames like `001.png`, `002.png` (do NOT name them "visual")
- Subfolders without `sample.*` get skipped

### 4. Compose

```bash
glitch compose my_track/ -o track.mp4
```

What happens:
1. Each subfolder becomes one clip
2. The tool detects each clip's BPM
3. All clips get time-stretched to match the dominant BPM (your source files are NOT modified — this happens in memory only)
4. Clips play one after another in alphabetical subfolder order
5. Everything renders to a single .mp4

**Options:**

```bash
# Force all clips to 140 BPM instead of auto-detecting
glitch compose my_track/ -o track.mp4 --bpm 140

# Apply micro-sampling to everything during composition
glitch compose my_track/ -o track.mp4 --shuffle 0.4 --stutter 0.2

# Preview just the first 10 seconds (fast iteration)
glitch compose my_track/ -o preview.mp4 --preview 10
```

### 5. (Optional) Fine-tune with a manifest

For more control (overlapping clips, volume adjustments, per-clip glitch settings):

```bash
# Generate a starting manifest from your folder
glitch manifest my_track/ -o manifest.json
```

Open `manifest.json` in a text editor. It looks like this:

```json
{
  "title": "my_track",
  "resolution": [1920, 1080],
  "fps": 30,
  "bpm": 140,
  "pairs": [
    {
      "audio": "01_kick/sample.wav",
      "visual": "01_kick/visual.png",
      "offset": 0.0,
      "gain_db": 0.0,
      "bpm": null,
      "microsample": null
    },
    {
      "audio": "02_break/sample.wav",
      "visual": "02_break/",
      "offset": 4.0,
      "gain_db": -3.0,
      "microsample": {
        "slice_ms": 40,
        "shuffle_chance": 0.6,
        "stutter_chance": 0.3
      }
    }
  ]
}
```

What you can tweak:
- **`offset`** — when this clip starts (in seconds). Set different offsets to overlap clips.
- **`gain_db`** — volume adjustment. 0 = unchanged, -6 = half volume, +6 = double
- **`bpm`** (top-level) — target BPM for all clips. Remove it for auto-detection.
- **`bpm`** (per-pair) — override BPM for one specific clip
- **`microsample`** — per-clip glitch settings (null = no glitching)

Then compose from the manifest:

```bash
glitch compose manifest.json -o track.mp4
```

---

## Audio Commands Reference

### `glitch microsample` — The Main Tool

Slices audio at transients (drum hits) and rearranges the pieces.

```bash
# Basic: shuffle pieces around
glitch microsample drums.wav --shuffle 0.6 -o out.wav

# Heavy: lots of stuttering, some reversal
glitch microsample drums.wav --shuffle 0.8 --stutter 0.5 --reverse 0.3 -o destroyed.wav

# Fixed-size slices instead of transient-based
glitch microsample texture.wav --mode fixed --slice 40 --shuffle 0.5 -o out.wav

# Random slice sizes
glitch microsample texture.wav --mode random --slice-min 20 --slice-max 100 --shuffle 0.5 -o out.wav

# Save the cut list for later reuse
glitch microsample drums.wav --shuffle 0.6 -o out.wav --cutlist cuts.json
```

**Parameters:**
- `--mode` — where to cut: `transients` (default, cuts at drum hits), `fixed` (even intervals), `random` (variable lengths)
- `--slice N` — slice length in ms (used with `fixed` mode, default: 50)
- `--slice-min` / `--slice-max` — range for `random` mode
- `--shuffle 0.0-1.0` — chance of moving a piece to a random position (0 = in order, 1 = fully scrambled)
- `--stutter 0.0-1.0` — chance of repeating a piece
- `--max-repeats N` — max stutter repetitions (default: 4)
- `--reverse 0.0-1.0` — chance of playing a piece backwards
- `--drop 0.0-1.0` — chance of replacing a piece with silence
- `--cutlist path.json` — save the cut decisions to a file

### `glitch chop` — Quick Shuffle

Chops audio into N equal pieces and shuffles them like a deck of cards.

```bash
glitch chop breakbeat.wav --slices 16 -o shuffled.wav
glitch chop vocals.wav --slices 32 --reverse 0.3 --drop 0.2 -o chopped.wav
```

### `glitch spectral` — Ghostly Textures

FFT-based processing. Makes things sound metallic, underwater, ghostly.

```bash
glitch spectral piano.wav -phase 0.8 -blur 0.5 -o ghost.wav
glitch spectral vocals.wav -phase 1.0 -blur 0.8 -freeze 0.3 -o underwater.wav
```

- `-phase 0.0-1.0` — randomize phase (0 = original, 1 = fully smeared)
- `-blur 0.0-1.0` — smooth the frequency spectrum
- `-freeze 0.0-1.0` — chance of reusing the previous frame's spectrum

### `glitch mangle` — Heavy Destruction

More aggressive than spectral. Swaps frequencies, zeros out bands, multiplies by noise.

```bash
glitch mangle synth.wav -swap 0.5 -zero 0.3 -noise 0.8 -o mangled.wav
```

### `glitch quantize` — BPM Detection & Time-Stretch

```bash
glitch quantize breakbeat.wav --detect-only -o dummy.wav    # just print the BPM
glitch quantize breakbeat.wav -bpm 140 -o stretched.wav      # stretch to 140 BPM
```

### `glitch pattern` — Step Sequencer

Build beat patterns from one-shot samples. This is how you turn individual hits into loops.

```bash
# Basic 4/4 kick
glitch pattern kick.wav -bpm 140 -steps 16 -bars 8 -o kick_loop.wav

# Full beat with multiple sounds
glitch pattern kick.wav snare.wav hat.wav -bpm 140 -steps 16 -bars 8 -swing 0.2 -o beat.wav
```

### `glitch sequence` — Probabilistic Arranger

Scatters clips across a timeline with intentional gaps and clusters.

```bash
glitch sequence texture1.wav texture2.wav pad.wav -duration 60 -density 0.5 -o arrangement.wav
```

### `glitch mix` — Mix Stems

Sums audio files together and normalizes.

```bash
glitch mix drums.wav bass.wav melody.wav -o mixdown.wav
```

### `glitch chain` — Multiple Passes

Apply multiple micro-sampling passes in one command:

```bash
glitch chain input.wav \
  --op microsample --slice 50 --shuffle 0.4 \
  --op microsample --stutter 0.6 --max-repeats 8 \
  -o out.wav
```

### `glitch apply` — Reuse Cut Lists

Apply a saved cut list to different audio (or audio+video):

```bash
glitch apply cuts.json different_drums.wav -o recut.wav
glitch apply cuts.json drums.wav frames/ -o output.mp4
```

---

## Visual Commands

### How Bonding Works

When you give the tool both audio and a visual, it **bonds** them into a single unit. After that, all processing (slicing, shuffling, stuttering) applies to both in sync.

The visual can be:
1. **A still image** (`visual.png`) — the image stays on screen for the full audio duration
2. **A folder of numbered images** (`001.png`, `002.png`, ...) — images switch at audio transients (drum hits)
3. **A video clip** (`visual.mp4`) — trimmed or looped to match the audio length

### `glitch render` — Make One Video

Bond one audio file to one visual and render:

```bash
glitch render loop.wav photo.png -o output.mp4
glitch render loop.wav frames/ -o output.mp4
glitch render loop.wav footage.mp4 -o output.mp4
```

Add glitch processing (audio and video get cut in sync):

```bash
glitch render loop.wav frames/ -o output.mp4 \
  --shuffle 0.6 --stutter 0.3 --reverse 0.2

# With visual effects too
glitch render loop.wav photo.png -o output.mp4 \
  --shuffle 0.5 --effects rgb_split scan_lines --effect-chance 0.4
```

### Visual Effects

Applied per-frame during micro-sampling with `--effects`:

| Effect | What it does |
|--------|-------------|
| `rgb_split` | Shifts color channels apart (chromatic aberration) |
| `scan_lines` | CRT-style horizontal lines |
| `corrupt` | Random black rectangles (data corruption look) |
| `invert_region` | Color-inverts random areas |
| `posterize` | Reduces color depth (flat, graphic look) |
| `noise` | Adds pixel noise |

Combine multiple: `--effects rgb_split scan_lines corrupt`

---

## Reproducibility

Every command accepts `--seed N`. Same seed = identical output. Leave it off for random results.

```bash
glitch microsample drums.wav --shuffle 0.5 --seed 42 -o a.wav   # always the same
glitch microsample drums.wav --shuffle 0.5 -o b.wav              # different every time
```

---

## Auto-Quantize (BPM Matching)

When you use `glitch compose`, the tool automatically time-stretches all your loops to a common BPM. **Your original files are never modified** — stretching happens in memory only.

How it picks the target BPM:
1. If you pass `--bpm 140`, all clips stretch to 140
2. If the manifest has a top-level `"bpm"` field, that's used
3. If neither, it detects every clip's BPM and uses the median

Per-clip `"bpm"` in the manifest overrides everything for that specific clip.

---

## Python API

```python
import glitch
from glitch.av import bond, microsample, render

# Audio
audio, sr = glitch.load("drums.wav")
processed, cut_list = glitch.microsample(audio, sr, shuffle_chance=0.6, seed=42)
glitch.save("glitched.wav", processed, sr)

# AV
clip = bond("drums.wav", "frames/", resolution=(1920, 1080), fps=30)
result, cut_list = microsample(clip, shuffle_chance=0.5, stutter_chance=0.3)
render(result, "output.mp4")
```
