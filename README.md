# glitch-engine

Procedural sampling & glitch toolkit for IDM music production. Think Arca, Aphex Twin, Autechre.

No DAWs, no GUIs — just scripts that take audio files in and spit processed audio out. Every function is probabilistic: re-running produces different results.

## Installation

```bash
cd glitch-engine
pip install -e .

# Optional: install pyrubberband for high-quality time-stretching
pip install pyrubberband
```

Requires Python 3.10+.

## Quick Start

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

Every command accepts `-seed N` for reproducibility.

## Chaining Effects

The real power comes from chaining. Each command takes audio in and spits audio out:

```bash
# Granular scatter → stutter → spectral smear
glitch granular break.wav -grains 300 -spread 0.9 -pitch 6 -o step1.wav
glitch stutter step1.wav -chance 0.7 -repeats 6 -o step2.wav
glitch spectral step2.wav -blur 0.6 -phase 0.9 -freeze 0.3 -o final.wav
```

## Python API

All functions are pure: audio array in, audio array out.

```python
import glitch

# Load
audio, sr = glitch.load("input.wav")

# Granular scatter
scattered = glitch.scatter(audio, sr, num_grains=200, spread=0.8, pitch_drift=4)

# Chain into stutter
stuttered = glitch.stutter_glitch(scattered, sr, stutter_chance=0.6, max_repeats=4)

# Spectral smear
smeared = glitch.smear(stuttered, sr, phase_randomize=0.8, blur=0.5)

# Save
glitch.save("output.wav", smeared, sr)
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

## Design Principles

1. **Pure functions** — audio array in, audio array out. No side effects.
2. **Sensible defaults** — every parameter has a default that sounds interesting.
3. **Composable** — chain any function's output into any other.
4. **Probabilistic** — different seed = different result, same aesthetic.
5. **Minimal dependencies** — NumPy, SciPy, librosa, soundfile. That's it.
