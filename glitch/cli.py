"""Unified CLI entry point for glitch-engine."""

import argparse
import sys
import numpy as np

from glitch.core import load, save, normalize


def _add_seed_arg(parser):
    parser.add_argument("-seed", "--seed", type=int, default=None,
                        help="Random seed for reproducibility")


def _add_output_arg(parser):
    parser.add_argument("-o", "--output", required=True,
                        help="Output file path")


def cmd_granular(args):
    """Run granular scatter on input audio."""
    from glitch.granular import scatter
    audio, sr = load(args.input)
    result = scatter(
        audio, sr,
        grain_min_ms=args.grain_min,
        grain_max_ms=args.grain_max,
        num_grains=args.grains,
        density=args.density,
        pitch_drift=args.pitch,
        spread=args.spread,
        envelope_shape=args.envelope,
        output_duration_s=args.duration,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Granular scatter -> {args.output}")


def cmd_cloud(args):
    """Run granular cloud on input audio."""
    from glitch.granular import cloud
    audio, sr = load(args.input)
    result = cloud(
        audio, sr,
        grain_min_ms=args.grain_min,
        grain_max_ms=args.grain_max,
        density=args.density,
        pitch_drift=args.pitch,
        spread=args.spread,
        output_duration_s=args.duration,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Granular cloud -> {args.output}")


def cmd_stutter(args):
    """Run stutter/glitch effect on input audio."""
    from glitch.stutter import glitch
    audio, sr = load(args.input)
    result = glitch(
        audio, sr,
        stutter_chance=args.chance,
        max_repeats=args.repeats,
        reverse_chance=args.reverse,
        crush_chance=args.crush,
        crush_bits=args.bits,
        preserve_length=not args.no_preserve,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Stutter glitch -> {args.output}")


def cmd_chop(args):
    """Run chop/shuffle on input audio."""
    from glitch.stutter import chop
    audio, sr = load(args.input)
    result = chop(
        audio, sr,
        slices=args.slices,
        reverse_chance=args.reverse,
        drop_chance=args.drop,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Chop -> {args.output}")


def cmd_spectral(args):
    """Run spectral smear on input audio."""
    from glitch.spectral import smear
    audio, sr = load(args.input)
    result = smear(
        audio, sr,
        phase_randomize=args.phase,
        blur=args.blur,
        freeze_chance=args.freeze,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Spectral smear -> {args.output}")


def cmd_mangle(args):
    """Run spectral mangle on input audio."""
    from glitch.spectral import mangle
    audio, sr = load(args.input)
    result = mangle(
        audio, sr,
        bin_swap_chance=args.swap,
        zero_band_chance=args.zero,
        noise_amount=args.noise,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Spectral mangle -> {args.output}")


def cmd_quantize(args):
    """Detect BPM and time-stretch to target BPM."""
    from glitch.quantize import quantize, detect_bpm
    audio, sr = load(args.input)

    if args.detect_only:
        bpm = detect_bpm(audio, sr)
        print(f"Detected BPM: {bpm:.1f}")
        return

    result = quantize(audio, sr, target_bpm=args.bpm, source_bpm=args.source_bpm)
    save(args.output, normalize(result), sr)
    print(f"Quantized to {args.bpm} BPM -> {args.output}")


def cmd_sequence(args):
    """Arrange multiple stems probabilistically."""
    from glitch.sequence import arrange
    stems = []
    sr = None
    for path in args.inputs:
        audio, file_sr = load(path)
        if sr is None:
            sr = file_sr
        stems.append(audio)

    result = arrange(
        stems, sr,
        duration_s=args.duration,
        density=args.density,
        cluster_factor=args.cluster,
        max_layers=args.layers,
        fade_ms=args.fade,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Sequence arranged -> {args.output}")


def cmd_pattern(args):
    """Run probabilistic step sequencer."""
    from glitch.sequence import pattern
    stems = []
    sr = None
    for path in args.inputs:
        audio, file_sr = load(path)
        if sr is None:
            sr = file_sr
        stems.append(audio)

    result = pattern(
        stems, sr,
        bpm=args.bpm,
        steps=args.steps,
        bars=args.bars,
        swing=args.swing,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    print(f"Pattern sequenced -> {args.output}")


def cmd_mix(args):
    """Mix multiple stems together."""
    from glitch.mix import mixdown
    stems = []
    sr = None
    for path in args.inputs:
        audio, file_sr = load(path)
        if sr is None:
            sr = file_sr
        stems.append(audio)

    result = mixdown(stems, sr, do_normalize=not args.no_normalize)
    save(args.output, result, sr)
    print(f"Mixed -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glitch",
        description="Procedural sampling & glitch engine for IDM production",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- granular ---
    p = subparsers.add_parser("granular", help="Granular scatter synthesis")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-grains", "--grains", type=int, default=None)
    p.add_argument("-density", "--density", type=float, default=None)
    p.add_argument("-grain-min", "--grain-min", type=float, default=10.0)
    p.add_argument("-grain-max", "--grain-max", type=float, default=80.0)
    p.add_argument("-pitch", "--pitch", type=float, default=0.0,
                    help="Max pitch drift in semitones")
    p.add_argument("-spread", "--spread", type=float, default=0.5)
    p.add_argument("-envelope", "--envelope", default="hann",
                    choices=["hann", "triangle", "tukey", "random"])
    p.add_argument("-duration", "--duration", type=float, default=None,
                    help="Output duration in seconds")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_granular)

    # --- cloud ---
    p = subparsers.add_parser("cloud", help="Dense granular cloud texture")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-density", "--density", type=float, default=500.0)
    p.add_argument("-grain-min", "--grain-min", type=float, default=1.0)
    p.add_argument("-grain-max", "--grain-max", type=float, default=10.0)
    p.add_argument("-pitch", "--pitch", type=float, default=12.0)
    p.add_argument("-spread", "--spread", type=float, default=1.0)
    p.add_argument("-duration", "--duration", type=float, default=None)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_cloud)

    # --- stutter ---
    p = subparsers.add_parser("stutter", help="Stutter/glitch effect")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-chance", "--chance", type=float, default=0.4)
    p.add_argument("-repeats", "--repeats", type=int, default=4)
    p.add_argument("-reverse", "--reverse", type=float, default=0.2)
    p.add_argument("-crush", "--crush", type=float, default=0.1)
    p.add_argument("-bits", "--bits", type=int, default=8)
    p.add_argument("--no-preserve", action="store_true")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_stutter)

    # --- chop ---
    p = subparsers.add_parser("chop", help="Chop and shuffle audio slices")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-slices", "--slices", type=int, default=16)
    p.add_argument("-reverse", "--reverse", type=float, default=0.2)
    p.add_argument("-drop", "--drop", type=float, default=0.1)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_chop)

    # --- spectral ---
    p = subparsers.add_parser("spectral", help="Spectral smear processing")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-phase", "--phase", type=float, default=0.5)
    p.add_argument("-blur", "--blur", type=float, default=0.3)
    p.add_argument("-freeze", "--freeze", type=float, default=0.1)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_spectral)

    # --- mangle ---
    p = subparsers.add_parser("mangle", help="Aggressive spectral destruction")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-swap", "--swap", type=float, default=0.3)
    p.add_argument("-zero", "--zero", type=float, default=0.2)
    p.add_argument("-noise", "--noise", type=float, default=0.5)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_mangle)

    # --- quantize ---
    p = subparsers.add_parser("quantize", help="BPM detection and time-stretch")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-bpm", "--bpm", type=float, default=None)
    p.add_argument("-source-bpm", "--source-bpm", type=float, default=None)
    p.add_argument("--detect-only", action="store_true",
                    help="Only detect BPM, don't stretch")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_quantize)

    # --- sequence ---
    p = subparsers.add_parser("sequence", help="Probabilistic stem arranger")
    p.add_argument("inputs", nargs="+", help="Input audio files")
    p.add_argument("-duration", "--duration", type=float, default=30.0)
    p.add_argument("-density", "--density", type=float, default=0.5)
    p.add_argument("-cluster", "--cluster", type=float, default=0.3)
    p.add_argument("-layers", "--layers", type=int, default=3)
    p.add_argument("-fade", "--fade", type=float, default=10.0)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_sequence)

    # --- pattern ---
    p = subparsers.add_parser("pattern", help="Probabilistic step sequencer")
    p.add_argument("inputs", nargs="+", help="Input audio files")
    p.add_argument("-bpm", "--bpm", type=float, default=120.0)
    p.add_argument("-steps", "--steps", type=int, default=16)
    p.add_argument("-bars", "--bars", type=int, default=4)
    p.add_argument("-swing", "--swing", type=float, default=0.0)
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_pattern)

    # --- mix ---
    p = subparsers.add_parser("mix", help="Mix multiple stems")
    p.add_argument("inputs", nargs="+", help="Input audio files")
    p.add_argument("--no-normalize", action="store_true")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_mix)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
