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


def _add_microsample_audio_args(p):
    """Add micro-sampling arguments for audio-only commands."""
    p.add_argument("--slice", type=float, default=None, help="Fixed slice length ms")
    p.add_argument("--slice-min", type=float, default=None)
    p.add_argument("--slice-max", type=float, default=None)
    p.add_argument("--mode", choices=["fixed", "transients", "random"], default="fixed")
    p.add_argument("--shuffle", type=float, default=0.3)
    p.add_argument("--stutter", type=float, default=0.2)
    p.add_argument("--max-repeats", type=int, default=4)
    p.add_argument("--reverse", type=float, default=0.1)
    p.add_argument("--drop", type=float, default=0.1)


def _get_audio_microsample_kwargs(args):
    """Extract microsample kwargs from parsed args."""
    kwargs = {}
    if args.slice is not None:
        kwargs["slice_ms"] = args.slice
    if args.slice_min is not None:
        kwargs["slice_min_ms"] = args.slice_min
    if args.slice_max is not None:
        kwargs["slice_max_ms"] = args.slice_max
    kwargs["mode"] = args.mode
    kwargs["shuffle_chance"] = args.shuffle
    kwargs["stutter_chance"] = args.stutter
    kwargs["max_repeats"] = args.max_repeats
    kwargs["reverse_chance"] = args.reverse
    kwargs["drop_chance"] = args.drop
    if hasattr(args, "seed") and args.seed is not None:
        kwargs["seed"] = args.seed
    return kwargs


def cmd_microsample(args):
    """Micro-sample audio: slice, shuffle, stutter, reverse, drop."""
    from glitch.microsample import microsample
    audio, sr = load(args.input)
    kwargs = _get_audio_microsample_kwargs(args)
    result, cut_list = microsample(audio, sr, **kwargs)
    save(args.output, normalize(result), sr)
    if args.cutlist:
        cut_list.save(args.cutlist)
        print(f"Cut list ({len(cut_list.cuts)} cuts) -> {args.cutlist}")
    print(f"Microsample -> {args.output}")


def cmd_chop(args):
    """Chop and shuffle audio slices."""
    from glitch.microsample import chop
    audio, sr = load(args.input)
    result, cut_list = chop(
        audio, sr,
        slices=args.slices,
        reverse_chance=args.reverse,
        drop_chance=args.drop,
        seed=args.seed,
    )
    save(args.output, normalize(result), sr)
    if args.cutlist:
        cut_list.save(args.cutlist)
        print(f"Cut list ({len(cut_list.cuts)} cuts) -> {args.cutlist}")
    print(f"Chop -> {args.output}")


def cmd_apply_audio(args):
    """Apply a saved cut list to audio."""
    from glitch.cutlist import CutList, apply_cut_list
    audio, sr = load(args.audio)
    cut_list = CutList.load(args.cutlist_path)
    result = apply_cut_list(audio, sr, cut_list)
    save(args.output, normalize(result), sr)
    print(f"Applied cut list -> {args.output}")


def cmd_chain(args):
    """Chain multiple micro-sampling passes."""
    from glitch.microsample import chain as ms_chain
    audio, sr = load(args.input)
    ops = _parse_chain_ops(args.ops)
    result, cut_lists = ms_chain(audio, sr, ops)
    save(args.output, normalize(result), sr)
    print(f"Chain ({len(cut_lists)} passes) -> {args.output}")


def _parse_chain_ops(ops_tokens: list[str]) -> list[dict]:
    """Parse flat token list from chain --op into list of param dicts.

    Tokens arrive as e.g.:
      ["microsample", "--slice", "50", "--shuffle", "0.4",
       "microsample", "--stutter", "0.6", "--max-repeats", "8"]

    Each "microsample" keyword starts a new operation.
    """
    param_map = {
        "slice": "slice_ms",
        "shuffle": "shuffle_chance",
        "stutter": "stutter_chance",
        "max_repeats": "max_repeats",
        "reverse": "reverse_chance",
        "drop": "drop_chance",
        "slice_min": "slice_min_ms",
        "slice_max": "slice_max_ms",
        "mode": "mode",
        "seed": "seed",
    }

    if not ops_tokens:
        return [{}]

    # Split tokens into groups at each "microsample" keyword
    groups: list[list[str]] = []
    for token in ops_tokens:
        if token == "microsample":
            groups.append([])
        elif groups:
            groups[-1].append(token)

    if not groups:
        return [{}]

    result = []
    for group in groups:
        params = {}
        i = 0
        while i < len(group):
            token = group[i]
            if token.startswith("--"):
                key = token.lstrip("-").replace("-", "_")
                if i + 1 < len(group) and not group[i + 1].startswith("--"):
                    raw = group[i + 1]
                    try:
                        val = float(raw)
                        if val == int(val) and "." not in raw:
                            val = int(raw)
                    except ValueError:
                        val = raw
                    mapped = param_map.get(key, key)
                    params[mapped] = val
                    i += 2
                    continue
            i += 1
        result.append(params)

    return result if result else [{}]


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
        description="Procedural micro-sampling & glitch engine for IDM production",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- microsample ---
    p = subparsers.add_parser("microsample", help="Micro-sample audio (slice, shuffle, stutter, reverse, drop)")
    p.add_argument("input", help="Input audio file")
    _add_microsample_audio_args(p)
    p.add_argument("--cutlist", type=str, default=None,
                    help="Export cut list to JSON (for Phase 2 video sync)")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_microsample)

    # --- chop ---
    p = subparsers.add_parser("chop", help="Chop and shuffle audio slices (shortcut for microsample)")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-slices", "--slices", type=int, default=16)
    p.add_argument("-reverse", "--reverse", type=float, default=0.2)
    p.add_argument("-drop", "--drop", type=float, default=0.1)
    p.add_argument("--cutlist", type=str, default=None,
                    help="Export cut list to JSON")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_chop)

    # --- apply (audio-only) ---
    p = subparsers.add_parser("apply", help="Apply saved cut list to audio or AV material")
    p.add_argument("cutlist_path", help="Path to cut list JSON")
    p.add_argument("audio", help="Input audio file")
    p.add_argument("visual", nargs="?", default=None, help="Visual media (optional, for AV mode)")
    _add_output_arg(p)

    def cmd_apply(args):
        if args.visual:
            from glitch.av.bond import bond
            from glitch.av.microsample import apply_cut_list as av_apply
            from glitch.av.render import render as av_render
            from glitch.cutlist import CutList
            clip = bond(args.audio, args.visual)
            cut_list = CutList.load(args.cutlist_path)
            result = av_apply(clip, cut_list)
            av_render(result, args.output)
            print(f"Applied cut list (AV) -> {args.output}")
        else:
            cmd_apply_audio(args)

    p.set_defaults(func=cmd_apply)

    # --- chain ---
    p = subparsers.add_parser("chain", help="Chain multiple micro-sampling passes")
    p.add_argument("input", help="Input audio file")
    _add_output_arg(p)
    _add_seed_arg(p)
    p.set_defaults(func=cmd_chain)

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

    # ===== Phase 2: AV subcommands =====

    def _add_microsample_args(p):
        p.add_argument("--slice", type=float, default=None, help="Fixed slice length ms")
        p.add_argument("--slice-min", type=float, default=None)
        p.add_argument("--slice-max", type=float, default=None)
        p.add_argument("--mode", choices=["fixed", "transients", "random"], default="fixed")
        p.add_argument("--shuffle", type=float, default=0.3)
        p.add_argument("--stutter", type=float, default=0.2)
        p.add_argument("--max-repeats", type=int, default=4)
        p.add_argument("--reverse", type=float, default=0.15)
        p.add_argument("--drop", type=float, default=0.05)
        p.add_argument("--effects", nargs="*", default=None,
                        choices=["rgb_split", "scan_lines", "corrupt",
                                 "invert_region", "posterize", "noise"])
        p.add_argument("--effect-chance", type=float, default=0.3)

    def _get_microsample_kwargs(args):
        kwargs = {}
        if args.slice is not None:
            kwargs["slice_ms"] = args.slice
        if args.slice_min is not None:
            kwargs["slice_min_ms"] = args.slice_min
        if args.slice_max is not None:
            kwargs["slice_max_ms"] = args.slice_max
        kwargs["mode"] = args.mode
        kwargs["shuffle_chance"] = args.shuffle
        kwargs["stutter_chance"] = args.stutter
        kwargs["max_repeats"] = args.max_repeats
        kwargs["reverse_chance"] = args.reverse
        kwargs["drop_chance"] = args.drop
        if args.effects:
            kwargs["effects"] = args.effects
            kwargs["effect_chance"] = args.effect_chance
        if hasattr(args, "seed") and args.seed is not None:
            kwargs["seed"] = args.seed
        return kwargs

    def _has_microsample_args(args):
        return any([
            args.slice is not None,
            args.slice_min is not None,
            args.slice_max is not None,
            getattr(args, "shuffle", 0.3) != 0.3,
            getattr(args, "stutter", 0.2) != 0.2,
            getattr(args, "max_repeats", 4) != 4,
            getattr(args, "reverse", 0.15) != 0.15,
            getattr(args, "drop", 0.05) != 0.05,
            getattr(args, "mode", "fixed") != "fixed",
            bool(getattr(args, "effects", None)),
        ])

    # --- render ---
    p = subparsers.add_parser("render", help="Bond audio+visual and render to .mp4")
    p.add_argument("audio", help="Input audio file")
    p.add_argument("visual", help="Image, image directory, or video file")
    p.add_argument("--resolution", type=str, default="1920x1080",
                    help="Output resolution WxH")
    p.add_argument("--fps", type=int, default=30)
    _add_microsample_args(p)
    _add_output_arg(p)
    _add_seed_arg(p)

    def cmd_render(args):
        from glitch.av.bond import bond
        from glitch.av.microsample import microsample
        from glitch.av.render import render as av_render
        w, h = [int(x) for x in args.resolution.split("x")]
        clip = bond(args.audio, args.visual, resolution=(w, h), fps=args.fps)
        ms_kwargs = _get_microsample_kwargs(args)
        if ms_kwargs.get("slice_ms") or ms_kwargs.get("slice_min_ms") or _has_microsample_args(args):
            clip, cut_list = microsample(clip, **ms_kwargs)
        av_render(clip, args.output)
        print(f"Rendered -> {args.output}")

    p.set_defaults(func=cmd_render)

    # --- compose ---
    p = subparsers.add_parser("compose", help="Compose from manifest or folder")
    p.add_argument("source", help="Manifest JSON or folder path")
    p.add_argument("--resolution", type=str, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--preview", type=float, default=None,
                    help="Preview first N seconds")
    _add_microsample_args(p)
    _add_output_arg(p)
    _add_seed_arg(p)

    def cmd_compose(args):
        import os
        from glitch.av.render import composite_from_manifest, composite_from_folder
        res = None
        if args.resolution:
            w, h = [int(x) for x in args.resolution.split("x")]
            res = (w, h)

        ms_kwargs = _get_microsample_kwargs(args)
        global_ms = ms_kwargs if _has_microsample_args(args) else None

        if os.path.isfile(args.source) and args.source.endswith(".json"):
            composite_from_manifest(
                args.source, args.output,
                resolution=res, fps=args.fps,
                global_microsample=global_ms,
                preview_s=args.preview,
            )
        else:
            composite_from_folder(
                args.source, args.output,
                resolution=res or (1920, 1080),
                fps=args.fps or 30,
                global_microsample=global_ms,
                preview_s=args.preview,
            )
        print(f"Composed -> {args.output}")

    p.set_defaults(func=cmd_compose)

    # --- manifest ---
    p = subparsers.add_parser("manifest", help="Generate manifest from folder")
    p.add_argument("folder", help="Folder containing subfolders with samples + visuals")
    _add_output_arg(p)

    def cmd_manifest(args):
        import json as _json
        from pathlib import Path
        folder = Path(args.folder)
        pairs = []
        for subdir in sorted(folder.iterdir()):
            if not subdir.is_dir():
                continue
            audio_path = None
            for ext in [".wav", ".flac", ".mp3"]:
                candidate = subdir / f"sample{ext}"
                if candidate.exists():
                    audio_path = str(candidate.relative_to(folder))
                    break
            if not audio_path:
                continue
            visual_path = None
            for name in ["visual.png", "visual.jpg", "visual.mp4"]:
                candidate = subdir / name
                if candidate.exists():
                    visual_path = str(candidate.relative_to(folder))
                    break
            if visual_path is None:
                imgs = [f for f in subdir.iterdir()
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                if imgs:
                    visual_path = str(subdir.relative_to(folder))
            if visual_path:
                pairs.append({
                    "audio": audio_path,
                    "visual": visual_path,
                    "offset": 0.0,
                    "gain_db": 0.0,
                    "microsample": None,
                })
        manifest = {
            "title": folder.name,
            "resolution": [1920, 1080],
            "fps": 30,
            "pairs": pairs,
        }
        with open(args.output, "w") as f:
            _json.dump(manifest, f, indent=2)
        print(f"Manifest ({len(pairs)} pairs) -> {args.output}")

    p.set_defaults(func=cmd_manifest)

    # --- cutlist ---
    p = subparsers.add_parser("cutlist", help="Generate and export a cut list from AV material")
    p.add_argument("audio", help="Input audio file")
    p.add_argument("visual", help="Visual media")
    _add_microsample_args(p)
    _add_output_arg(p)
    _add_seed_arg(p)

    def cmd_cutlist(args):
        from glitch.av.bond import bond
        from glitch.av.microsample import microsample as av_microsample
        clip = bond(args.audio, args.visual)
        ms_kwargs = _get_microsample_kwargs(args)
        _, cut_list = av_microsample(clip, **ms_kwargs)
        cut_list.save(args.output)
        print(f"Cut list ({len(cut_list.cuts)} cuts) -> {args.output}")

    p.set_defaults(func=cmd_cutlist)

    return parser


def _extract_chain_ops(argv: list[str]) -> tuple[list[str], list[str]]:
    """For `chain` command, split argv into argparse-safe args and --op tokens.

    Returns (clean_argv, op_tokens) where clean_argv has --op blocks removed
    and op_tokens is the flat list of tokens from all --op blocks.
    """
    clean = []
    ops = []
    i = 0
    in_ops = False
    while i < len(argv):
        if argv[i] == "--op":
            in_ops = True
            i += 1
            continue
        if in_ops and argv[i] in ("-o", "--output", "-seed", "--seed"):
            in_ops = False
        if in_ops:
            ops.append(argv[i])
        else:
            clean.append(argv[i])
        i += 1
    return clean, ops


def main():
    raw_argv = sys.argv[1:]

    # For chain: extract --op tokens before argparse sees them
    op_tokens = []
    if len(raw_argv) >= 1 and raw_argv[0] == "chain":
        raw_argv, op_tokens = _extract_chain_ops(raw_argv)

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "chain":
        args.ops = op_tokens

    args.func(args)


if __name__ == "__main__":
    main()
