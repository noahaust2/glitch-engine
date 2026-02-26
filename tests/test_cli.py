"""Tests for CLI module."""

import pytest
from glitch.cli import build_parser, _parse_chain_ops, _extract_chain_ops


class TestCLI:
    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_microsample_args(self):
        parser = build_parser()
        args = parser.parse_args(["microsample", "input.wav", "-o", "out.wav",
                                   "--slice", "40", "--shuffle", "0.6"])
        assert args.command == "microsample"
        assert args.input == "input.wav"
        assert args.output == "out.wav"
        assert args.slice == 40.0
        assert args.shuffle == 0.6

    def test_microsample_cutlist_export(self):
        parser = build_parser()
        args = parser.parse_args(["microsample", "input.wav", "-o", "out.wav",
                                   "--cutlist", "cuts.json"])
        assert args.cutlist == "cuts.json"

    def test_chop_args(self):
        parser = build_parser()
        args = parser.parse_args(["chop", "input.wav", "-o", "out.wav",
                                   "--slices", "32", "--reverse", "0.3"])
        assert args.command == "chop"
        assert args.slices == 32
        assert args.reverse == 0.3

    def test_apply_args(self):
        parser = build_parser()
        args = parser.parse_args(["apply", "cuts.json", "input.wav", "-o", "out.wav"])
        assert args.command == "apply"
        assert args.cutlist_path == "cuts.json"
        assert args.audio == "input.wav"

    def test_apply_with_visual(self):
        parser = build_parser()
        args = parser.parse_args(["apply", "cuts.json", "input.wav", "frames/", "-o", "out.mp4"])
        assert args.visual == "frames/"

    def test_spectral_args(self):
        parser = build_parser()
        args = parser.parse_args(["spectral", "input.wav", "-o", "out.wav",
                                   "-phase", "0.8", "-blur", "0.5"])
        assert args.command == "spectral"
        assert args.phase == 0.8
        assert args.blur == 0.5

    def test_quantize_args(self):
        parser = build_parser()
        args = parser.parse_args(["quantize", "input.wav", "-o", "out.wav",
                                   "-bpm", "140"])
        assert args.command == "quantize"
        assert args.bpm == 140.0

    def test_sequence_args(self):
        parser = build_parser()
        args = parser.parse_args(["sequence", "a.wav", "b.wav", "-o", "out.wav",
                                   "-duration", "60", "-density", "0.5"])
        assert args.command == "sequence"
        assert args.inputs == ["a.wav", "b.wav"]
        assert args.duration == 60.0

    def test_mix_args(self):
        parser = build_parser()
        args = parser.parse_args(["mix", "a.wav", "b.wav", "-o", "out.wav"])
        assert args.command == "mix"
        assert args.inputs == ["a.wav", "b.wav"]

    def test_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_render_args(self):
        parser = build_parser()
        args = parser.parse_args(["render", "sample.wav", "visual.png",
                                   "-o", "output.mp4", "--slice", "40"])
        assert args.command == "render"
        assert args.audio == "sample.wav"
        assert args.visual == "visual.png"

    def test_compose_args(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "manifest.json", "-o", "track.mp4"])
        assert args.command == "compose"

    def test_cutlist_args(self):
        parser = build_parser()
        args = parser.parse_args(["cutlist", "sample.wav", "frames/",
                                   "--slice", "30", "-o", "cuts.json"])
        assert args.command == "cutlist"


class TestChainParsing:
    def test_extract_chain_ops_basic(self):
        argv = ["chain", "input.wav",
                "--op", "microsample", "--slice", "50", "--shuffle", "0.4",
                "--op", "microsample", "--stutter", "0.6", "--max-repeats", "8",
                "-o", "out.wav"]
        clean, ops = _extract_chain_ops(argv)
        assert clean == ["chain", "input.wav", "-o", "out.wav"]
        assert ops == ["microsample", "--slice", "50", "--shuffle", "0.4",
                        "microsample", "--stutter", "0.6", "--max-repeats", "8"]

    def test_extract_chain_ops_with_seed(self):
        argv = ["chain", "input.wav",
                "--op", "microsample", "--shuffle", "0.5",
                "-o", "out.wav", "--seed", "42"]
        clean, ops = _extract_chain_ops(argv)
        assert clean == ["chain", "input.wav", "-o", "out.wav", "--seed", "42"]
        assert ops == ["microsample", "--shuffle", "0.5"]

    def test_parse_chain_ops_single(self):
        ops = _parse_chain_ops(["microsample", "--slice", "50", "--shuffle", "0.4"])
        assert len(ops) == 1
        assert ops[0]["slice_ms"] == 50
        assert ops[0]["shuffle_chance"] == 0.4

    def test_parse_chain_ops_multi(self):
        ops = _parse_chain_ops([
            "microsample", "--slice", "50", "--shuffle", "0.4",
            "microsample", "--stutter", "0.6", "--max-repeats", "8",
        ])
        assert len(ops) == 2
        assert ops[0]["slice_ms"] == 50
        assert ops[0]["shuffle_chance"] == 0.4
        assert ops[1]["stutter_chance"] == 0.6
        assert ops[1]["max_repeats"] == 8

    def test_parse_chain_ops_empty(self):
        ops = _parse_chain_ops([])
        assert ops == [{}]

    def test_chain_argv_parses_correctly(self):
        """Full integration: extract ops, parse remaining with argparse."""
        argv = ["chain", "input.wav",
                "--op", "microsample", "--slice", "40",
                "-o", "out.wav"]
        clean, ops = _extract_chain_ops(argv)
        parser = build_parser()
        args = parser.parse_args(clean)
        assert args.command == "chain"
        assert args.input == "input.wav"
        assert args.output == "out.wav"
        parsed_ops = _parse_chain_ops(ops)
        assert parsed_ops[0]["slice_ms"] == 40

    def test_per_op_seed_not_swallowed(self):
        """Per-op --seed inside --op blocks should stay with the op."""
        argv = ["chain", "input.wav",
                "--op", "microsample", "--slice", "50", "--seed", "11",
                "--op", "microsample", "--stutter", "0.6", "--seed", "22",
                "-o", "out.wav"]
        clean, ops = _extract_chain_ops(argv)
        assert "-o" in clean
        parsed = _parse_chain_ops(ops)
        assert len(parsed) == 2
        assert parsed[0]["seed"] == 11
        assert parsed[1]["seed"] == 22
