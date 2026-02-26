"""Tests for CLI module."""

import pytest
from glitch.cli import build_parser


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
