"""Tests for CLI module."""

import pytest
from glitch.cli import build_parser


class TestCLI:
    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_granular_args(self):
        parser = build_parser()
        args = parser.parse_args(["granular", "input.wav", "-o", "out.wav",
                                   "-grains", "200", "-spread", "0.8"])
        assert args.command == "granular"
        assert args.input == "input.wav"
        assert args.output == "out.wav"
        assert args.grains == 200
        assert args.spread == 0.8

    def test_stutter_args(self):
        parser = build_parser()
        args = parser.parse_args(["stutter", "input.wav", "-o", "out.wav",
                                   "-chance", "0.6", "-repeats", "4"])
        assert args.command == "stutter"
        assert args.chance == 0.6
        assert args.repeats == 4

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
