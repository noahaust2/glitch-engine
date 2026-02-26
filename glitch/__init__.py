"""glitch-engine: Procedural micro-sampling & glitch toolkit for IDM production."""

from glitch.core import load, save, normalize, fade, envelope
from glitch.cutlist import Cut, CutList, apply_cut_list
from glitch.microsample import microsample, chop, chain
from glitch.spectral import smear, mangle
from glitch.quantize import detect_bpm, quantize
from glitch.sequence import arrange, pattern
from glitch.mix import layer, mixdown

__version__ = "0.3.0"
