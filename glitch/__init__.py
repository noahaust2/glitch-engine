"""glitch-engine: Procedural sampling & glitch toolkit for IDM production."""

from glitch.core import load, save, normalize, fade, envelope
from glitch.granular import scatter, cloud
from glitch.stutter import glitch as stutter_glitch, chop
from glitch.spectral import smear, mangle
from glitch.quantize import detect_bpm, quantize
from glitch.sequence import arrange, pattern
from glitch.mix import layer, mixdown

__version__ = "0.1.0"
