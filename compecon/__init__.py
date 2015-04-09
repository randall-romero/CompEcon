"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.
"""

from .basisChebyshev import BasisChebyshev
from .basis import Basis, SmolyakGrid
from .interpolator import Interpolator, InterpolatorArray