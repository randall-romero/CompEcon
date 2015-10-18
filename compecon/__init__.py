"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.
"""

from .basis import Basis, SmolyakGrid, BasisOptions
from .basisChebyshev import BasisChebyshev
from .basisSpline import BasisSpline
from .basisLinear import BasisLinear
from .nonlinear import MCP, NLP, LCP
from .optimize import OP, MLE
# from .interpolator import Interpolator
from .dpmodel import DPmodel, DPoptions
from .tools import tic, toc