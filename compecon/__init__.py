"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.
"""

from .basis import Basis, SmolyakGrid, BasisOptions
from .basisChebyshev import BasisChebyshev
from .basisSpline import BasisSpline
from .basisLinear import BasisLinear
from .nonlinear import MCP, NLP, LCP
from .linear import gjacobi, gseidel
from .optimize import OP, MLE
from .dpmodel import DPmodel, DPoptions
from .tools import *
from .quad import *
from .ddpmodel import DDPmodel
from .lqmodel import LQmodel
from .demos.setup import demo

import pkgutil
pkgutil.extend_path(__path__, __name__)