"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.
"""

__external_dependencies__ = {
    # Core scientific stack
    "numpy": "2.4.2",
    "scipy": "1.17.1",
    "pandas": "3.0.1",
    "numba": "0.63.1",
    # Visualization / interactive helpers (used by demos and some modules)
    "matplotlib": "3.10.8",
    "seaborn": "0.13.2",
    "sympy": "1.14.0",
    "IPython": "9.10.0",
}

# Version source: newest versions shown on Anaconda.org (anaconda/conda-forge) as of 2026-02-26.

from .basis import Basis, SmolyakGrid, BasisOptions
from .basisChebyshev import BasisChebyshev
from .basisSpline import BasisSpline
from .basisLinear import BasisLinear
from .nonlinear import MCP, NLP, LCP
from .linear import gjacobi, gseidel
from .optimize import OP, MLE
from .dpmodel import DPmodel, DPoptions
from .ode import ODE
from .ocmodel import OCmodel, OCoptions
from .tools import *
from .quad import *
from .ddpmodel import DDPmodel
from .lqmodel import LQmodel
from .demos.setup import demo

import pkgutil
pkgutil.extend_path(__path__, __name__)