"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.

The demos folder contains (sometimes different) versions of the original demos.
"""

from .setup import demo
import pkgutil
pkgutil.extend_path(__path__, __name__)

