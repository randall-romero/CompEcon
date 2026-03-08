"""CompEcon: A Python implementation of Miranda & Fackler's CompEcon for MATLAB

The CompEcon package provides tools for computational economics. It's code is based on Miranda-Fackler toolboox, but
much of the functionality is implemented by OOP when possible.

The demos folder contains (sometimes different) versions of the original demos.
"""

from .setup import demo

import numpy as np
import warnings
import seaborn as sns
sns.set_style('dark')
np.set_printoptions(4, linewidth=120)
import matplotlib as mpl
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter('ignore')


from matplotlib import rcParams
rcParams['lines.linewidth'] = 2.5
rcParams['figure.subplot.hspace'] = 0.25
rcParams['legend.frameon'] = False
rcParams['savefig.directory'] = './figures/'
rcParams['savefig.format'] = 'pdf'
rcParams['font.size'] = 24

mpl.rc('lines', linewidth=2.5)
mpl.rc('font', size=18)
mpl.rc('savefig', directory='./figures', format='pdf')
plt.rcParams['figure.figsize'] = 12, 6

from compecon import tic, toc

