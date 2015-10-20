__author__ = 'Randall'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
np.set_printoptions(4, linewidth=120)

from matplotlib import rcParams
rcParams['lines.linewidth'] = 2.5
rcParams['figure.subplot.hspace'] = 0.25
rcParams['legend.frameon'] = False
rcParams['savefig.directory'] = '/figures/'
rcParams['savefig.format'] = 'pdf'

from compecon import tic, toc


def demoaxes(title, xlab, ylab, xlim=None, ylim=None):
    plt.axes(title=title, xlabel=xlab, ylabel=ylab)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def demofigure(title, xlab, ylab, xlim=None, ylim=None):
    plt.figure()
    demoaxes(title, xlab, ylab, xlim, ylim)
