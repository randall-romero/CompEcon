__author__ = 'Randall'

import warnings
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
    warnings.warn('Deprecated: use demo.axes(*args, **kwargs) instead')
    demo.axes(title, xlab, ylab, xlim, ylim)

def demofigure(title, xlab, ylab, xlim=None, ylim=None):
    warnings.warn('Deprecated: use demo.figure(*args, **kwargs) instead')
    demo.figure(title, xlab, ylab, xlim, ylim)


class demo(object):
    """
    A class of static methods to facilitate making demo figures
    """

    @staticmethod
    def axes(title, xlabel, ylabel, xlim=None, ylim=None):
        warnings.warn('DO NOT USE, use demo.subplot() if really need this')
        plt.axes(title=title, xlabel=xlabel, ylabel=ylabel)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

    @staticmethod
    def figure(title, xlab, ylab, xlim=None, ylim=None, **kwargs):
        plt.figure(**kwargs)
        plt.axes(title=title, xlabel=xlab, ylabel=ylab)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

    @staticmethod
    def subplot(nr, nc, i, title, xlab, ylab, xlim=None, ylim=None):
        plt.subplot(nr, nc, i, title=title, xlabel=xlab, ylabel=ylab)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)


    @staticmethod
    def bullet(x, y, spec='k.', ms=16):
        plt.plot(x, y, spec, markersize=ms, lw=0)

    @staticmethod
    def text(x, y, txt, ha='center', va='center', fs=18, **kwargs):
        plt.text(x, y, txt, ha=ha, va=va, fontsize=fs, **kwargs)

    @staticmethod
    def annotate(x, y, txt, spec='ko', offset=(5, 5), fs=18, ms=14, **kwargs):
        xl = plt.xlim()
        yl = plt.ylim()
        h0 = offset[0] * (xl[1] - xl[0]) / 100
        v0 = offset[1] * (yl[1] - yl[0]) / 100
        demo.bullet(x, y, spec, ms)
        demo.text(x + h0, y + v0, txt, fs=fs, **kwargs)
