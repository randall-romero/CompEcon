__author__ = 'Randall'

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('dark')
np.set_printoptions(4, linewidth=120)
import os
import inspect
#import plotnine as gg

from warnings import simplefilter
simplefilter('ignore')


# from matplotlib import rcParams
# rcParams['lines.linewidth'] = 2.5
# rcParams['figure.subplot.hspace'] = 0.25
# rcParams['legend.frameon'] = False
# rcParams['savefig.directory'] = './figures/'
# rcParams['savefig.format'] = 'pdf'
# rcParams['font.size'] = 24

mpl.rc('lines', linewidth=2.5)
mpl.rc('font', size=18)
mpl.rc('savefig', directory='./figures', format='pdf')
plt.rcParams['figure.figsize'] = 12, 6

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
        return plt.gcf()

    @staticmethod
    def subplot(nr, nc, i, title, xlab, ylab, xlim=None, ylim=None):
        plt.subplot(nr, nc, i, title=title, xlabel=xlab, ylabel=ylab)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        return plt.gca()


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

    @staticmethod
    def qplot(x, y, color, data):
        '''Makes plot from pandas
        x: numeric variable
        y: numeric variable
        color: categoric variable
        '''
        return plt.plot(data.pivot(index=x,columns=color, values=y))


    def savefig(*args, name=None, format=['pdf', 'png']):
        """
        Saves a list of figures to disk, in the specified formats
        :param args: a collection of matplotlib figure objects
        :param name: a string, root name for saved files
        :param format: a string of list of strings indicating file formats.
        :return: None
        """
        name = name if name else 'figures/' + os.path.basename(inspect.stack()[1].filename)[:-3]
        formats = format if isinstance(format, list) else [format]

        for n, fig in enumerate(*args):
            for ext in formats:
                fname = f'{name}--{n+1:02d}.{ext}'
                fig.savefig(fname, bbox_inches='tight')


