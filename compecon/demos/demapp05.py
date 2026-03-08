from demos.setup import np, plt
from compecon import BasisChebyshev, BasisSpline
from compecon.tools import nodeunif

__author__ = 'Randall'

## DEMAPP05 Chebychev polynomial and spline approximantion of various functions

# Preliminary tasks

# Demonstrates Chebychev polynomial, cubic spline, and linear spline approximation for the following functions
#   1: y = sqrt(x+1)
#   2: y = exp(-x)
#   3: y = 1./(1+25*x.^2).
#   4: y = sqrt(abs(x))

## Functions to be approximated
funcs = [lambda x: 1 + x + 2 * x ** 2 - 3 * x ** 3,
         lambda x: np.exp(-x),
         lambda x: 1 / ( 1 + 25 * x ** 2),
         lambda x: np.sqrt(np.abs(x))]

# Set degree of approximation and endpoints of approximation interval
n = 7   # degree of approximation
a = -1  # left endpoint
b = 1   # right endpoint

# Construct uniform grid for error ploting
x = nodeunif(2001, a, b)

def subfig(k, x, y, xlim, ylim, title):
    plt.subplot(2, 2, k)
    plt.plot(x, y)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)

for ifunc, ff in enumerate(funcs):

    # Construct interpolants
    C = BasisChebyshev(n, a, b, f=ff)
    S = BasisSpline(n, a, b, f=ff)
    L = BasisSpline(n, a, b, k=1, f=ff)

    # Compute actual and fitted values on grid
    y = ff(x)  # actual
    yc = C(x)  # Chebychev approximant
    ys = S(x)  # cubic spline approximant
    yl = L(x)  # linear spline approximant

    # Plot function approximations
    plt.figure()
    ymin = np.floor(y.min())
    ymax = np.ceil(y.max())
    xlim = [a, b]
    ylim = [-0.2, 1.2] if ifunc==2 else [ymin, ymax]

    subfig(1, x, y, xlim, ylim, 'Function')
    subfig(2, x, yc, xlim, ylim, 'Chebyshev')
    subfig(3, x, ys, xlim, ylim, 'Cubic Spline')
    subfig(4, x, yl, xlim, ylim, 'Linear Spline')

    # Plot function approximation error
    plt.figure()
    plt.plot(x, np.c_[yc - y, ys - y], linewidth=3)
    plt.xlabel('x')
    plt.ylabel('Approximation Error')
    plt.legend(['Chebychev Polynomial','Cubic Spline'])
    # pltlegend('boxoff')

plt.show()

