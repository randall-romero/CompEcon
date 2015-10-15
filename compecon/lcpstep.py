__author__ = 'Randall'
import numpy as np


def minmax(x, y, z):
    """ Minmax Rootfinding Formulation

    Minimax transformation for solving NCP as rootfinding problem

    Args:
      x:
      y:
      z:

    Returns:
      vector: min( max(x, y), z)

    """
    return np.minimum(np.maximum(x, y), z)

def lcpstep(method, x, xl, xu, F, Fx=None):
    """ Newton step for Array Linear Complementarity Problem

    :param method:
    :param x: evaluation point
    :param xl: lower bound
    :param xu: upper bound
    :param F:  function value at x
    :param Fx: derivative of function at x
    :param dx: whether to return the derivative
    :return:
    """
    xlx = xl - x
    xux = xu - x

    if Fx is None:
        return minmax(F, xlx, xux) if method == 'minmax' else arrayss(x, xl, xu, F)

    if method == 'minmax':
        F = minmax(F, xlx, xux)
        dx = - arrayinvb(xlx, xux, F, Fx)
    else:
        F, Fx = arrayss(x, xl, xu, F, Fx)
        dx = - arrayinv(F, Fx)

    return F, minmax(dx, xlx, xux)


def arrayinvb(xlx, xux, F, Fx):
    nx, nx2, ns = Fx.shape
    B = minmax(F, xlx, xux).T
    ind1 = (F <= xlx).T
    ind2 = (F >= xux).T

    AA = np.tile(-np.identity(nx), [ns, 1, 1])
    A = Fx.swapaxes(0, 2)
    A[ind1] = AA[ind1]
    A[ind2] = AA[ind2]

    y = np.array([np.linalg.solve(a, b) for a, b in zip(A, B)])
    return y.T


def _arrayinvb2(xlx, xux, F, Fx):
    '''
       Same as arrayinvb, closer to Matlab's implementation.
       arrayinvb is vectorized and runs almost twice faster.
    '''
    nx, nx2, ns = Fx.shape
    y = np.zeros_like(F)
    AA = -np.identity(nx)

    ind1 = (F <= xlx).T
    ind2 = (F >= xux).T
    b = minmax(F, xlx, xux).T

    for i in range(ns):
        A = Fx[:, :, i].T
        A[ind1[i]] = AA[ind1[i]]
        A[ind2[i]] = AA[ind2[i]]
        y[:, i] = np.linalg.solve(A, b[i])
    return y



def arrayinv(F, Fx):
    return np.array([np.linalg.solve(a, b) for a, b in zip(Fx.swapaxes(0,2), F.T)]).T




def arrayss(x, xl, xu, F, Fx=None):
    if Fx is None:
        return arrayssx(x, xl, xu, F)

    Fnew, ff, aa = arrayssx(x, xl, xu, F, True)
    n, m = x.shape


def arrayssx(x, xl, xu, F):
    return None