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
    return np.fmin(np.fmax(x, y), z)


def lcpstep(method: str, x: np.array, xl: np.array, xu: np.array, F: np.array, Fx=None):
    """ Newton step for Array Linear Complementarity Problem

    Args:
        method: 'minmax' or 'ssmooth'
        x: dx.ds evaluation point
        xl: dx.ds lower bound
        xu: dx.ds upper bound
        F: dx.ds function value at x
        Fx: dx.dx.ds derivative of function at x

    Returns:

    """
    xlx = xl - x
    xux = xu - x

    if Fx is None:
        return minmax(F, xlx, xux) if method == 'minmax' else arrayss(x, xl, xu, F)

    if Fx.shape[0] != Fx.shape[1]:
        raise 'Fx must be square.'

    if method == 'minmax':
        F = minmax(F, xlx, xux)
        dx = - arrayinvb(xlx, xux, F, Fx)
    else:
        F, Fx = arrayss(x, xl, xu, F, Fx)
        dx = - arrayinv(F, Fx)

    return F, minmax(dx, xlx, xux)


def arrayinvb(xlx, xux, F, Fx):
    """

    Args:
        xlx: dx.ds lower bound
        xux: dx.ds upper bound
        F: dx.ds function value at x
        Fx: dx.dx.ds derivative of function at x

    Returns:

    """
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
    """

    Same as arrayinvb, closer to Matlab's implementation. arrayinvb is vectorized and runs almost twice faster.

    Args:
        xlx: dx.ds lower bound
        xux: dx.ds upper bound
        F: dx.ds function value at x
        Fx: dx.dx.ds derivative of function at x

    Returns:

    """
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
    """

    Args:
        F: dx.ds function value at x
        Fx: dx.dx.ds derivative of function at x

    Returns:

    """
    return np.array([np.linalg.solve(a, b) for a, b in zip(Fx.swapaxes(0,2), F.T)]).T


def arrayss(x, xl, xu, F, Fx=None):
    """

    Args:
        x: dx.ds evaluation point
        xl: dx.ds lower bound
        xu: dx.ds upper bound
        F: dx.ds function value at x
        Fx: dx.dx.ds derivative of function at x

    Returns:
        Fnew: dx.ds function value at x
        Fxnew: dx.dx.ds derivative of function at x (if Fx is not None)

    """
    if Fx is None:
        return arrayssx(x, xl, xu, F)

    Fnew, ff, aa = arrayssx(x, xl, xu, F, True)
    n, m = x.shape
    Fxnew = np.tile(ff, (n, 1)) * Fx.reshape(n * n, m)
    # index for the diagonal elements of the Jacobian
    ind = np.arange(0, n * n, n) + np.arange(n)
    Fxnew[ind] = Fxnew[ind] - aa
    return Fnew, Fxnew.reshape(n, n, m)



def arrayssx(x: np.array,
             xl: np.array,
             xu: np.array,
             F: np.array,
             derivatives=False
             ):
    """

    Args:
        x: dx.ds evaluation point
        xl: dx.ds lower bound
        xu: dx.ds upper bound
        F: dx.ds function value at x
        derivatives:

    Returns:

    """
    nx, ns = x.shape
    n = x.size

    # Flattening is necessary for linear indexing a-la MATLAB
    F, x, xl, xu = (z.flatten() for z in (F, x, xl, xu))


    Fnew = np.zeros_like(x)
    if derivatives:
        ffout = np.zeros_like(x)
        aaout = np.zeros_like(x)



    for j in range(n):
        # compute phi+
        if np.isinf(xl[j]):
            d = F[j]
        else:
            dxl = xl[j] - x[j]
            y, z = (F[j], dxl) if abs(F[j]) > abs(dxl) else (dxl, F[j])
            z /= y
            dplus = np.sqrt(1 + z * z)
            d = y * (1 + dplus + z if (y > 0) else  z - ((1 - dplus) * (1 - dplus) + z * z) / dplus / 2)

        # compute phi-
        if np.isinf(xu[j]):
            Fnew[j] = d
        else:
            dxu = xu[j]-x[j]
            g, h = (d, dxu) if (abs(d) > abs(dxu)) else (dxu, d)
            h /= g
            dminus = np.sqrt(1 + h * h)
            Fnew[j] = g * (1 + dminus + h if (g<0) else h - ((1 - dminus) * (1 - dminus) + h * h) / dminus / 2)

        # compute Jacobian factors if requested
        if derivatives:
            if np.isinf(xu[j]):
                ff, aa, bb = 1, 1, 0
            else:
                if g < 0: dminus = -dminus
                hh1 = np.array([1, 1, h] if (abs(d) > abs(dxu)) else [h, h, 1])
                ff, aa, bb = 1 - hh1 / dminus

            if np.isinf(xl[j]):
                aa = 0
            else:
                if y < 0: dplus = -dplus
                z_1 = np.array([1, z] if (abs(F[j]) > abs(dxl)) else [z, 1])
                temp = 1 + z_1 / dplus
                ff *= temp[0]
                aa *= temp[1]

            ffout[j] = ff
            aaout[j] = aa + bb

    return (Fnew.reshape(nx, ns), ffout.reshape(nx, ns), aaout.reshape(nx, ns)) if derivatives else Fnew.reshape(nx, ns)