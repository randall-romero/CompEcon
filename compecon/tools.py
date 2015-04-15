from functools import reduce
import numpy as np


def gridmake(*arrays):
    """
    Forms grid points
    USAGE
    X = gridmake(x1,x2,...,xn)
    X1,..., Xn = gridmake(x1,x2,...,xn)


    Expands matrices into the associated grid points.
    If N is the 2xd array that indexes the size of the inputs, GRIDMAKE returns a sum(N[0]) by prod(N[1]) array.
    The output can also be returned as either
      d matrices or
      sum(N(:,2)) matrices

    Note: the grid is expanded so the last variable change most quickly.

    Example:
    X = gridmake([1, 2, 3], [4, 5])
    array([[1, 1, 2, 2, 3, 3],
           [4, 5, 4, 5, 4, 5]])

    Also the inputs need not be vectors.
    Y = gridmake(X, [10, 20])
    array([[ 1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3],
           [ 4,  4,  5,  5,  4,  4,  5,  5,  4,  4,  5,  5],
           [10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20]])
    """
    arrays = [np.atleast_2d(a) for a in arrays]
    n = len(arrays)
    idx = np.indices([a.shape[1] for a in arrays]).reshape([n, -1])
    return np.vstack(arrays[k][:, idx[k]] for k in range(n))


def ckron(*arrays, invert=False):
    """
    Repeatedly applies the np.kron function to an arbitrary number of
    input arrays

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray

    Returns
    -------
    out : np.ndarray
        The result of repeated kronecker products

    Notes
    -----
    Based of original function `ckron` in CompEcon toolbox by Miranda
    and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    if invert:
        arrays = (np.linalg.inv(a) for a in arrays)

    return reduce(np.kron, arrays)


def nodeunif(n, a, b, lst = False):
    """
    UPDATE THIS DOCSTRING!!!
    NODEUNIF Computes uniform nodes for intervals R^n
     USAGE
       [x,xcoord] = nodeunif(n,a,b)

     If dimension d of n, a and b is one, returns n by 1 vector x containing n uniform nodes spanning
     the interval [a,b]. If dimension d>1, returns 1 by d cell array xcoord whose kth entry is the n(k) by 1
     vector of n(k) uniform nodes spanning the interval [a(k),b(k)] also returns prod(n) by d matrix x of
     grid points created by forming Cartesian product of  vectors in xcoord.
    """
    n, a, b = map(np.atleast_1d, (n, a, b))  # Convert n, a, and b to np.arrays
    d = n.size  # dimension of basis

    if d == 1:
        x = np.linspace(a, b, n)
        xlst = [x]
    else:
        xlst = list(np.linspace(a[k], b[k], n[k]) for k in range(d))
        x = gridmake(*xlst)

    return (x, xlst) if lst else x


def discmoments(w, x):
    """ Generate means and standard deviations of discrete distribution.

    :param w: discrete probabilities
    :param x: values for k distinct discrete variates
    :return: (xavg, xstd) tuple with means and standard deviations
    """
    xavg = np.dot(x, w)
    xstd = np.sqrt(np.dot(x ** 2, w) - xavg ** 2)
    return xavg, xstd