from functools import reduce
import numpy as np
from scipy.linalg import qz
import time

from scipy.sparse import identity


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
    if len(arrays) == 1:
        return arrays[0]

    arrays = np.atleast_2d(*arrays)
    n = len(arrays)
    idx = np.indices([a.shape[1] for a in arrays]).reshape([n, -1])
    return np.vstack([arrays[k][:, idx[k]] for k in range(n)])


def indices(*args):
    return np.array([a.flatten() for a in np.indices(args)])


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
    n, a, b = np.broadcast_arrays(*np.atleast_1d(n, a, b))
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


def jacobian(func, x, *args, **kwargs):

    # if type(func(x, *args, **kwargs)) is tuple:
    #     F = lambda x: func(x, *args, **kwargs)[0]
    # else:
    #     F = lambda x: func(x, *args, **kwargs)
    F = lambda z: np.asarray(func(z, *args, **kwargs))

    x = np.asarray(x).flatten()
    dx = x.size
    f = F(x)
    df = f.size
    x = x.astype(float)

    ''' Compute Jacobian'''
    tol = np.spacing(1) ** (1/3)

    h = tol * np.maximum(abs(x), 1)
    x_minus_h = x - h
    x_plus_h = x + h
    deltaX = x_plus_h - x_minus_h
    fx = np.zeros((dx, df))

    for k in range(dx):
        xx = x.copy()
        xx[k] = x_plus_h[k]
        fplus = F(xx)

        xx[k] = x_minus_h[k]
        fminus = F(xx)

        fx[k] = np.squeeze((fplus - fminus) / deltaX[k])  # fixme doing this to deal with broadcasting

    return fx.T


def hess1(func, x, *args, **kwargs):

    F = lambda z: np.asarray(func(z, *args, **kwargs))
    x = np.asarray(x).flatten().astype(float)

    k = x.size
    fx = F(x)

    '''Compute stepsize'''
    tol = np.spacing(1) ** (1 / 4)
    h = tol * np.maximum(np.abs(x), 1)
    xh = x + h
    h = xh - x
    ee = np.diag(h)

    '''Compute forward and backward steps'''
    gplus = np.zeros_like(x)
    gminus = np.zeros_like(x)
    for i, ei in enumerate(ee):
        gplus[i] = F(x + ei)
        gminus[i] = F(x - ei)

    H = np.outer(h, h)

    for i in range(k):
        for j in range(k):
            if i==j:
                H[i, i] = (gplus[i] + gminus[i] - 2*fx) / H[i, i]
            else:
                fxx = F(x + ee[i] - ee[j])
                H[i, j] = (gplus[i] + gminus[j] - fx - fxx) / H[i, j]

    return (H + H.T) / 2






def hessian(func, x, *args, **kwargs):

    F = lambda z: np.asarray(func(z, *args, **kwargs))
    x = np.asarray(x).flatten().astype(float)
    if x.ndim < 2:
        x = np.atleast_2d(x).T

    dx, nx = x.shape
    f = F(x)
    df = np.atleast_2d(f).shape[0]

    ''' Compute Hessian'''
    tol = np.spacing(1) ** (1/4)

    h = tol * np.maximum(abs(x), 1)
    x_minus_h = x - h
    x_plus_h = x + h
    deltaX = h  #repmat(h, 1, 1, dx)

    #deltaXX = deltaX .* permute(deltaX,[1,3,2])

    fxx = np.zeros([dx, dx, df, nx])
    for k in range(dx):
        for h in range(dx):
            xx = x.copy()
            if h == k:
                xx[k] = x_plus_h[k]
                fplus = F(xx)

                xx[k] = x_minus_h[k]
                fminus = F(xx)
                fxx[k, k] = (fplus - 2 * f + fminus) / (deltaX[k] ** 2)
            else:
                xx[k] = x_plus_h[k]
                xx[h] = x_plus_h[h]
                fpp = F(xx)

                xx[h] = x_minus_h[h]
                fpm = F(xx)

                xx[k] = x_minus_h[k]
                fmm = F(xx)

                xx[h] = x_plus_h[h]
                fmp = F(xx)

                fxx[k, h] = (fpp + fmm - fpm - fmp) / (4 * deltaX[k] * deltaX[h])

    fxx = (fxx + fxx.swapaxes(0, 1)) / 2
    return fxx.squeeze()


tic = lambda: time.time()
toc = lambda t: time.time() - t


class Options_Container(dict):

    """ A container for groups of attributes

    Used to overwrite the _getitem_ and the _setitem__ methods to handle several items at once
    Overwrites the __repr__ method to display instance's attributes
    """
    description = 'A container class'

    def __getitem__(self, item):
        """
        Get (possibly several) attributes of the container.

        Parameters
        ----------
        item: str or tuple(str)
            The key(s) to recover
        Returns
        -------
        Value(s)
            The values stored in the container, either one element (if item is str) or a tuple (if item is a tuple)

        """
        if isinstance(item, str):   # treat single key as list of length one
            return getattr(self, item)
        else:
            return (getattr(self, k) for k in item)

    def __setitem__(self, key, value):
        if type(key) is type({'a': None}.keys()):  # looks ugly, but I couldn't access the dict_keys class directly
            key = tuple(key)

        if type(key) is not tuple:
            key = (key,)
            value = (value,)

        valid_attributes = self.__dict__.keys()

        if len(key) != len(value):
            raise ValueError('Assigning {} attributes but {} values where provided!'.format(len(key), len(value)))

        for k, v in zip(key, value):
            if k in valid_attributes:
                self.__dict__[k] = v
            else:
                raise ValueError('{} is not a valid attribute for class {}.'.format(k, type(self)))

    def __repr__(self):
        sorted_keys = sorted(self.__dict__.keys())
        maxstr = str(max(len(s) for s in sorted_keys))
        form = '\t{:>' + maxstr + 's} = '

        txt = self.description + ':\n\n'
        for j in sorted_keys:
            txt += form.format(j) + str(self[j]) + '\n'
        return txt


''' Define some convenient functions '''


def example(page):
    print('\n' + '=' * 72)
    print('Example on page {:d}\n'.format(page))

def exercise(number):
    print('\n' + '=' * 72)
    print('Exercise {:s}\n'.format(number))


# def getindex(values, array):
#     return np.abs(np.subtract.outer(array, values)).argmin(0)


def getindex(s, S):
    S, s = np.atleast_2d(S, s)
    d, n = S.shape
    d2, p = s.shape

    if (p == d) and (d2 != d):
        s = s.T
        d2, p = p, d2

    assert d == d2, 's and S must have same number of rows'

    return np.array([np.abs(S.T - k).sum(1).argmin() for k in s.T])


def qzordered(A, B):
    """
    QZORDERED QZ decomposition ordered by the absolute value of the generalized eigenvalues
    See QZ

    Based on code by Pedro Oviedo & Chris Sims
    """
    n = A.shape[0]
    S, T, Q, Z = qz(A, B)
    Q = Q.T

    i = 0
    while i < n - 1:
        if abs(T[i, i] * S[i+1, i+1]) > abs(S[i, i] * T[i+1, i+1]):
            qzswitch(i,S, T, Q, Z)
            if i > 0:
                i -=1
            else:
                i += 1
        else:
            i += 1

    return S, T, Q, Z


def qzswitch(i, S, T, Q, Z):
    a, b = S[i, i:i+2]
    d, e = T[i, i:i+2]
    c = S[i+1, i+1]
    f = T[i+1, i+1]
    wz = np.array([c*e - f*b, c*d - f*a])
    n = np.sqrt(np.inner(wz, wz))

    if n == 0:
       return
    else:
       xy = np.array([b*d - e*a, c*d - f*a])
       m = np.sqrt(np.inner(xy, xy))
       wz /= n
       xy /= m
       wz = np.r_[np.atleast_2d(wz), np.atleast_2d([-wz[1], wz[0]])]
       xy = np.r_[np.atleast_2d(xy), np.atleast_2d([-xy[1], xy[0]])]

       S[i:i+2] = xy.dot(S[i:i+2])
       T[i:i+2] = xy.dot(T[i:i+2])
       S[:,i:i+2] = S[:,i:i+2].dot(wz)
       T[:,i:i+2] = T[:,i:i+2].dot(wz)
       Z[:,i:i+2] = Z[:,i:i+2].dot(wz)
       Q[i:i+2] = xy.dot(Q[i:i+2])

def ix(A):
    """
    Return the numpy ix_ indices to recreate array A, that is

    A[ix(A)] = A

    Useful to change the indices of a specific dimension
    """
    return list(np.ix_(*(np.arange(k) for k in A.shape)))



def markov(p):
    # taken from Mario's compecon

    # Error Checking to ensure P is a valid stochastic matrix
    assert p.ndim == 2, 'Transition matrix does not have two dimensions'
    n, n2 = p.shape
    assert n == n2, 'Transition matrix is not square'
    assert np.all(p >= 0), 'Transition matrix contains negative elements'
    assert np.all(np.abs(p.sum(1) - 1) < 1e-12), 'Rows of transition matrix do not sum to 1'

    spones = lambda A: (A != 0).astype(int)

    # Determine accessibility from i to j
    f = np.empty_like(p)
    for j in range(n):
        dr = 1
        r = spones(p[:,j])  # a vector of ones where p(i,j)~=0
        while np.any(dr):
            dr = r
            r = spones(p.dot(r) + r)
            dr = r - dr
            f[:, j] = r

    # Determine membership in recurrence classes
    ind = np.zeros_like(p)
    numrec = -1  # number of recurrence classes
    for i in range(n):
        if np.all(ind[i] == 0):
            j = f[i]  # states accessible from i
            if np.all((f[:, i].T * j) == j):  # are all accessible states communicating states?
                j = np.where(j)[0]           # members in class with state i
                k = j.size                  # number of members
                if k:
                    numrec += 1
                    ind[j, numrec] = 1

    ind = ind[:, :numrec + 1]        # ind(i,j)=1 if state i is in class j

    # Determine recurrence class invariant probabilities
    q = np.zeros((n, numrec + 1))
    for j in range(1 + numrec):
        k = np.where(ind[:, j])[0]
        nk = k.size
        k0, k1 = np.ix_(k, k)
        A = np.asarray(np.r_[np.ones((1, nk)), identity(nk) - p[k0, k1].T])
        B = np.asarray(np.r_[np.ones((1, 1)), np.zeros((nk, 1))])
        q[k, j] = np.linalg.lstsq(A, B)[0].flatten()

    return q  # todo: Mario's code has a second output argument