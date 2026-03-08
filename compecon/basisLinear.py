import numpy as np
from scipy.sparse import csc_matrix, diags, tril
from compecon import Basis

__author__ = 'Randall'
# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators
# fixme: review interpolation matrix

class BasisLinear(Basis):
    def __init__(self, *args, **kwargs):

        nargs = len(args)
        if nargs == 1:
            if isinstance(args[0], tuple):
                breaks = [np.sort(br) for br in args[0]]
                n = np.array([br.size for br in breaks])
                a = np.array([br[0] for br in breaks])
                b = np.array([br[-1] for br in breaks])
                kwargs['nodetype'] = 'user'
            else:
                raise ValueError("If only 1 positional argument is provided, it must be a tuple of 'd' array-like, " +
                                 "each of them containing the breaks for one dimension.")
        elif nargs == 3:
            n, a, b = np.broadcast_arrays(*np.atleast_1d(*args))
            breaks = [aa + (bb - aa) * np.arange(nn) / (nn - 1) for aa, bb, nn in zip(a, b, n)]  # Warning: using linspace introduces rounding error
            kwargs['nodetype'] = 'canonical'
        else:
            txt = 'Either 1 or 3 positional arguments must be provided\n'
            txt += '\t1 argument -> break points\n'
            txt += '\t3 argument -> n, a, b'
            raise ValueError(txt)

        ''' Check inputs '''
        assert np.all(n > 1), 'There must be at least two nodes per dimension'

        ''' Make instance '''
        kwargs['basistype'] = 'linear'
        super().__init__(n, a, b, **kwargs)
        self.breaks = breaks
        self._set_nodes()

    def _set_nodes(self):
            """
            Sets the basis nodes

            :return: None
            """
            self._nodes = list()

            for i in range(self.d):
                self._nodes.append(self.breaks[i])
            self._expand_nodes()

    def _augbreaks(self, i, m,):
        aa = np.repeat(self.a[i], m)
        bb = np.repeat(self.b[i], m)
        return np.concatenate((aa, self.breaks[i], bb))

    def _update_diff_operators(self, i, order):
        """
        Updates the list _D of differentiation operators

        :param order: order of required derivative
        :return: None
        """
        keys = set(self._diff_operators[i].keys())

        if (order in keys) or (order == 0):
            return  # Use previously stored values if available

        n = self.n[i]
        newbreaks = self.breaks[i].copy()

        def data(d):
            return np.concatenate((-d, d))

        if order > 0:

            for j in range(1, 1 + order):
                hh = np.arange(n-1)
                r_id = np.tile(hh, 2)
                c_id = np.concatenate((hh, hh + 1))
                d = 1 / np.diff(newbreaks)
                dd = csc_matrix((data(d), (r_id, c_id)), shape=(n - 1, n))
                if j > 1:
                    # self._diff_operators[i][j] = np.dot(dd, self._diff_operators[i][j - 1])  #make sure * is matrix mult
                    self._diff_operators[i][j] = dd * self._diff_operators[i][j - 1]
                else:
                    self._diff_operators[i][1] = dd

                newbreaks = (newbreaks[:-1] + newbreaks[1:]) / 2
                n -= 1
        else:
            for j in np.arange(-1, order - 1, -1):
                newbreaks = np.concatenate((
                    np.inner([3, -1], newbreaks[:2]),
                    newbreaks[:-1] + newbreaks[1:],
                    np.inner([-1, 3], newbreaks[-2:])
                )) / 2
                d = np.diff(newbreaks)
                n += 1
                dd = tril(np.repeat(np.atleast_2d(d), n, 0), -1)
                if j < -1:
                    self._diff_operators[i][j] = np.dot(dd, self._diff_operators[i][j + 1])
                else:
                    self._diff_operators[i][-1] = dd

                # adjustment to make value at original left endpoint equal 0
                temp = self._phi1d(i, self.a, 0)[0]  # todo review that this is equivalent to Mario's
                assert np.isscalar(temp), 'ASUMI MAL, CREI QUE ERA ESCALAR, REVISAR CODIGO QUE SIGUE'
                self._diff_operators[i][j][:, 0] -= temp


    """
        Interpolation methods
    """
    def _phi1d(self, i, x=None, order=0):
        """
        Computes interpolation matrices for given data x and order of differentiation 'order' (integration if negative)

        :param x:  evaluation points (defaults to nodes)
        :param order: a list of orders for differentiation (+) / integration (-)
        :return a: dictionary with interpolation matrices, keys given by unique elements of order.

        Example: Create a basis with 5 nodes, get the interpolation matrix evaluated at 20 points::

                n, a, b = 5, 0, 4
                x = numpy.linspace(a,b, 20)
                Phi = BasisSpline(n, a, b)
                Phi.Phi(x)
                Phi(x)

        Calling an instance directly (as in the last line) is equivalent to calling the interpolation method.
        """
        n = self.n[i]
        a = self.a[i]
        b = self.b[i]
        breaks = self.breaks[i]

        if order is None:
            order = 0

        orderIsScalar = np.isscalar(order)
        order = np.atleast_1d(order).flatten()

        nn = n + np.maximum(0, -np.min(order))

        # Check for x argument
        xIsProvided = (x is not None)
        x = x.flatten() if xIsProvided else self._nodes[i]
        nx = x.size

        # Determine the maximum index of the breakpoints that are less than or equal to x,
        # (if x=b, use the index of the next to last breakpoint).

        if self.opts.nodetype == 'canonical':
            ind = np.fix((x - a) * ((n - 1) / (b - a))).astype(int)
            np.maximum(ind, 0, ind)
            np.minimum(ind, n - 2, ind)
        else:
            ind = self._lookup(breaks, x)

        z = (x - breaks[ind]) / (breaks[ind + 1] - breaks[ind])
        hh = np.arange(nx)
        r_id = np.tile(hh, 2)
        c_id = np.concatenate((ind, ind + 1))
        data = np.concatenate(((1 - z), z))
        bas = csc_matrix((data, (r_id, c_id)), shape=(nx, n))

        # Compute Phi
        Phidict = dict()
        for ii in set(order):
            if ii == 0:
                Phidict[ii] = bas
            else:
                Phidict[ii] = np.dot(bas, self._diff(i, ii))

        Phi = np.array([Phidict[k] for k in order])
        return Phi

