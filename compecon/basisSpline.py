import warnings
import numpy as np
from scipy.sparse import csc_matrix, diags, tril
from numba import jit, float64, void
import matplotlib.pyplot as plt
from compecon.tools import Options_Container

__author__ = 'Randall'
# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators


class OptionsSpline(Options_Container):
    NodeTypes = ['cardinal', 'user']

    def __init__(self, k=3, nodetype='cardinal', label='V0', tol=0.02, warn=True):
        self.k = k
        self.nodetype = nodetype.lower()
        self.label = label
        self.tol = tol
        self.warn = warn


class BasisSpline(Options_Container):
    def __init__(self, *args, **kwargs):
        self.opts = OptionsSpline(**kwargs)

        nargs = len(args)
        if nargs == 1:
            breaks = np.array(args[0])
            breaks.sort()
            assert breaks.size > 1, 'breakpoint sequence must contain at least two elements'
            self.breaks = breaks
            self.opts.nodetype = 'user'
        elif nargs == 3:
            n, a, b = args
            k = self.opts.k
            assert a < b, 'lower bound must be less than upper bound'
            assert n > k, 'number of nodes must exceed order of spline'
            self.breaks = np.linspace(a, b, n + 1 - k)
        else:
            txt = 'Either 1 or 3 positional arguments must be provided\n'
            txt += '\t1 argument -> break points\n'
            txt += '\t3 argument -> n, a, b'
            raise ValueError(txt)

        self._reset()

    @property
    def a(self):
        return self.breaks[0]

    @property
    def b(self):
        return self.breaks[-1]

    @property
    def k(self):
        return self.opts.k

    @property
    def n(self):
        return self.breaks.size + self.k - 1

    def _setNodes(self):
            """
            Sets the basis nodes

            :return: None
            """
            n, a, b, k = self['n', 'a', 'b', 'k']
            x = np.cumsum(self._augbreaks(k))
            x = (x[k : n + k] - x[:n]) / k
            x[0] = a
            x[-1] = b
            self.nodes = x

    def _reset(self):
        # self._validate()
        self._setNodes()
        self._Diff = dict()

    """
    Method Diff return operators to differentiate/integrate, which are stored in _Diff
    """

    def _augbreaks(self, m):
        aa, bb = [np.repeat(h, m) for h in self['a', 'b']]
        return np.concatenate((aa, self.breaks, bb))


    def _Diff_(self, k):
        """
        Operator to differentiate

        :param k: order of differentiation
        :return: operator (matrix)
        """
        if k not in self._Diff.keys():
            self._update_Diff(k)
        return self._Diff[k]

    def _update_Diff(self, order=1):
        """
        Updates the list _D of differentiation operators

        :param order: order of required derivative
        :return: None
        """
        if order in self._Diff.keys() or order == 0:
            return  # Use previously stored values if available

        n, a, b, k = self['n', 'a', 'b', 'k']
        assert order <= k, 'order must be less or equal to k'
        keys = set(self._Diff.keys())
        kk = k - 1 - min(order, 0)
        augbreaks = self._augbreaks(kk)

        if order > 0:
            def sptemp(j):
                temp = np.atleast_2d((k + 1 - j) / (augbreaks[k:(n + k - j)] - augbreaks[(j - 1):(n - 1)]))
                return diags((-temp, temp), [0, 1], (n - j, n + 1 - j))

            missing_keys = set(range(1, order + 1)) - keys

            if 1 in missing_keys:
                self._Diff[1] = sptemp(1)
                missing_keys -= {1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=True)
            while missing_keys:
                j = missing_keys.pop()
                self._Diff[j] = np.dot(sptemp(j), self._Diff[j - 1])
        else:
            def sptemp(j):
                temp = (augbreaks[(kk + 1):(kk + n - j)] -
                        augbreaks[(kk - k + j + 1):(kk + n - k)]) / (k - j)

                return tril(np.tile(temp, (n - j, 1)), -1)

            missing_keys = set(range(order, 0)) - keys

            if -1 in missing_keys:
                self._Diff[-1] = sptemp(-1)
                missing_keys -= {-1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=False)
            while missing_keys:
                j = missing_keys.pop()
                self._Diff[j] = sptemp(j) * self._Diff[j + 1]

    @staticmethod
    def lookup(table, x):
        # TODO: add parameter endadj -> in Mario's code it always has value=3
        # Here, I'm assuming that's the only case
        ind = np.searchsorted(table, x, 'right')
        ind[ind == 0] = (table == table[0]).sum()
        ind[ind >= table.size] = ind[-1] - (table == table[-1]).sum()
        return ind - 1

    """
        Interpolation methods
    """
    def interpolation(self, x=None, order=0):
        """
        Computes interpolation matrices for given data x and order of differentiation 'order' (integration if negative)

        :param x:  evaluation points (defaults to nodes)
        :param order: a list of orders for differentiation (+) / integration (-)
        :return a: dictionary with interpolation matrices, keys given by unique elements of order.

        Example: Create a basis with 5 nodes, get the interpolation matrix evaluated at 20 points::

                n, a, b = 5, 0, 4
                x = numpy.linspace(a,b, 20)
                Phi = BasisSpline(n, a, b)
                Phi.interpolation(x)
                Phi(x)

        Calling an instance directly (as in the last line) is equivalent to calling the interpolation method.
        """
        n, a, b, k = self['n', 'a', 'b', 'k']

        orderIsScalar = np.isscalar(order)
        order = np.atleast_1d(order).flatten()
        assert np.max(order) < k, 'Derivatives defined for order less than k'

        nn = n + np.maximum(0, -np.min(order))

        # Check for x argument
        xIsProvided = (x is not None)
        x = x.flatten() if xIsProvided else self.nodes
        nx = x.size

        minorder = np.min(order)
        kaug = k - minorder
        augbreaks = self._augbreaks(kaug)
        ind = self.lookup(augbreaks, x)

        # Recursively determine the values of a k-order basis matrix.
        # This is placed in an (m x k+1-order) matrix
        bas = np.zeros((kaug + 1, nx))
        bas[0] = 1
        Phi = np.array([csc_matrix((nx, n), dtype=np.float) for h in range(order.size)])

        for j in range(1, kaug + 1):
            for jj in range(j, 0, -1):
                b0 = augbreaks[ind + jj - j]
                b1 = augbreaks[ind + jj]
                temp = bas[jj - 1] / (b1 - b0)
                bas[jj] = (x - b0) * temp + bas[jj]
                bas[jj - 1] = (b1 - x) * temp

            # as now contains the order j spline basis
                ii = np.where((k - j) == order)[0]
                if ii.size > 0:
                    ii = ii[0]
                    oi = order[ii]
                    # Put values in appropriate columns of a sparse matrix
                    r = np.tile(np.arange(nx), k - oi + 1)
                    c = np.atleast_2d(np.arange(oi - k, 1)).T + np.atleast_2d(ind)
                    c = (c - (oi - minorder)).flatten()
                    data = bas[:k - oi + 1].flatten()
                    Phi[ii] = csc_matrix((data, (r, c)), (nx, n-oi))

                    if oi:
                        # If needed compute derivative or anti-derivative
                        Phi[ii] = np.dot(Phi[ii], self._Diff_(oi))

        if orderIsScalar:
            return Phi[0]
        else:
            return Phi




    """
    Display output for the basis
    """

    def __repr__(self):
        """
        Creates a description of the basis
        :return: string (description)
        """
        n, a, b = self['n', 'a', 'b']
        bstr = "A Spline basis function of order {:d}:  ".format(self.k)
        bstr += "using {:d} {} nodes in [{:.2f}, {:.2f}]".format(n, self.opts.nodetype.upper(), a, b)
        return bstr

    def plot(self, order=0, m=None):
        """
        Plots the first k basis functions

        :param order: order of differentiation
        :param k: number of functions to include in plot
        :return: a plot
        """
        a, b, = self['a', 'b']
        if m is None:
            m = self.n

        nodes = self.nodes
        x = np.linspace(a, b, 120)
        y = self(x, order)[:, :m].toarray()
        x.resize((x.size, 1))
        plt.plot(x, y)

        plt.plot(nodes, 0 * nodes, 'ro')
        plt.xlim(a, b)
        plt.show()

    """
    Calling the basis directly returns the interpolation matrix
    """
    def __call__(self, x=None, order=0):
        """
        Equivalent to self.interpolation(x, order)
        """
        return self.interpolation(x, order)
