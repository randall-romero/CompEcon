import warnings
import numpy as np
from scipy.sparse import csc_matrix
from numba import jit, float64, void
from compecon import Basis

__author__ = 'Randall'
# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators


class BasisChebyshev(Basis):
    def __init__(self, n, a, b, **kwargs):
        """
        Creates an instance of a BasisChebyshev object

        :param int n: number of nodes
        :param float a: lower bound
        :param float b: upper bound
        :param str nodetype: type of collocation nodes, ('gaussian','lobatto', or 'endpoint')
        :param str varName: a string to name the variable
        """

        kwargs['basistype'] = 'chebyshev'
        super().__init__(n, a, b, **kwargs)
        self._set_nodes()

    def _set_nodes(self):
        """
        Sets the basis nodes

        :return: None
        """
        nodetype = self.opts.nodetype

        for i in range(self.d):
            n = self.n[i]
            a = self.a[i]
            b = self.b[i]

            if nodetype in ['gaussian', 'endpoint']:
                x = np.array([-np.cos(np.pi * k / (2 * n)) for k in range(1, 2 * n, 2)])
            elif nodetype == 'lobatto':
                x = np.array([-np.cos(np.pi * k / (n - 1)) for k in range(n)])
            elif nodetype == 'uniform':
                x = np.linspace(-1, 1, n)
            else:
                raise Exception('Unknown node type')

            if nodetype == 'endpoint':
                x /= x[-1]

            x *= (b - a) / 2
            x += (b + a) / 2
            self._nodes.append(x)
        self._expand_nodes()

    def _rescale201(self, i, x):
        """
        Rescales nodes from [a, b] domain to [-1, 1] domain

        :param x: nodes in [a, b] domain (array)
        :return: nodes in [-1, 1] domain
        """
        n = self.n[i]
        a = self.a[i]
        b = self.b[i]

        # if not(a <= min(x) <= max(x) <= b):
        # warnings.warn('x values must be between a and b.')
        return (2 / (b - a)) * (x - (a + b) / 2)

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
        a = self.a[i]
        b = self.b[i]

        if order > 0:
            if order > n - 2:
                warnings.warn('order must be less or equal to n - 2; setting order = n - 2')
                order = n - 2

            missing_keys = set(range(1, order + 1)) - keys

            if 1 in missing_keys:
                hh = np.arange(n) + 1
                jj, ii = np.meshgrid(hh, hh)
                rc = np.logical_and(np.asarray((ii + jj) % 2, bool), jj > ii)
                d = np.zeros([n, n])
                d[rc] = (4 / (b - a)) * (jj[rc] - 1)
                d[0, :] = d[0, :] / 2
                # todo: convert d to sparse matrix
                d = csc_matrix(d[:-1, :])
                self._diff_operators[i][1] = d
                missing_keys -= {1}
            else:
                d = self._diff_operators[i][1]

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=True)
            while missing_keys:
                k = missing_keys.pop()
                self._diff_operators[i][k] = d[:n - k, :n - k + 1] * self._diff_operators[i][k - 1]
        else:

            nn = n - order
            ii = np.array([(0.25 * (b - a)) / k for k in range(1, nn + 1)])
            d = np.diag(ii) - np.diag(ii[:-2], 2)
            # todo: make d sparse
            d[0, 0] *= 2
            d0 = np.array([(-1) ** k for k in range(nn)]) * sum(d)
            d0.resize((1, d0.size))  # need to have 2 dimensions to concatenate with d
            dd = np.mat(np.r_[d0, d])

            missing_keys = set(range(order, 0)) - keys

            if -1 in missing_keys:
                self._diff_operators[i][-1] = dd[:n + 1, :n]
                missing_keys -= {-1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=False)
            while missing_keys:
                k = missing_keys.pop()
                self._diff_operators[i][k] = dd[:n - k, :n - k - 1] * self._diff_operators[i][k + 1]

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
                Phi = BasisChebyshev(n, a, b)
                Phi.Phi(x)
                Phi(x)

        Calling an instance directly (as in the last line) is equivalent to calling the interpolation method.
        """
        if order is None:
            order = 0

        orderIsScalar = np.isscalar(order)
        order = np.atleast_1d(order).flatten()

        n = self.n[i]
        nn = n + np.maximum(0, -np.min(order))

        # Check for x argument
        xIsProvided = (x is not None)
        x = np.asarray(x).flatten() if xIsProvided else self._nodes[i]
        nx = x.size

        # Compute order 0 interpolation matrix
        if xIsProvided:
            bas = np.zeros([nx, nn])
            z = self._rescale201(i, x)
            cheby_polynomials(z, bas)
        else:
            z = np.atleast_2d(np.arange(n - 0.5, -0.5, -1)).T
            bas = np.cos((np.pi / n) * z * np.arange(0, nn))

        # Compute Phi
        Phidict = dict()
        for ii in set(order):
            if ii == 0:
                Phidict[ii] = bas
            else:
                Phidict[ii] = bas[:, :n - ii] * self._diff(i, ii)  # as matrix multiplication, because diff is sparse

        Phi = np.array([Phidict[k] for k in order])
        return Phi


@jit(void(float64[:], float64[:, :]), nopython=True)
def cheby_polynomials(z, bas):
    for node in range(z.size):
        bas[node, 0] = 1
        bas[node, 1] = z[node]
        z[node] *= 2
        for k in range(2, bas.shape[1]):
            bas[node, k] = z[node] * bas[node, k - 1] - bas[node, k - 2]
    return None


