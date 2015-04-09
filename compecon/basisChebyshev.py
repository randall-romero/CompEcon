__author__ = 'Randall'
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators

class BasisChebyshev(object):
    def __init__(self, n=3, a=-1.0, b=1.0, nodetype="gaussian", varName=""):
        """
        Creates an instance of a BasisChebyshev object

        :param int n: number of nodes
        :param float a: lower bound
        :param float b: upper bound
        :param str nodetype: type of collocation nodes, ('gaussian','lobatto', or 'endpoint')
        :param str varName: a string to name the variable
        """
        nodetype = nodetype.lower()

        self._n, self._a, self._b = n, a, b
        self._nodetype = nodetype.lower()
        self.varName = varName
        self._nodes = None
        self._Diff = dict()
        self._reset()

    def _reset(self):
        self._validate()
        self._setNodes()
        self._Diff = dict()

    def _validate(self):
        """
        Validates values of n, a, b, nodetype

        :return: None
        """
        n, a, b, nodetype = self._n, self._a, self._b, self._nodetype
        if not n > 2:
            raise Exception('n must be at least 3')
        if not a < b:
            raise Exception('a must be less than b')
        if not nodetype in ["gaussian", "endpoint", "lobatto"]:
            raise Exception("nodetype must be 'gaussian', 'endpoint', or 'lobatto'.")

    def _setNodes(self):
        """
        Sets the basis nodes

        :return: None
        """
        n = self._n

        if self.nodetype in ['gaussian', 'endpoint']:
            x = np.array([-np.cos(np.pi * k / (2 * n)) for k in range(1, 2 * n, 2)])
        elif self.nodetype == 'lobatto':
            x = np.array([-np.cos(np.pi * k / (n - 1)) for k in range(n)])
        else:
            raise Exception('Unknown node type')

        if self.nodetype == 'endpoint':
            x /= x[-1]

        self._nodes = self._rescale2ab(x)

    """
        Methods to standardize and rescale the nodes
    """

    def _rescale2ab(self, x):
        """
        Rescales nodes from [-1,1] domain to [a,b] domain

        :param x: nodes in [-1,1] domain (array)
        :return: nodes in [a, b] domain
        """
        n, a, b = self._n, self._a, self._b
        if n != len(x) or min(x) < -1 or max(x) > 1:
            raise Exception('x must have {} nodes between -1 and 1.'.format(n))

        return (a + b + (b - a) * x) / 2

    def _rescale201(self,x):
        """
        Rescales nodes from [a, b] domain to [-1, 1] domain

        :param x: nodes in [a, b] domain (array)
        :return: nodes in [-1, 1] domain
        """
        n, a, b = self._n, self._a, self._b
        # if min(x) < a or max(x) > b:
        #     raise Exception('x values must be between a and b.')  # todo: turn this into a warning!
        return (2 / (b-a)) * (x - (a + b) /2)



    """
    Method Diff return operators to differentiate/integrate, which are stored in _Diff
    """

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

        n, a, b = self._n, self._a, self._b
        keys = set(self._Diff.keys())

        if order > 0:
            if order > n - 2:
                # todo:  use warning about this change
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
                self._Diff[1] = d
                missing_keys -= {1}
            else:
                d = self._Diff[1]

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=True)
            while missing_keys:
                k = missing_keys.pop()
                self._Diff[k] = d[:n - k, :n - k + 1] * self._Diff[k - 1]
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
                self._Diff[-1] = dd[:n + 1, :n]
                missing_keys -= {-1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=False)
            while missing_keys:
                k = missing_keys.pop()
                self._Diff[k] = dd[:n - k, :n - k - 1] * self._Diff[k + 1]


    """
        Interpolation methods
    """
    def interpolation(self, x=None, order=None):
        """
        Computes interpolation matrices for given data x and order of differentiation 'order' (integration if negative)

        :param x:  evaluation points (defaults to nodes)
        :param order: a list of orders for differentiation (+) / integration (-)
        :return a: dictionary with interpolation matrices, keys given by unique elements of order.

        Example: Create a basis with 5 nodes, get the interpolation matrix evaluated at 20 points::

                n, a, b = 5, 0, 4
                x = numpy.linspace(a,b, 20)
                Phi = BasisChebyshev(n, a, b)
                Phi.interpolation(x)
                Phi(x)

        Calling an instance directly (as in the last line) is equivalent to calling the interpolation method.
        """
        if order is None:  #REVISAR ESTO!!!
            order = 0

        orderIsScalar = np.isscalar(order)
        order = np.atleast_1d(order)

        n, a, b = self._n, self._a, self._b
        nn = n + max(0, -min(order))

        # Check for x argument
        xIsProvided = (x is not None)
        x = np.mat(x) if xIsProvided else self._nodes
        nx = x.size

        # Compute order 0 interpolation matrix
        if xIsProvided:
            z = self._rescale201(x)
            bas = np.mat(np.zeros([nn,nx]))
            bas[0] = 1
            bas[1] = z
            z *=2
            for k in range(2, nn):
                bas[k] = np.multiply(z, bas[k - 1]) - bas[k - 2]
            bas = bas.T
        else:
            z = np.mat(np.arange(n - 0.5, -0.5, -1)).T
            bas = np.cos((np.pi/n) * z * np.mat(np.arange(0, nn)))

        # Compute Phi
        Phidict = dict()
        for ii in set(order):
            if ii == 0:
                Phidict[ii] = bas
            else:
                Phidict[ii] = bas[:, :n - ii] * self._Diff_(ii)

        Phi = np.array([Phidict[k] for k in order])

        if orderIsScalar:
                return Phi[0]
        else:
            return Phi





    """
        SETTERS AND GETTERS:  these methods update the basis if n,a,b or nodetype are changed
    """

    @property
    def n(self):
        """ number of nodes (int)"""
        return self._n

    @property
    def a(self):
        """ lower bound (float)"""
        return self._a

    @property
    def b(self):
        """ upper bound (float)"""
        return self._b

    @property
    def nodes(self):
        """ basis nodes """
        return self._nodes

    @property
    def nodetype(self):
        """ type of node ('gaussian','lobatto','endpoint')"""
        return self._nodetype

    @n.setter
    def n(self, val):
        self._n = val
        self._reset()

    @a.setter
    def a(self, val):
        self._a = val
        self._reset()

    @b.setter
    def b(self, val):
        self._b = val
        self._reset()

    @nodetype.setter
    def nodetype(self, val):
        self._nodetype = val.lower()
        self._reset()

    """
    Display output for the basis
    """

    def __repr__(self):
        """
        Creates a description of the basis
        :return: string (description)
        """
        n, a, b, nodetype = self._n, self._a, self._b, self._nodetype
        bstr = "A Chebyshev basis function:  "
        bstr += "using {:d} {} nodes in [{:6.2f}, {:6.2f}]".format(n, nodetype.upper(), a, b)
        return bstr

    def plot(self, order=0, k=5):
        """
        Plots the first k basis functions

        :param order: order of differentiation
        :param k: number of functions to include in plot
        :return: a plot
        """
        a, b, = self._a, self._b
        nodes = self._nodes
        x = np.linspace(a, b, 120)
        y = self(x, order)
        x.resize((x.size, 1))
        plt.plot(x, y[:, :k])

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