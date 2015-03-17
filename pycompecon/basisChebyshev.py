__author__ = 'Randall'
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators

class BasisChebyshev:
    """
        A univariate Chebyshev basis with properties:
            * n:    scalar, number of nodes
            * a:    scalar, lower bound
            * b:    scalar, upper bound
            * nodes: d.1 vector, basis nodes
            * D: operators to compute derivative
            * I: operators to compute integrals
            * nodetype:  node type, i.e. 'gaussian','lobatto', or 'endpoint'
            * varName: 	string, variable name. Defaults to ''.
            * WarnOutOfBounds: boolean, warns if interpolating outside [a,b] if true. Defaults to 'false'.

        This class is based on code from Miranda and Fackler.
    """

    def __init__(self, n=3, a=-1.0, b=1.0, nodetype="gaussian", varName=""):
        """
        Creates an instance of a BasisChebyshev object

        :param float n: number of nodes (scalar > 2)
        :param float a: lower bound (scalar)
        :param float b: upper bound (scalar)
        :param str nodetype: type of collocation nodes, ('gaussian','lobatto', or 'endpoint')
        :param str varName: a string to name the variable
        :return: a BasisChebyshev instance
        """
        nodetype = nodetype.lower()

        self._n, self._a, self._b = n, a, b
        self._nodetype = nodetype.lower()
        self.varName = varName
        self._nodes = None
        self._D = []
        self._I = []
        self._reset()

    def _reset(self):
        self._validate()
        self._setNodes()
        self._D = []
        self._I = []

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
    Methods D and I return operators to differentiate/integrate, which are stored in _D and _I
    """

    def D(self, k=1):
        """
        Operator to differentiate

        :param k: order of differentiation
        :return: operator (matrix)
        """
        if len(self._D) < k:
            self._update_D(k)
        return self._D[k - 1]

    def _update_D(self, order=1):
        """
        Updates the list _D of differentiation operators

        :param order: order of required derivative
        :return: None
        """

        if len(self._D) >= order:
            return  # Use previously stored values if available

        n, a, b = self._n, self._a, self._b

        if n - order < 2:
            # todo:  use warning about this change
            order = n - 2

        if len(self._D) == 0:
            hh = np.arange(n) + 1
            jj, ii = np.meshgrid(hh, hh)
            rc = np.logical_and(np.asarray((ii + jj) % 2, bool), jj > ii)
            d = np.zeros([n, n])
            d[rc] = (4 / (b - a)) * (jj[rc] - 1)
            d[0, :] = d[0, :] / 2
            # todo: convert d to sparse matrix
            d = csc_matrix(d[:-1, :])
            self._D.append(d)
        else:
            d = self._D[0]

        while len(self._D) < order:
            h = len(self._D)
            dd = d[:n - h - 1, :n - h]
            self._D.append(dd * self._D[-1])

    def I(self, k=1):
        """
        Operator to integrate

        :param k: order of integration
        :return: operator (matrix)
        """
        if len(self._I) < k:
            self._update_I(k)
        return self._I[k - 1]

    def _update_I(self, order=1):
        """
        Updates the list _I of integration operators

        :param order: order of required integral
        :return: None
        """

        if len(self._I) >= order:
            return  # Use previously stored values if available

        n, a, b = self._n, self._a, self._b
        nn = n + order
        ii = np.array([(0.25 * (b - a)) / k for k in range(1, nn + 1)])
        d = np.diag(ii) - np.diag(ii[:-2], 2)
        # todo: make d sparse
        d[0, 0] *= 2
        d0 = np.array([(-1) ** k for k in range(nn)]) * sum(d)
        d0.resize((1, d0.size))  # need to have 2 dimensions to concatenate with d
        dd = np.mat(np.r_[d0, d])

        while len(self._I) < order:
            h = len(self._I)
            if h > 0:
                self._I.append(dd[:n + h + 1, :n + h] * self._I[-1])
            else:
                self._I.append(dd[:n + 1, :n])

    """
        Interpolation methods
    """
    def interpolation(self, x=None, order=[0]):
        """
        Computes interpolation matrices for given data x and order of differentiation 'order' (integration if negative)

        :param x:  evaluation points (defaults to nodes)
        :param order: a list of orders for differentiation (+) / integration (-)
        :return a: dictionary with interpolation matrices, keys given by unique elements of order.
        """
        n, a, b = self._n, self._a, self._b
        nn = n + max(0,-min(order))

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
        Phi = dict()
        for ii in set(order):

            if ii > 0: # take derivative
                Phi[ii] = bas[:, :n - ii] * self.D(ii)
            elif ii < 0:
                Phi[ii] = bas[:, :n - ii] * self.I(-ii)
            else:
                Phi[ii] = bas
        return Phi

    """
        SETTERS AND GETTERS:  these methods update the basis if n,a,b or nodetype are changed
    """

    @property
    def n(self):
        """ :return: number of nodes """
        return self._n

    @property
    def a(self):
        """ :return: lower bound """
        return self._a

    @property
    def b(self):
        """ :return: upper bound """
        return self._b

    @property
    def nodes(self):
        """ :return: basis nodes """
        return self._nodes

    @property
    def nodetype(self):
        """ :return: type of node ('gaussian','lobatto','endpoint')"""
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

    def plot(self, k=5):
        """
        Plots the first k basis functions

        :param k: number of functions to include in plot
        :return: None
        """
        a, b, = self._a, self._b
        x = np.linspace(a, b, 120)
        y = self.interpolation(x)
        plt.plot(x.T, y[0][:, :k])
        plt.plot(self.nodes,0*self.nodes,'ro')
        return None
