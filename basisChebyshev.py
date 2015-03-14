__author__ = 'Randall'


# TODO: complete this class, this is just the beginning!!

import numpy as np


class BasisChebyshev:
    """
        BasisChebyshev class

        Defines a class to represent a univariate Chebyshev basis.

        Objects of class |basisChebyshev| have the following properties:
            n:    scalar, number of nodes
            a:    scalar, lower bound
            b:    scalar, upper bound
            nodes: d.1 vector, basis nodes
            D: operators to compute derivative
            I: operators to compute integrals
            nodetype:  node type, i.e. 'gaussian','lobatto', or 'endpoint'
            varName: 	string, variable name. Defaults to ''.
            WarnOutOfBounds: boolean, warns if interpolating outside [a,b] if true. Defaults to 'false'.

    This class is based on code from Miranda and Fackler.
    """

    def __init__(self, n=3, a=-1.0, b=1.0, nodetype="gaussian", varName=""):
        """
        Creates an instance of a BasisChebyshev object
        :param n: number of nodes (scalar > 2)
        :param a: lower bound (scalar)
        :param b: upper bound (scalar)
        :param nodetype: type of collocation nodes, ('gaussian','lobatto', or 'endpoint')
        :param varName: a string to name the variable
        :return: a BasisChebyshev instance
        """
        nodetype = nodetype.lower()

        # Validate the inputs
        # todo: find Matlab equivalent isscalar()
        if not n > 2:
            raise Exception('n must be at least 3')
        if not a < b:
            raise Exception('a must be less than b')
        if not nodetype in ["gaussian", "endpoint", "lobatto"]:
            raise Exception("nodetype must be 'gaussian', 'endpoint', or 'lobatto'.")

        self.n, self.a, self.b = n, a, b
        self.nodetype = nodetype
        self.varName = varName
        self.nodes = None
        self.setNodes()
        self.D = []
        self.I = []

    def setNodes(self):
        """
        Sets the basis nodes
        :return: None
        """
        n = self.n

        if self.nodetype in ['gaussian', 'endpoint']:
            x = np.array([-np.cos(np.pi * k / (2 * n)) for k in range(1, 2 * n, 2)])
        elif self.nodetype == 'lobatto':
            x = np.array([-np.cos(np.pi * k / (n - 1)) for k in range(n)])
        else:
            raise Exception('Unknown node type')

        if self.nodetype == 'endpoint':
            x /= x[-1]

        self.nodes = self.rescale2ab(x)

    def rescale2ab(self, x):
        """
        Rescales nodes from [-1,1] domain to [a,b] domain
        :param x: nodes in [-1,1] domain (array)
        :return: None
        """
        n, a, b = self.n, self.a, self.b
        if n != len(x) or min(x) < -1 or max(x)> 1:
            raise Exception('x must have {} nodes between -1 and 1.'.format(n))

        return (a + b + (b - a) * x) / 2

    def update_D(self, order=1):
        """
        Updates the list D of differentiation operators
        :param order: order of required derivative
        :return: None
        """

        if len(self.D) >= order:
            return  # Use previously stored values if available

        n, a, b = self.n, self.a, self.b

        if n - order < 2:
            # todo:  use warning about this change
            order = n - 2

        if len(self.D) == 0:
            hh = np.arange(n) + 1
            jj, ii = np.meshgrid(hh, hh)
            rc = np.logical_and(np.asarray((ii + jj) % 2, bool), jj > ii)
            d = np.zeros([n, n])
            d[rc] = (4 / (b - a)) * (jj[rc] - 1)
            d[0, :] = d[0, :] / 2
            # todo: convert d to sparse matrix
            d = d[:-1, :]
            self.D.append(d)
        else:
            d = self.D[0]

        while len(self.D) < order:
            h = len(self.D)
            dd = d[:n-h-1, :n-h]
            self.D.append(np.dot(dd, self.D[-1]))

    def update_I(self, order=1):
        """
        Updates the list I of integration operators
        :param order: order of required integral
        :return: None
        """

        if len(self.I) >= order:
            return  # Use previously stored values if available

        n, a, b = self.n, self.a, self.b
        nn = n + order
        ii = np.array([(0.25 * (b - a)) / k for k in range(1, nn + 1)])
        d = np.diag(ii) - np.diag(ii[:-2], 2)
        # todo: make d sparse
        d[0, 0] *= 2
        d0 = np.array([(-1) ** k for k in range(nn)]) * sum(d)
        d0.resize((1, d0.size))  # need to have 2 dimensions to concatenate with d
        dd = np.r_[d0, d]

        while len(self.I) < order:
            h = len(self.I)
            if h > 0:
                self.I.append(np.dot(dd[:n + h + 1, :n + h], self.I[-1]))
            else:
                self.I.append(dd[:n + 1, :n])

    def __repr__(self):
        """
        Creates a description of the basis
        :return: string (description)
        """
        n, a, b, nodetype = self.n, self.a, self.b, self.nodetype
        bstr = "A Chebyshev basis function:  "
        bstr += "using {:d} {} nodes in [{:6.2f}, {:6.2f}]".format(n, nodetype.upper(), a, b)
        return bstr