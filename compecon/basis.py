from warnings import warn
import numpy as np
import scipy as sp
from .tools import gridmake, Options_Container
import matplotlib.pyplot as plt
from functools import reduce
from scipy.sparse import csc_matrix
import copy

__author__ = 'Randall'


class Basis(object):
    """ A class for function interpolation.

    The CompEcon toolbox includes the classes BasisChebyshev, BasisSpline, and BasisLinear for function interpolation.
    All of them inherit methods from this class to deal with common functionality.

    Attributes:
        d (int): Number of dimensions of the basis space.
        n (numpy array): Number of nodes for each dimension (d-array).
        a (numpy array): Lower bounds for each dimension (d-array).
        b (numpy array): Upper bound for each dimension (d-array).
        N (int): Total number of nodes.
        M (int): Total number of polynomials.
        nodes (numpy array): Basis nodes (d.N array).
        x (numpy array): Same as nodes.
        y (numpy array):  Value of interpolated function(s) at basis nodes (s0...sk.N array).
        c (numpy array):  Coefficients of interpolated function(s) at basis nodes (s0...sk.M array).
        shape (tuple): Dimensions of interpolated function, i.e. (s0, s1, ..., sk).
        shape_N (tuple): Shape of y. Same as shape, N.
        size (int): Total number of interpolated functions, i.e. s0 x ... x sk.
        ndim (int): Number of dimension of interpolated functions, i.e.  k+1.
        opts (BasisOptions): Options for basis (see BasisOptions class for details).
        _diff_operators (list): Operators to differentiate/integrate (a list of d dictionaries).
        _Phi (numpy array): Interpolation array (basis functions evaluated at nodes, N.M array).
        _PhiT (numpy array): Transpose of _Phi (stored to updated y given c).
        _PhiInvT (numpy array): Inverse of _PhiT (stored to update c given y).
        _cIsOutdated (numpy array): Boolean array indicating which coefficients are outdated (following a change in y).
        _yIsOutdated (numpy array): Boolean array indicating which function values are outdated (following a change in c).


    Methods:
        Phi: Computes interpolation matrix at arbitrary values.
        plot: Plots interpolation basis functions.
        copy: Returns a shallow copy of a Basis instance.
        duplicate: Similar to copy, but allows to specify new functions to be interpolated.
        update_y: Update function values (called after changing c).
        update_c: Update interpolation coefficients (called after changing y).
        _phi1d: Computes interpolation matrix for a given basis dimension.
        _diff: Returns a specified differentiation operator.
        _update_diff_operators: Compute differentiation operators, storing them in _diff_operators.
        _set_function_values: Sets values of y and c attributes.
        _expand_nodes: Compute the nodes grid (required when d > 1).
        _lookup: find nearest node to given point (required by spline and linear bases).
        __init__: Return new instance of Basis. Should be called directly by Basis subclasses only.
        __repr__: String representation of a Basis contents.
        __getitem__: Return a copy of Basis for specified item.
        __setitem__: Modify the y values for a speficied item.
        __call__: Evaluate the interpolated function at arbitrary values.


    """

    def __init__(self, n, a, b, **kwargs):
        n, a, b = np.broadcast_arrays(*np.atleast_1d(n, a, b))
        assert np.all(n > 2), 'n must be at least 3'
        assert np.all(a < b), 'lower bound must be less than upper bound'
        self.d = n.size
        self.n = n
        self.a = a
        self.b = b
        self.opts = BasisOptions(self.d, **kwargs)
        self.opts.expandGrid(n)
        self.N = self.opts.ix.shape[1]  # Total number of nodes
        self.M = self.opts.ip.shape[1]  # Total number of polynomials

        self._nodes = list()
        self._diff_operators = [dict() for h in range(self.d)]
        self._PhiT = None



    ''' This methods are declared here, but implemented in children classes '''
    def _phi1d(self, i, x=None, order=0):
        pass

    def _update_diff_operators(self, i, m):
        pass

    """
    Method _diff return operators to differentiate/integrate, which are stored in _diff_operators
    """

    def _set_function_values(self):

        y, s, f, c, l = self.opts['y', 's', 'f', 'c', 'l']

        y_provided = True

        if c is not None:
            c = np.atleast_2d(c)
            y_provided = False
        elif callable(f):
            y = np.atleast_2d(f(self.nodes))
        elif y is not None:
            y = np.atleast_2d(y)
        elif l is not None:
            if isinstance(l[0], str):
                l = [l]
            s = [len(a) for a in l]  # assumes that l is a list of list or tuples
            s = np.r_[np.atleast_1d(s), self.N]
            y = np.zeros(s)
            self.opts.ylabels = l
        else:
            s = np.r_[np.atleast_1d(s), self.N]
            y = np.zeros(s)

        s = y.shape[:-1] if y_provided else c.shape[:-1]
        self._c = np.zeros(np.r_[s, self.M])
        self._y = np.zeros(np.r_[s, self.N])
        self._cIsOutdated = np.full(s, True, bool)
        self._yIsOutdated = np.full(s, True, bool)

        if y_provided:
            self.y = y
        else:
            self.c = c

        self.opts.clear_data()

    def _expand_nodes(self):
        ix = self.opts.ix
        self.nodes = np.array([self._nodes[k][ix[k]] for k in range(self.d)])
        phi = self.Phi()
        self._PhiT = phi.T
        if self.opts.basistype is 'chebyshev':
            self._PhiInvT = np.linalg.pinv(phi).T
        else:
            self._PhiInvT = sp.sparse.linalg.inv(phi).T
        self._set_function_values()

    @property
    def _Phi(self):
        return self._PhiT.T

    def _diff(self, i, m):
        """
        Operator to differentiate

        :param k: order of differentiation
        :return: operator (matrix)
        """
        if m not in self._diff_operators[i].keys():
            self._update_diff_operators(i, m)
        return self._diff_operators[i][m]

    def Phi(self, x=None, order=None, dropdim=True):
        """Compute the interpolation matrix :math:`\Phi(x)`

        :param np.array x: evaluation points
        :param np.array order: order of derivatives (integrates if negative)
        :return: Interpolation matrix

        Example::

                n, a, b = [9, 9], [0, 0], [5, 7]
                Phi = Basis(n, a, b, method='smolyak', qn = 3, qp = 3)
                Phi.Phi()
                Phi()                                       # same as previous line
                Phi(order = [[2, 1, 1, 0], [0, 1, 1, 2]])   # Hessian of interpolation matrix

        Calling an instance directly (as in the last two lines) is equivalent to calling the interpolation method.
        """
        if np.all([x is None, order is None, self._PhiT is not None, dropdim == True]):
            return self._Phi

        if order is None:
            order = np.zeros([self.d, 1], 'int')
        else:
            order = np.atleast_1d(order)
            if order.ndim == 1:
                assert (order.size == self.d), 'order should have {:d} elements (one per dimension)'.format(self.d)
                order.shape = (self.d, 1)
            else:
                assert (order.shape[0] == self.d)

        orderIsScalar = order.shape[1] == 1

        if self.d == 1:
            phi = self._phi1d(0, x, order)
            return phi[0] if (orderIsScalar and dropdim) else phi

        ''' check what type of input x is provided, and get row- and column- indices '''
        if x is None:
            x = [None] * self.d
            r = self.opts.ix
            c = self.opts.ip
        elif type(x) == list:
            assert (len(x) == self.d)
            r = gridmake(*[np.arange(xi.size) for xi in x])
            c = self.opts.ip
        elif type(x) == np.ndarray:
            assert (x.shape[0] == self.d), 'x must have {:d} columns'.format(self.d)
            r = [np.arange(x.shape[1])] * self.d
            c = self.opts.ip
        else:
            raise Exception('wrong x input')

        ''' Call _phi1d method for each dimension, then reduce the matrix '''
        oo = np.arange(order.shape[1])

        if self.opts.basistype is 'chebyshev':
            PHI = (self._phi1d(k, x[k], order[k])[np.ix_(oo, r[k], c[k])] for k in range(self.d))
            phi = reduce(np.multiply, PHI)
        else:  # results come in sparse matrices
            PHI = np.ndarray((oo.size, self.d), dtype=csc_matrix)
            for k in range(self.d):
                phitemp = self._phi1d(k, x[k], order[k])
                for o in oo:
                    PHI[o, k] = phitemp[o][np.ix_(r[k], c[k])]

            phi = [reduce(lambda A, B: A.multiply(B), PHI[o]) for o in oo]

        return phi[0] if (orderIsScalar and dropdim) else phi


    @staticmethod
    def _lookup(table, x):  # required by spline and linear bases
        # TODO: add parameter endadj -> in Mario's code it always has value=3
        # Here, I'm assuming that's the only case
        ind = np.searchsorted(table, x, 'right')
        ind[ind == 0] = (table == table[0]).sum()
        ind[ind >= table.size] = ind[-1] - (table == table[-1]).sum()
        return ind - 1

    def __repr__(self):
        # if self.d == 1:
        #     return self._B1[0].__repr__()

        n, a, b = self.n, self.a, self.b
        nodetype = self.opts.nodetype.capitalize()
        basistype = self.opts.basistype
        vnames = self.opts.labels
        vnamelen = str(max(len(v) for v in vnames))

        if basistype is 'spline':
            term = ['linear', 'quadratic', 'cubic']
            if self.k < 4:
                basistype = term[self.k - 1] + ' spline'
            else:
                basistype = 'order {:d} spline'.format(self.k)

        bstr = "A {}-dimension {} basis:  ".format(self.d, basistype.capitalize())
        bstr += "using {:d} {} nodes and {:d} polynomials".format(self.N, nodetype, self.M)
        if self.d > 1:
            bstr += ", expanded by {}".format(self.opts.method)

        bstr += '\n' + '_' * 75 + '\n'
        for k in range(self.d):
            bstr += ("\t{:<" + vnamelen +"s}: {:d} nodes in [{:6.2f}, {:6.2f}]\n").format(vnames[k], n[k], a[k], b[k])

        bstr += '\n' + '=' * 75 + '\n'
        bstr += "WARNING! Class Basis is still work in progress"
        return bstr

    def plot(self, order=0, m=None, i=0, nx=120):
        """
        Plots the first k basis functions

        :param order: order of differentiation
        :param k: number of functions to include in plot
        :return: a plot
        """
        i = min(i, self.d-1)

        a = self.a[i]
        b = self.b[i]

        assert np.isscalar(order), 'order must be a scalar; plot only works with 1d bases'

        if m is None:
            m = self.n[i]

        nodes = self._nodes[i]
        x = np.linspace(a, b, nx)
        y = self._phi1d(i, x, order)[0][:, :m]
        if self.opts.basistype is 'spline':
            y = y.toarray()

        x.resize((x.size, 1))
        plt.plot(x, y)

        plt.plot(nodes, 0 * nodes, 'ro')
        plt.xlim(a, b)
        plt.xlabel(self.opts.labels[i])


        basistype = self.opts.basistype
        if basistype is 'spline':
            if self.k < 4:
                basistype = ['linear', 'quadratic', 'cubic'][self.k - 1] + ' spline'
            else:
                basistype = 'order {:d} spline'.format(self.k)

        plt.title('A {:s} basis with {:d} nodes'.format(basistype.title(), self.n[i]))
        plt.show()

    ''' <<<<<<<<<<<<<<<<<<<<<<<<REVISAR DESDE AQUI >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    @property
    def x(self):
        """  :return: interpolation nodes  """
        return self.nodes

    def update_y(self):
        # todo: try to update only the outdated values. Can't yet figure out how to index them
        # ii = self._yIsOutdated
        if self.opts.basistype is 'chebyshev':
            self._y = np.dot(self._c, self._PhiT)
        else:
            self._y = self._c * self._PhiT
        self._yIsOutdated = np.full(self._yIsOutdated.shape, False)

    def update_c(self):
        # todo: try to update only the outdated values. Can't yet figure out how to index them
        ii = self._cIsOutdated
        if self.opts.basistype is 'chebyshev':
            self._c = np.dot(self._y, self._PhiInvT)
        else:
            try:
                self._c = self._y * self._PhiInvT
            except:
                self._c = np.dot(self._y, self._PhiInvT.toarray())

        self._cIsOutdated = np.full(self._cIsOutdated.shape, False)

    @property
    def shape(self):
        if np.any(self._yIsOutdated):
            return self._c.shape[:-1]
        else:
            return self._y.shape[:-1]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape) - 1

    @property
    def shape_N(self):
        temp = list(self.shape)
        temp.append(self.N)
        return temp

    def copy(self):
        return copy.copy(self)

    def duplicate(self, y=None, c=None, f=None, s=None, l=None):
        """ Duplicate the basis

        Makes a shallow copy of the basis, allowing the new instance to have its own
        approximation coefficients while sharing the nodes and interpolation matrices

        New instance values are specified with the same arguments used to create a new Basis,
        if none is provided, then it copies original coefficients by simply getting all iterms:
        copy_instance = original_instance[:]



        """
        nfunckw = sum([z is not None for z in [y, c, f, s, l]])
        assert nfunckw < 2, 'To specify the function, only one keyword from [y, c, f, s] should be used.'
        if nfunckw == 0:
            return self[:]

        other = self.copy()
        other.opts.add_data(y, c, f, s, l)
        other._set_function_values()
        return other

    @property
    def y(self):
        """ :return: function values at nodes """
        if np.any(self._yIsOutdated):
            self.update_y()
        return self._y

    @property
    def c(self):
        """ :return: interpolation coefficients """
        if np.any(self._cIsOutdated):
            self.update_c()
        return self._c

    @y.setter
    def y(self, val):
        val = np.atleast_2d(np.asarray(val))
        if val.shape[-1] != self.N:
            raise ValueError('y must be an array with {} elements in its last dimension.'.format(self.N))
        self._y = val
        self._yIsOutdated = np.full(val.shape[:-1], False, bool)
        self._cIsOutdated = np.full(val.shape[:-1], True, bool)

    @c.setter
    def c(self, val):
        val = np.atleast_2d(np.asarray(val))
        if val.shape[-1] != self.M:
            raise ValueError('c must be an array with {} elements in its last dimension.'.format(self.M))
        self._c = val
        self._yIsOutdated = np.full(val.shape[:-1], True, bool)
        self._cIsOutdated = np.full(val.shape[:-1], False, bool)

    """  Interpolation method """
    def __call__(self, x=None, order=None, dropdim=True):

        d = self.d

        if type(order) is str:
            if order is 'jac':
                order_array = np.identity(d, int)
            elif order is 'hess':
                order_array, mapping = hess_order(d)
            elif order is 'all':
                order_array, mapping = hess_order(d)
                order_array = np.hstack([np.zeros([d, 1],int), np.identity(d,int), order_array])
                mapping += 1 + d
            else:
                raise NotImplementedError
        else:
            order_array = order
            order = 'none' if (order is None) else 'provided'

        Phix = self.Phi(x, order_array, False)
        # if Phix.ndim == 2:
        #     Phix = Phix[np.newaxis]

        if self.opts.basistype is 'chebyshev':
            cPhix = np.array([np.dot(self.c, phix.T) for phix in Phix])
        else:
            try:
                cPhix = np.array([self.c * phix.T for phix in Phix])
            except:
                cPhix = np.array([np.dot(self.c, phix.T.toarray()) for phix in Phix])

        def clean(A):
            A = np.squeeze(A) if dropdim else A
            return np.asscalar(A) if (A.size == 1 and dropdim) else A


        if order in ['none', 'provided', 'jac']:
            return clean(cPhix)
        elif order in ['hess', 'all']:
            new_shape = [d, d] + list(cPhix.shape[1:])
            Hess = np.zeros(new_shape)
            for i in range(d):
                for j in range(d):
                    Hess[i,j] = cPhix[mapping[i,j]]
            if order is 'hess':
                return clean(Hess)
            else:
                return clean(cPhix[0]), clean(cPhix[1:(1 + d)]), clean(Hess)
        else:
            raise ValueError

    def __getitem__(self, item):
        litem = list(item) if isinstance(item, tuple) else [item]

        for j, k in enumerate(litem):
            if isinstance(k, str):
                try:
                    litem[j] = self.opts.ylabels[j].index(k)
                except:
                    txt = "Dimension {} has no '{}' variable.".format(j, k)
                    txt += '  Valid options are: ' + str(self.opts.ylabels[j])
                    raise ValueError(txt)
        item = tuple([k for k in litem])
        other = self.copy()
        other._y = np.atleast_2d(self._y[item])
        other._c = np.atleast_2d(self._c[item])
        other._yIsOutdated = other._yIsOutdated[item]
        other._cIsOutdated = other._cIsOutdated[item]
        # todo: copy the ylabels too!
        return other

    def __setitem__(self, key, value):
        litem = list(key) if isinstance(key, tuple) else [key]

        for j, k in enumerate(litem):
            if isinstance(k, str):
                try:
                    litem[j] = self.opts.ylabels[j].index(k)
                except:
                    txt = "Dimension {} has no '{}' variable.".format(j, k)
                    txt += '  Valid options are: ' + str(self.opts.ylabels[j])
                    raise ValueError(txt)
        item = tuple([k for k in litem])
        self._y[item] = value
        self._yIsOutdated[item] = False
        self._cIsOutdated[item] = True









''' ADDITIONAL FUNCTIONS'''
def hess_order(n):
    """
    Returns orders required to evaluate the Hessian matrix for a function with n variables
    and location of hessian entries in resulting array
    """
    A = np.array([a.flatten() for a in np.indices(3*np.ones(n))])
    A = A[:,A.sum(0)==2]

    C = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            v = np.zeros(n)
            v[i] += 1
            v[j] += 1
            C[i,j] = (v==A.T).all(1).nonzero()[0]

    return A, C















class BasisOptions(Options_Container):
    """
    Options for Basis class.

    This class stores options for creating a Basis class. It takes care of validating options given by user to Basis constructor.
    """
    valid_basis_types = ['chebyshev', 'spline', 'linear']
    valid_node_types = {'chebyshev': ['gaussian', 'lobatto', 'endpoint', 'uniform'],
                        'spline': ['canonical', 'user'],
                        'linear': ['canonical', 'user']}
    valid_methods = {'chebyshev': ['tensor', 'smolyak', 'complete', 'cluster', 'zcluster'],
                     'spline': ['tensor'],
                     'linear': ['tensor']}

    def __init__(self, d, basistype=None, nodetype=None, method=None, qn=None, qp=None, labels=None, ylabels=None,
                 f=None, y=None, c=None, s=None, l=None):
        """
        Make default options dictionary
        :param int d: dimension of the basis
        :return: an object with default values for Basis.opts
        """
        method = method if method else self.valid_methods[basistype][0]
        nodetype = nodetype if nodetype else self.valid_node_types[basistype][0]


        assert basistype in self.valid_basis_types, "basistype must be 'chebyshev', 'spline', or 'linear'."
        assert nodetype in self.valid_node_types[basistype], "nodetype must be one of " + str(self.valid_node_types[basistype])
        assert method in self.valid_methods[basistype], "method must be one of " + str(self.valid_methods[basistype])

        self.d = d
        self.basistype = basistype
        self.nodetype = nodetype
        self.method = method  # method to expand the basis (tensor, smolyak, cluster, zcluster, complete)
        self.qn = qn  # node parameter, to guide the selection of node combinations
        self.qp = qp  # polynomial parameter, to guide the selection of polynomial combinations
        self.labels = labels if labels else ["V{}".format(dim) for dim in range(d)]
        self.ylabels = ylabels
        self._ix = []
        self._ip = []
        self.add_data(y, c, f, s, l)

    def add_data(self, y, c, f, s, l):
        nfunckw = sum([z is not None for z in [y, c, f, s, l]])
        assert nfunckw < 2, 'To specify the function, only one keyword from [y, c, f, s] should be used.'
        if nfunckw == 0:
            s = 1

        self.y = y
        self.c = c
        self.f = f
        self.s = s
        self.l = l


    def clear_data(self):
        self.y = None
        self.c = None
        self.f = None
        self.s = None
        self.l = None

    ''' Properties'''
    @property
    def ix(self):
        """  numpy array with valid combination of nodes """
        return self._ix

    @property
    def ip(self):
        """  numpy array with valid combination of basis polynomials """
        return self._ip

    @property
    def labels(self):
        """  list of variable names """
        return self._labels

    @property
    def keys(self):
        """ list of available options in BasisOptions """
        return([name for name in BasisOptions.__dict__ if not name.startswith('_')])


    ''' Setters '''
    @ix.setter
    def ix(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == self.d:
            self._ix = value
        else:
            raise ValueError('validX must be a 2-dimensional numpy array with {} rows'.format(self.d))

    @ip.setter
    def ip(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == self.d:
            self._ip = value
        else:
            raise ValueError('validPhi must be a 2-dimensional numpy array with {} rows'.format(self.d))

    @labels.setter
    def labels(self, value):
        if isinstance(value, list) and len(value) == self.d:
            self._labels = value
        else:
            raise ValueError('labels must be a list of {} strings'.format(self.d))

    def validateChebyshev(self, n):
        """ Validates the options given for a Chebyshev Basis """
        if self.d == 1:
            return

        if self.method in ['complete', 'cluster', 'zcluster']:
            if self.qn is None:
                self.qn = 0
            if self.qp is None:
                self.qp = max(n) - 1
        elif self.method == 'smolyak':
            n_valid = 2 ** np.ceil(np.log2(n - 1)) + 1
            if np.any(n != n_valid):
                warn('For smolyak expansion, number of nodes should be n = 2^k+1 for some k=1,2,...')
                print('Adjusting nodes\n {:7s}  {:7s}'.format('old n', 'new n'))
                for n1, n2 in zip(n, n_valid):
                    print('{:7.0f} {:7.0f}'.format(n1, n2))
                n = np.array(n_valid,'int')
            if self.nodetype != 'lobatto':
                warn('Smolyak expansion requires Lobatto nodes: changing "nodetype".')
                self.nodetype = 'lobatto'
            if self.qn is None:
                self.qn = np.atleast_1d(2)
            else:
                self.qn = np.atleast_1d(self.qn)
            if self.qp is None:
                self.qp = self.qn
            else:
                self.qp = np.atleast_1d(self.qp)

    def expandGrid(self, n):
        """
             Compute nodes for multidimensional basis and other auxiliary fields required to keep track
             of the basis (how to combine unidimensional nodes bases). It is called by the constructor method, and as
             such is not directly needed by the user. expandGrid updates the following fields

             * validPhi: indices to combine unidimensional bases
             * validX:   indices to combine unidimensional nodes

             Combining polynomials depends on value of input "method":

             * 'tensor' takes all possible combinations
             * 'smolyak' computes Smolyak basis, given qp parameter
             * 'complete', 'cluster', and 'zcluster' choose polynomials with degrees not exceeding qp

             Expanding nodes depends on value of field opts.method

             * 'tensor' and 'complete' take all possible combinations
             * 'smolyak' computes Smolyak basis, given qn parameter
             * 'cluster' and 'zcluster' compute clusters of the tensor nodes based on qn

        :return: None
        """
        if self.d == 1:
            self.ix = np.arange(n, dtype=int).reshape(1, -1)
            self.ip = np.arange(n, dtype=int).reshape(1, -1)
            return

        ''' Smolyak interpolation: done by SmolyakGrid function'''
        if self.method == 'smolyak':
            self.ix, self.ip = SmolyakGrid(n, self.qn, self.qp)

        ''' All other methods'''
        degs = n - 1  # degree of polynomials
        ldeg = [np.arange(degs[ni] + 1) for ni in range(self.d)]

        idxAll = gridmake(*ldeg)   # degree of polynomials = index

        ''' Expanding the polynomials'''
        if self.method == 'tensor':
            self.ip = idxAll
        else:
            degValid = np.sum(idxAll, axis=0) <= self.qp
            self.ip = idxAll[:, degValid]

        ''' Expanding the nodes'''
        if self.method in ['tensor', 'complete']:
            self.ix = idxAll
        elif self.method in ['cluster', 'zcluster']:
            raise NotImplementedError # todo: implement this method


def SmolyakGrid(n, qn, qp=None):
    """

    :param n: number of nodes per dimension
    :param qn: cut-off parameters for node selection (array)
    :param qp: cut-off parameters for polynomial selection(array)
    :return: a (node_indices, polynomial_indices) tuple to form the Smolyak grid from univariate nodes and polynomials
    """

    n = np.atleast_1d(n)
    qn = np.atleast_1d(qn)
    qp = qn if qp is None else np.atleast_1d(qp)

    # Dimensions
    d = n.size
    ngroups = np.log2(n - 1) + 1

    N = max(ngroups)

    # node parameter
    node_q = max(qn)
    node_isotropic = qn.size == 1
    if node_isotropic:
        qn = np.zeros(d)

    # polynomial parameter
    poly_q = max(qp)
    poly_isotropic = qp.size == 1
    if poly_isotropic:
        qp = np.zeros(d)

    # make grid that identifies node groups
    k = np.arange(2 ** (N - 1) + 1)
    g = np.zeros(k.size,'int')

    p2 = 2
    for it in np.arange(N, 2, -1):
        odds = np.array(k % p2,'bool')
        g[odds] = it
        k[odds] = 0
        p2 *= 2

    g[0] = g[-1] = 2
    g[(-1 + g.size) / 2] = 1

    #gg = np.copy(g)

    # make disjoint sets
    nodeMapping = [g[g <= ngroups[i]] for i in range(d)]
    polyMapping = [np.sort(A) for A in nodeMapping]

    # set up nodes for first dimension
    nodeSum = nodeMapping[0]
    theNodes = np.atleast_2d(np.arange(n[0]))
    if not node_isotropic:
        isvalid = nodeSum  <= (qn[0] + 1)
        theNodes = theNodes[:, isvalid ]  # todo: not sure this index is ok
        nodeSum = nodeSum[isvalid]

    # set up polynomials for first dimension
    polySum = polyMapping[0]
    thePolys = np.atleast_2d(np.arange(n[0]))
    if not poly_isotropic:
        isvalid = polySum  <= (qp[0] + 1)
        thePolys = thePolys[:, isvalid]   # todo: not sure this index is ok
        polySum = polySum[isvalid]

    # compute the grid
    for k in range(1, d):
        theNodes, nodeSum = ndgrid2(theNodes, nodeSum, nodeMapping[k], 1 + k + node_q, qn[k])
        thePolys, polySum = ndgrid2(thePolys, polySum, polyMapping[k], 1 + k + poly_q, qp[k])

    return theNodes, thePolys


def ndgrid2(Indices, indSum, newDim, q, qk):
    """
    Expanding a Smolyak grid, 2 dimensions

    :param Indices: Previous iteration smolyak grid
    :param newDim: new indices to be combined with Indices
    :param q: cutt-off parameter for new sum of indices
    :param qk: adjustment for anisotropic grids
    :return: Updated "Indices" and "groupsum"
    """
    idx = np.indices((indSum.size, newDim.size)).reshape(2, -1)
    NewSum = indSum[idx[0]] + newDim[idx[1]]
    isValid = NewSum <= q
    if qk != 0: #anisotropic
        isValid &= (newDim[idx[1]] <= qk + 1)

    idxv = idx[:, isValid]
    NewSum = NewSum[isValid]
    NewIndices = np.vstack((Indices[:, idxv[0]], newDim[idxv[1]]))
    return NewIndices, NewSum


