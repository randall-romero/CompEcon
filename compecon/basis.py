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
        __setitem__: Modify the y values for a specified item.
        __call__: Evaluate the interpolated function at arbitrary values.


    See Also:
        BasisChebyshev class
        BasisSpline class
        BasisLinear class

    Notes:
        The code in this class and its subclasses is based on Miranda and Fackler's CompEcon 2014 toolbox. In particular,
        the following functions were used or their functionality is replicated by Basis and its subclasses:
            * chebbase, chebbasex, chebdef, chebdop chebnode (Chebyshev interpolation)
            * linbase, lindef, lindop, linnode (Linear interpolation)
            * splibase, splidef, splidop, splinode (Spline interpolation)
            * funbase, funbasex, fundef, fundefn, funeval, funfitf, funfitxy, funnode (similar to the above functions)
            * lookup (for linear and spline interpolation).

        Unlike the original Matlab implementation, in which a basis definition (as created by calling fundefn) is coded
        as a structure and the interpolation coefficients are kept separately in an array, this Python implementation
        combines both elements into a single class. This simplifies the code by reducing the number of arguments passed
        to interpolation functions. For example, to evaluate a 'basis' at 'x' with coefficients 'c', in Matlab one needs
        to call funeval(c, basis, x) whereas in Python it simplies to basis(x).

    References:
        Miranda and Fackler 2002 Applied Computational Economics and Finance. MIT Press.
        Judd, Maliar, Maliar, Valero 2013 Smolyak Method for Solving Dynamic Economic Models. NBER Working Paper 19326
    """

    def __init__(self, n, a, b, **kwargs):
        """ Create an instance of Basis.

        Args:
            n: number of nodes per dimension
            a: lower bounds
            b: upper bounds
            **kwargs: options passed to BasisOptions

            The dimension of the basis is inferred from the number of elements in n, a, b. If all of them are scalars,
            then d = 1. Otherwise, they are broadcast to a common size array.

            Notice that only one of the keyword arguments f, y, c, s, l can be specified. If none is, then s=1.

        Notes:
            Users should not create Basis instances directly, as they would be useless (class Basis does not implement
            methods to compute nodes nor interpolation matrices). Instead, they should create instances of the classes
            BasisChebyshev, BasisSpline or BasisLinear.

        Returns:
            A Basis instance.

        """
        n, a, b = np.broadcast_arrays(*np.atleast_1d(n, a, b))
        assert np.all(n > 2), 'n must be at least 3'
        assert np.all(a < b), 'lower bound must be less than upper bound'
        self.d = n.size
        self.n = n
        self.a = a
        self.b = b
        self.opts = BasisOptions(n, **kwargs)

        self.N = self.opts.ix.shape[1]  # Total number of nodes
        self.M = self.opts.ip.shape[1]  # Total number of polynomials

        self._nodes = list()
        self._diff_operators = [dict() for h in range(self.d)]
        self._PhiT = None


    def _set_nodes(self):
        """ Computes interpolation nodes for all dimensions.

        It computes nodes for each dimension and store them in a list of d numpy arrays (attribute _nodes).

        Class Basis only declares this method; implementation is done by each of its subclasses (BasisChebyshev,
        BasisSpline, and BasisLinear). None of them takes arguments.

        Returns:
            None
        """
        pass


    def _phi1d(self, i, x=None, order=0):
        """ Computes interpolation matrix for a given basis dimension.

        Class Basis only declares this method; implementation is done by each of its subclasses (BasisChebyshev,
        BasisSpline, and BasisLinear). However, they all take the same arguments.

        Args:
            i (int): The required dimension (scalar between 0 and d-1)
            x (numpy array): vector of 'nx' evaluation points. If None, then basis nodes are used.
            order (numpy array): vector of 'no' integers, indicating orders of differentiation (integration if negative).
            Default is zero.

        Returns:
            Interpolation array of dimension no.nx.n[i].
        """
        pass

    def _update_diff_operators(self, i, order):
        """ Compute differentiation operators, storing them in _diff_operators.

        Class Basis only declares this method; implementation is done by each of its subclasses (BasisChebyshev,
        BasisSpline, and BasisLinear). However, they all take the same arguments.

        Args:
            i (int): The required dimension (scalar between 0 and d-1)
            order (int): Order of differentiation (integration if negative).

        Returns:
            None
        """
        pass

    def _set_function_values(self):
        """ Set value of interpolated function and corresponding coefficients, at basis nodes.

        When creating a Basis, there are five options to specify the values of the interpolated function at the nodes:
            f: a callable (lambda or def), sets Basis.y = f(Basis.nodes).
            y: a numpy array with N elements in last dimension, sets Basis.y = y.
            c: a numpy array with M elements in last dimension, sets Basis.c = c.
            s: a numpy array of integers, sets Basis.y = numpy.zeros([s, N]).
            l: a list of lists, sets s = [len(labels) for labels in l], then Basis.y = numpy.zeros([s, N]).

        Examples:
            Basis(..., f=np.sin, ...)
            Basis(..., y=np.ones((2,3, N)), ...)
            Basis(..., c=np.zeros((2, M)), ...)
            Basis(..., s=[2, 3], ...)
            Basis(..., l=['unemployed', 'employed'], ...)
            Basis(..., l=[['unemployed', 'employed'], ['rest', 'work']], ...)

        Returns:
            None
        """

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
        """ Compute the nodes grid (required when d > 1).

        Uses the nodes from each dimension (attribute _nodes) to compute the nodes grid, as indicated by opts.ix.
        It calls Phi() to fill in the fields _PhiT and _PhiInvT.

        Returns:
            None
        """
        # TODO: this is too slow for sparse basis, implement Mario's equivalent functions
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
        """ Interpolation matrix at basis nodes.

        Returns:
            numpy array with dimensions N.M
        """
        return self._PhiT.T

    def _diff(self, i, m):
        """ Returns a specified differentiation operator.

        If operator has been computed already, it fetches it from attribute _diff_operators. Otherwise it calls
        _update_diff_operators(i, m) to compute it and then returns it.

        Args:
            i (int): Required dimension.
            m (int): Required order.

        Returns:
            Numpy array with desired operator.
        """
        if m not in self._diff_operators[i].keys():
            self._update_diff_operators(i, m)
        return self._diff_operators[i][m]

    def Phi(self, x=None, order=None, dropdim=True):
        """Compute the interpolation matrix :math:`\Phi(x)`


        Args:
            x: evaluation points. Either a d.nx numpy array of domain values at which function is evaluated or
               d list of arrays of coordinates in each domain dimension.
            order (d.no numpy array): order of derivatives (integrals if negative), default is zero (no derivative).
            dropdim (bool): squeeze dimensions if True (default).

        Returns:
            A numpy array with interpolation matrices
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
        """ String representation of a Basis contents.

        A string describing the contents of the basis.

        Returns:
            A string.
        """
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

    def plot(self, order=0, m=None, i=0, nx=120, shape=None):
        """ Plots interpolation basis functions.

        Args:
            order (int): Order of derivative (zero)
            m (int): How many basis function to plot,  m <= n[i],  n[i] if None
            i: Dimension to be plotted (0 by default)
            nx: Number of points to evaluate the polynomials (120)
            shape: subplot grid, to plot each polynomial in its own axis. All together if None.

        Returns:
            A matplotlib.pyplot instance.
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

        if shape is None:
            x.shape = x.size, 1
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

        else:
            plt.figure()
            for j, yi in enumerate(y.T):
                plt.subplot(shape[0], shape[1], j)
                plt.plot(x, yi)
                plt.plot(nodes, 0 * nodes, 'ro')
                plt.xlim(a, b)
                # plt.xlabel(self.opts.labels[i])

    @property
    def x(self):
        """ Interpolation nodes """
        return self.nodes

    def update_y(self):
        """ Update function values (called after changing c).

        It computes :math:`y = \Phi(x)c`, where x are the basis nodes.

        Returns:
            None
        """
        # todo: try to update only the outdated values. Can't yet figure out how to index them
        # ii = self._yIsOutdated
        if self.opts.basistype is 'chebyshev':
            self._y = np.dot(self._c, self._PhiT)
        else:
            self._y = self._c * self._PhiT
        self._yIsOutdated = np.full(self._yIsOutdated.shape, False)

    def update_c(self):
        """ Update interpolation coefficients (called after changing y).

        It computes :math:`c = \Phi(x)^{-1}y`, where x are the basis nodes.

        Returns:
            None
        """
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
        """ Dimensions of interpolated function, i.e. the tuple (s0, s1, ..., sk). """
        if np.any(self._yIsOutdated):
            return self._c.shape[:-1]
        else:
            return self._y.shape[:-1]

    @property
    def size(self):
        """ Total number of interpolated functions, i.e. s0 x ... x sk. """
        return np.prod(self.shape)

    @property
    def ndim(self):
        """ Number of dimension of interpolated functions. """
        return len(self.shape) - 1

    @property
    def shape_N(self):
        """ Shape of y. Same as shape, N. """
        temp = list(self.shape)
        temp.append(self.N)
        return temp

    def copy(self):
        """ Returns a shallow copy of a Basis instance. """
        return copy.copy(self)

    def duplicate(self, y=None, c=None, f=None, s=None, l=None):
        """ A shallow copy of the basis, allowing to specify new functions to be interpolated.

        Makes a shallow copy of the basis, allowing the new instance to have its own approximation coefficients
        while sharing the nodes and interpolation matrices.  New instance values are specified with the same
        arguments used to create a new Basis, if none is provided, then it copies original coefficients by simply
        getting all items:

        copy_instance = original_instance[:]

        Args: Only one of these arguments must be provided.
            y: a numpy array with N elements in last dimension, sets Basis.y = y.
            c: a numpy array with M elements in last dimension, sets Basis.c = c.
            f: a callable (lambda or def), sets Basis.y = f(Basis.nodes).
            s: a numpy array of integers, sets Basis.y = numpy.zeros([s, N]).
            l: a list of lists, sets s = [len(labels) for labels in l], then Basis.y = numpy.zeros([s, N]).

        Returns:
            A new Basis instance with same interpolation matrices but different interpolated functions.

        """

        """ Duplicate the basis





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
        """ Value of interpolated function(s) at basis nodes (s0...sk.N array)."""
        if np.any(self._yIsOutdated):
            self.update_y()
        return self._y

    @property
    def c(self):
        """ Coefficients of interpolated function(s) at basis nodes (s0...sk.M array). """
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
        """ Evaluate the interpolated function at arbitrary values.

        Args:
            x: evaluation points. Either a d.nx numpy array of domain values at which function is evaluated or
               d list of arrays of coordinates in each domain dimension.
            order (d.no numpy array): order of derivatives (integrals if negative), default is zero (no derivative).
            dropdim (bool): squeeze dimensions if True (default).

        Notice that all arguments are defined as in the Phi() method.

        Returns:
            Interpolated values.
        """

        d = self.d
        x = np.atleast_1d(x)

        if type(order) is str:
            if order is 'jac':
                order_array = np.identity(d, int)
            elif order is 'fjac':
                order_array = np.hstack([np.zeros([d, 1],int), np.identity(d,int)])
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
        elif order in ['fjac']:
            return clean(cPhix[0]), clean(cPhix[1:])
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
        """ Return a copy of Basis for specified item.

        Args:
            item: Index for the required functions. Either an int, a slice, or a string (see examples below).

        Examples:
            income = Basis(n, a, b, l=[['employed', 'unemployed'], ['male', 'female']])
            z = income[0]  # Basis with income for employed.
            z = income['employed']  # same as previous line.
            z = income[:, 'female']  # Basis with income for female, both employed and unemployed.
            z = income[1, 0]  # Basis for income of unemployed male.
            z = income['unemployed', 'male']  # same as previous line.

        Returns:
            The same Basis but to interpolate only the required functions.
        """
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
        """ Modify the y values for a specified item.

        Args:
            key: Index for the required functions. Either an int, a slice, or a string (see examples below).
            value: numpy arrays of appropriate dimensions (must have N elements in last dimension).

        Examples: (assuming z has the right shape in each case)
            income = Basis(n, a, b, l=[['employed', 'unemployed'], ['male', 'female']])
            income[0] = z  # sets income for employed.
            income['employed'] = z  # same as previous line.
            income[:, 'female'] = z  # sets income for female, both employed and unemployed.
            income[1, 0] = z  # sets income of unemployed male.
            income['unemployed', 'male'] = z # same as previous line.

        Returns:
            None
        """
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

    def __init__(self, n: np.array, basistype=None, nodetype=None, method=None, qn=None, qp=None, labels=None,
                 f=None, y=None, c=None, s=None, l=None):
        """
        Make default options dictionary
        :param int n: number of nodes per dimension
        :return: an object with default values for Basis.opts
        """
        method = method if method else self.valid_methods[basistype][0]
        nodetype = nodetype if nodetype else self.valid_node_types[basistype][0]


        assert basistype in self.valid_basis_types, "basistype must be 'chebyshev', 'spline', or 'linear'."
        assert nodetype in self.valid_node_types[basistype], "nodetype must be one of " + str(self.valid_node_types[basistype])
        assert method in self.valid_methods[basistype], "method must be one of " + str(self.valid_methods[basistype])

        self.d = n.size
        self.basistype = basistype
        self.nodetype = nodetype
        self.method = method  # method to expand the basis (tensor, smolyak, cluster, zcluster, complete)
        self.qn = qn  # node parameter, to guide the selection of node combinations
        self.qp = qp  # polynomial parameter, to guide the selection of polynomial combinations
        self.labels = labels if labels else ["V{}".format(dim) for dim in range(self.d)]
        self.ylabels = None
        self._ix = []
        self._ip = []
        self.add_data(y, c, f, s, l)
        if basistype == 'chebyshev':
            self.validateChebyshev(n)

        self.expandGrid(n)

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


def hess_order(n):
    """ Order array to evaluate Hessian matrices.

    Returns orders required to evaluate the Hessian matrix for a function with n variables
    and location of hessian entries in resulting array.

    Args:
        n: Number of variables in function domain.

    Returns:
        Array with orders of derivatives.
        Array with location of hessian entries.
    """
    A = np.array([a.flatten() for a in np.indices(3*np.ones(n, int))])
    A = A[:,A.sum(0)==2]

    C = np.zeros([n,n], int)
    for i in range(n):
        for j in range(n):
            v = np.zeros(n)
            v[i] += 1
            v[j] += 1
            C[i,j] = (v==A.T).all(1).nonzero()[0]

    return A, C

