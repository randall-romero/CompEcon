import numpy as np
from scipy.sparse import csc_matrix, diags, tril
from .basis import Basis

__author__ = 'Randall'
# TODO: complete this class
# todo: compare performance of csr_matrix and csc_matrix to deal with sparse interpolation operators
# fixme: interpolation is 25 slower than in matlab when 2 dimensions!! 2x slower with only one



class BasisSpline(Basis):
    def __init__(self, *args, k=3, **kwargs):
        """
        Initialize a spline basis

        Parameters
        ----------
        *args : either one tuple or three array-like parameters
            Parameters to define the breakpoints of the spline (see notes for details)
        k : int (optional, 3 by default)
            the degree of the spline polynomials
        **kwargs : dict
            options passed to :ref:`BasisOptions`.

        Notes
        -----
        1. If only 1 positional argument is provided, it must be a tuple of 'd' array-like, each of them containing the breaks for one dimension.
        2. If 3 positional parameters are provided, then they are interpreted as `n`, `a`, `b`, where
            - n : int or array_like, number of nodes per dimension.
            - a : int or array_like, lower bound(s) for interpolation.
            - b : int or array_like, upper bound(s) for interpolation.

        Examples
        --------
        1. A cubic spline basis to interpolate a function of wealth.

        >>> BasisSpline(15, -2, 3, labels=['wealth'])

        2. A linear spline basis to interpolate income as a function of wealth, for employed and unemployed workers.

        >>> income = BasisSpline(15, -2, 3, k=1, labels=['wealth'], l=['employed', 'unemployed')

        3. A 2-dimensional basis, using 9 nodes in each dimension, and forming grid by tensor product (81 nodes total)

        >>> BasisSpline(9, [0, 0], [2, 3])

        4. A cubic spline basis, with 7 user-defined nodes (note that they are passed as tuple)

        >>> BasisSpline(([0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1],))

        """

        nargs = len(args)
        if nargs == 1:
            if isinstance(args[0], tuple):
                breaks = [np.sort(br) for br in args[0]]
                n = np.array([br.size + k - 1 for br in breaks])
                a = np.array([br[0] for br in breaks])
                b = np.array([br[-1] for br in breaks])
                kwargs['nodetype'] = 'user'
            else:
                raise ValueError("If only 1 positional argument is provided, it must be a tuple of 'd' array-like, " +
                                 "each of them containing the breaks for one dimension.")
        elif nargs == 3:
            n, a, b = np.broadcast_arrays(*np.atleast_1d(*args))
            breaks = [np.linspace(aa, bb, nn + 1 - k) for aa, bb, nn in zip(a, b, n)]
            kwargs['nodetype'] = 'canonical'
        else:
            txt = 'Either 1 or 3 positional arguments must be provided\n'
            txt += '\t1 argument -> break points\n'
            txt += '\t3 argument -> n, a, b'
            raise ValueError(txt)

        ''' Check inputs '''
        assert ((k > 0) and type(k) is int), 'k must be a positive integer'
        assert np.all(n > k), 'number of nodes must exceed order of spline'
        assert np.all([(br.size > 1) for br in breaks]), 'breakpoint sequence must contain at least two elements'

        ''' Make instance '''
        kwargs['basistype'] = 'spline'
        super().__init__(n, a, b, **kwargs)
        self.k = k
        self.breaks = breaks
        self._set_nodes()

    def _set_nodes(self):
        # Sets the basis nodes
        n = self.n
        k = self.k

        self._nodes = list()

        for i in range(self.d):
            x = np.cumsum(self._augbreaks(i, k))
            x = (x[k : n[i] + k] - x[:n[i]]) / k
            x[0] = self.a[i]
            x[-1] = self.b[i]
            self._nodes.append(x)
        self._expand_nodes()

    def _augbreaks(self, i, m,):
        aa = np.repeat(self.a[i], m)
        bb = np.repeat(self.b[i], m)
        return np.concatenate((aa, self.breaks[i], bb))

    def _update_diff_operators(self, i, order):
        """
        Updates the list _D of differentiation operators in the basis

        Parameters
        ----------
        i : int
            dimension for which the derivative is required.
        order : int
            order of required derivative (if positive) or integral (if negative)

        """
        keys = set(self._diff_operators[i].keys())

        if (order in keys) or (order == 0):
            return  # Use previously stored values if available

        n = self.n[i]
        a = self.a[i]
        b = self.b[i]
        k = self.k

        assert order <= k, 'order must be less or equal to k'
        kk = k - 1 - min(order, 0)
        augbreaks = self._augbreaks(i, kk)

        if order > 0:
            def sptemp(j):
                temp = np.atleast_2d((k + 1 - j) / (augbreaks[k:(n + k - j)] - augbreaks[(j - 1):(n - 1)]))
                return diags((-temp, temp), [0, 1], (n - j, n + 1 - j))

            missing_keys = set(range(1, order + 1)) - keys

            if 1 in missing_keys:
                self._diff_operators[i][1] = sptemp(1)
                missing_keys -= {1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=True)
            while missing_keys:
                j = missing_keys.pop()
                self._diff_operators[i][j] = np.dot(sptemp(j), self._diff_operators[i][j - 1])
        else:
            def sptemp(j):
                temp = (augbreaks[(kk + 1):(kk + n - j)] -
                        augbreaks[(kk - k + j + 1):(kk + n - k)]) / (k - j)

                return tril(np.tile(temp, (n - j, 1)), -1)

            missing_keys = set(range(order, 0)) - keys

            if -1 in missing_keys:
                self._diff_operators[i][-1] = sptemp(-1)
                missing_keys -= {-1}

            missing_keys = list(missing_keys)
            missing_keys.sort(reverse=False)
            while missing_keys:
                j = missing_keys.pop()
                self._diff_operators[i][j] = sptemp(j) * self._diff_operators[i][j + 1]


    """
        Interpolation methods
    """
    def _phi1d(self, i, x=None, order=0):
        """
        Computes interpolation matrices for given data x and order of differentiation 'order' (integration if negative)

        Parameters
        ----------
        i : int
            dimension for which the derivative is required.
        x : array_like
            nx floats, evaluation points (defaults to nodes)
        order : array_like
            array with m ints, orders for differentiation (+) / integration (-)

        Returns
        -------
        interpolation matrices : np.ndarray, m differentiation basis matrices (one of each order)
             Each value is a sparse interpolation matrices, nx times n.

        Notes
        -----
        1. `order` may have repeated values, since this function is tipically needed to evaluate partial derivatives.
        2. To save on computational cost, only unique values in `order` are evaluated once, then results are rearranged by this function.
        3. Future version of this function will make use of @cached_property to acomplish this savings.
        4. This function takes care of only one dimension of the basis. The :ref:`Basis` class (on which `BasisSpline` is based) has a method `Phi` that takes care of combining these interpolation matrices.

        Examples
        --------

        Create a basis with 5 nodes, get the interpolation matrix evaluated at 20 points:

        >>> import numpy as np
        >>> n, a, b = 5, 0, 4
        >>> x = np.linspace(a,b, 20)
        >>> basis = BasisSpline(n, a, b)
        >>> basis._phi1d(0, x, order=[0, 1])


        Calling an instance directly (as in the last line) is equivalent to calling the interpolation method.
        """
        n = self.n[i]
        k = self.k

        if order is None:
            order = 0

        order = np.atleast_1d(order).flatten()
        assert np.max(order) < k, 'Derivatives defined for order less than k'

        nn = n + np.maximum(0, -np.min(order))  # todo review why nn is not used, weird

        # Check for x argument
        xIsProvided = (x is not None)
        x = x.flatten() if xIsProvided else self._nodes[i]
        nx = x.size

        minorder = np.min(order)
        kaug = k - minorder
        augbreaks = self._augbreaks(i, kaug)
        ind = self._lookup(augbreaks, x)

        # Recursively determine the values of a k-order basis matrix.
        # This is placed in an (m x k+1-order) matrix
        bas = np.zeros((kaug + 1, nx))
        bas[0] = 1
        Phidict = dict()

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
                    Phidict[oi] = csc_matrix((data, (r, c)), (nx, n-oi))

                    if oi:
                        # If needed compute derivative or anti-derivative
                        Phidict[oi] = Phidict[oi] * self._diff(i, oi)

        # todo: review, i think this will return only unique values

        Phi = np.array([Phidict[k] for k in order])
        return Phi
