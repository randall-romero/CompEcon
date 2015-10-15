from warnings import warn
import numpy as np
from .basisChebyshev import BasisChebyshev
from .tools import gridmake, Options_Container

__author__ = 'Randall'


class Basis(object):
    """
      A multivariate interpolation basis
    """

    def __init__(self, n=None, a=None, b=None, **options):
        """
        A multivariate basis

        :param n: number of nodes (d-array)
        :param a: lower bounds (d-array)
        :param b: upper bounds (d-vector)
        :param options: keyword-separated list of additional options (see below)

        Some examples::

                n, a, b = [5, 9], [0, 2], [4, 7]
                Basis(n, a, b)
                Basis(n, a, b, type='gaussian', nodetype='lobatto')
                Basis(n, a, b, type = 'spline', k = 3)
                Basis(n, a, b, method='smolyak', qn = 3, qp = 3)
                Basis(n, a, b, labels=['savings','productivity'])

        For more details about the options, see the OptBasis class documentation
        """

        if n is None:
            return
        if type(n) in [Basis, BasisChebyshev]:
            self.__dict__ = as_basis(n).__dict__
            return

        n, a, b = np.atleast_1d(n, a, b)  # Convert n, a, and b to np.arrays
        d = a.size  # dimension of basis

        # todo use same number of nodes in all dimension if n is scalar
        if not np.all(a < b):
            raise ValueError('All lower bounds must be less than their corresponding upper bound')

        # Read user options
        opts = OptBasis(d, **options)
        # for option, value in options.items():
        #     if option not in opts.keys:
        #         warn('Unknown option {} with value {}; ignoring'.format(option, value))
        #     else:
        #         setattr(opts, option, value)

        # Validate options for chebyshev basis of several dimensions
        if opts.type == 'chebyshev':
            opts.validateChebyshev(n)

        # make list of 1-basis

        if opts.type == 'chebyshev':
            B1 = [BasisChebyshev(n[i], a[i], b[i],
                                 **{'nodetype': opts.nodetype,
                                    'label': opts.labels[i]}) for i in range(d)]
        else:
            raise NotImplementedError

        # Pack value in object
        self.a = a              #: lower bounds
        self.b = b              #: upper bounds
        self.n = n              #: number of nodes
        self.d = d              #: dimension
        self._B1 = B1           #: list with 1-d bases
        self.opts = opts        #: OptBasis object, basis options
        self.opts.expandGrid(n)
        self.type = opts.type   #: type of basis (chebyshev, spline)
        self._nodes = np.array([self._B1[k].nodes[self.opts.ix[k]].flatten() for k in range(self.d)]) #: basis nodes

        # Compute interpolation matrix at nodes
        _Phi = self.interpolation()
        self._PhiT = _Phi.T

        # Compute inverse if not spline
        if self.type == 'chebyshev':
            self._PhiInvT = np.linalg.pinv(_Phi).T
        else:
            raise NotImplementedError



    def interpolation(self, x=None, order=None):
        """Compute the interpolation matrix :math:`\Phi(x)`

        :param np.array x: evaluation points
        :param np.array order: order of derivatives (integrates if negative)
        :return: Interpolation matrix

        Example::

                n, a, b = [9, 9], [0, 0], [5, 7]
                Phi = Basis(n, a, b, method='smolyak', qn = 3, qp = 3)
                Phi.interpolation()
                Phi()                                       # same as previous line
                Phi(order = [[2, 1, 1, 0], [0, 1, 1, 2]])   # Hessian of interpolation matrix

        Calling an instance directly (as in the last two lines) is equivalent to calling the interpolation method.
        """
        if self.d == 1:
            return self._B1[0](x, order)

        if order is None:
            order = np.zeros([self.d, 1], 'int')
        else:
            order = np.atleast_1d(order)
            if order.ndim == 1:
                assert (order.size == self.d)
                order.shape = (self.d, 1)
            else:
                assert (order.shape[0] == self.d)

        orderIsScalar = order.shape[1] == 1

        # check what type of input x is provided
        if x is None:
            Phi = self._interp_default(order)
        elif type(x) == list:
            Phi = self._interp_list(x,order)
        elif type(x) == np.ndarray:
            Phi = self._interp_matrix(x,order)
        else:
            raise Exception('wrong x input')
        return Phi[0] if orderIsScalar else Phi

    def __call__(self, x=None, order=None):
        """
        Equivalent to self.interpolation(x, order)

        Example: If V is a Basis instance, the following to lines are equivalent::

                V.interpolation(x, order)
                V(x, order)

        """
        return self.interpolation(x, order)

    def _interp_default(self, order):
        """

        :param order:
        :return:
        """
        r = self.opts.ix
        c = self.opts.ip
        oo = np.arange(order.shape[1])

        if self.opts.type == 'chebyshev':
            PHI = [self._B1[k](order=order[k])[np.ix_(oo, r[k], c[k])] for k in range(self.d)]
        else:
            raise NotImplemented
        return np.prod(PHI, 0)

    def _interp_matrix(self, x, order):
        """

        :param x:
        :param order:
        :return:
        """
        assert (x.shape[0] == self.d)  # 'In Interpolation, class basis: x must have d columns')
        r = np.arange(x.shape[1])
        c = self.opts.ip
        oo = np.arange(order.shape[1])

        if self.opts.type == 'chebyshev':
            PHI = [self._B1[k](x[k], order[k])[np.ix_(oo, r, c[k])] for k in range(self.d)]
        else:
            raise NotImplemented
        return np.prod(PHI, 0)

    def _interp_list(self, x, order):
        """

        :param x:
        :param order:
        :return:
        """
        assert (len(x) == self.d)
        r = gridmake(*[np.arange(xi.size) for xi in x])
        c = self.opts.ip
        oo = np.arange(order.shape[1])

        if self.opts.type == 'chebyshev':
            PHI = [self._B1[k](x[k], order[k])[np.ix_(oo, r[k], c[k])] for k in range(self.d)]
        else:
            raise NotImplemented
        return np.product(PHI,0)

    def plot(self):
        raise NotImplementedError # todo: implement this method

    @property
    def nodes(self):
        """Basis nodes"""
        return self._nodes.flatten() if self.d == 1 else self._nodes

    @property
    def N(self):
        """ Total number of nodes"""
        return self.opts.ix.shape[-1]

    @property
    def M(self):
        """ Total number of polynomials"""
        return self.opts.ip.shape[-1]

    @property
    def Phi(self):
        """ Interpolation matrix """
        return self._PhiT.T

    def __repr__(self):
        # if self.d == 1:
        #     return self._B1[0].__repr__()

        n, a, b = self.n, self.a, self.b
        nodetype = self.opts.nodetype
        vnames = self.opts.labels
        vnamelen = str(max(len(v) for v in vnames))

        bstr = "A {}-dimension {} basis:  ".format(self.d, self.type.capitalize())
        bstr += "using {:d} {} nodes and {:d} polynomials".format(self.N, nodetype.capitalize(), self.M)
        if self.d > 1:
            bstr += ", expanded by {}".format(self.opts.method)

        bstr += '\n' + '_' * 75 + '\n'
        for k in range(self.d):
            bstr += ("\t{:<" + vnamelen +"s}: {:d} nodes in [{:6.2f}, {:6.2f}]\n").format(vnames[k], n[k], a[k], b[k])

        bstr += '\n' + '=' * 75 + '\n'
        bstr += "WARNING! Class Basis is still work in progress"
        return bstr


class OptBasis(Options_Container):
    """
    Options for Basis class.

    This class stores options for creating a Basis class. It takes care of validating options given by user to Basis constructor.
    """
    valid_types = ['chebyshev', 'spline', 'linear']
    valid_node_types = {'chebyshev': ['gaussian', 'lobatto', 'endpoint'],
                        'spline': ['canonical', 'user'],
                        'linear': ['canonical', 'user']}
    valid_methods = ['tensor', 'smolyak', 'complete', 'cluster', 'zcluster']


    def __init__(self, d, type='chebyshev', nodetype=None, k=3, method='tensor', qn=None, qp=None, labels=None, ix=[], ip=[]):
        """
        Make default options dictionary
        :param int d: dimension of the basis
        :return: an object with default values for Basis.opts
        """
        self.d = d
        self.type = type if type in self.valid_types else 'chebyshev'
        self.nodetype = nodetype
        self.k = k
        self._method = method
        self._qn = None
        self._qp = None
        self._labels = ["V{}".format(dim) for dim in range(d)]
        self._ix = []
        self._ip = []



    ''' Properties'''
    @property
    def type(self):
        """ Basis type (chebyshev, spline)"""
        return self._type

    @property
    def nodetype(self):
        """ type of nodes (gaussian, lobatto, endpoint) """
        return self._nodetype

    @property
    def method(self):
        """ method to expand the basis (tensor, smolyak, cluster, zcluster, complete) """
        return self._method

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
        """ list of available options in OptBasis """
        return([name for name in OptBasis.__dict__ if not name.startswith('_')])

    @property
    def qn(self):
        """ node parameter, to guide the selection of node combinations"""
        return self._qn

    @property
    def qp(self):
        """ polynomial parameter, to guide the selection of polynomial combinations"""
        return self._qp

    ''' Setters '''
    @type.setter
    def type(self, value):
        valid = ['chebyshev', 'spline']
        if value in valid:
            self._type = value
        else:
            raise ValueError('type value must be in ' + str(valid))

    @nodetype.setter
    def nodetype(self, value):
        valid_options = self.valid_node_types[self.type]
        if value:
            value = value.lower()
            if value in valid_options:
                self._nodetype = value
            else:
                warn('For a '+ self.type.capitalize() +
                     ' basis, nodetype value must be in ' +
                     str(valid_options) + '.\nUsing nodetype = ' +
                     valid_options[0] + 'instead.')
                self._nodetype = valid_options[0]
        else:
            self._nodetype = valid_options[0]

    @method.setter
    def method(self, value):
        valid_options = self.valid_methods
        value = value.lower()
        if value in valid_options:
            self._method = value
        else:
            warn('Expansion method must be one of ' +
                 str(valid_options) + '.\nUsing method = ' +
                 valid_options[0] + 'instead.')
            self._method = valid_options[0]

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

    @qn.setter
    def qn(self, value):
        self._qn = value

    @qp.setter
    def qp(self, value):
        self._qp = value

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
            self.ix = np.arange(n).reshape(1, -1)
            self.ip = np.arange(n).reshape(1, -1)
            return

        ''' Smolyak interpolation: done by SmolyakGrid function'''
        if self.method == 'smolyak':
            self.ix, self.ip = SmolyakGrid(n, self.qn, self.qp)

        ''' All other methods'''
        degs = n - 1 # degree of polynomials
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


def as_basis(B):
    if type(B) == Basis:
        return B
    elif type(B) == BasisChebyshev:
        return Basis(B.n, B.a, B.b, type='chebyshev')

