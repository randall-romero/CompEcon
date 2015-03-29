import numpy as np
from .basisChebyshev import BasisChebyshev
from .ce_util import gridmake, ckron

__author__ = 'Randall'


class Basis(object):
    """
      A multivariate interpolation basis
    """

    def __init__(self, n=None, a=None, b=None, **options):
        """
        A multivariate basis

        :param n: number of nodes (:math:`1\\times d` vector)
        :param a: lower bounds (:math:`1\\times d` vector)
        :param b: upper bounds (:math:`1\\times d` vector)
        :param options: keyword-separated list of additional options (see below)
        :return: a Basis object
        """

        if n is None:
            return

        # Convert n, a, and b to np.arrays
        n = asArray(n)
        a = asArray(a)
        b = asArray(b)
        n = np.asarray(n,'int')

        d = a.size  # dimension of basis

        # todo use same number of nodes in all dimension if n is scalar
        if not np.all(a < b):
            raise ValueError('All lower bounds must be less than their corresponding upper bound')

        # Read user options
        opts = OptBasis(d)
        for option, value in options.items():
            if option not in opts.keys:
                print('Unknown option {} with value {}; ignoring'.format(option, value))  # todo: make this a warning
            else:
                setattr(opts, option, value)

        # Validate options for chebyshev basis of several dimensions
        if opts.type == 'chebyshev':
            opts.validateChebyshev(n)



        # make list of 1-basis
        nodetype, varnames = opts.nodetype, opts.varnames

        if opts.type == 'chebyshev':
            B1 = [BasisChebyshev(n[i], a[i], b[i], nodetype, varnames[i]) for i in range(d)]

        # Pack value in object
        self.a = a
        self.b = b
        self.n = n
        self.d = d
        self._B1 = B1
        self.opts = opts
        self.opts.expandGrid(n)
        self.type = opts.type
        self.nodes = np.array([self._B1[k].nodes[self.opts.validX[k]].flatten() for k in range(self.d)])

    def interpolation(self, x=None, order=None):
        if self.d == 1:
            return self._B1[0](x, order)

        if order is None:
            order = np.zeros([self.d, 1], 'int')
        else:
            order = np.asarray(order)
            if order.ndim == 1:
                order.reshape([order.size, 1])
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
        return self.interpolation(x, order)




    def _interp_default(self, order):
        """

        :param order:
        :return:
        """
        r = self.opts.validX
        c = self.opts.validPhi
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
        c = self.opts.validPhi
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
        r = gridmake(*[np.arange(xi.size) for xi in x]).T
        c = self.opts.validPhi
        oo = np.arange(order.shape[1])

        if self.opts.type == 'chebyshev':
            PHI = [self._B1[k](x[k], order[k])[np.ix_(oo, r[k], c[k])] for k in range(self.d)]
        else:
            raise NotImplemented
        return np.product(PHI,0)











    def plot(self):
        raise NotImplementedError # todo: implement this method

    @property
    def N(self):
        """ Total number of nodes"""
        return self.opts.validX.shape[-1]

    @property
    def M(self):
        """ Total number of polynomials"""
        return self.opts.validPhi.shape[-1]


    def __repr__(self):
        if self.d == 1:
            return self._B1[0].__repr__()

        n, a, b = self.n, self.a, self.b
        nodetype = self.opts.nodetype
        vnames = self.opts.varnames

        bstr = "A {}-dimension basis function:  ".format(self.d)
        bstr += "using {:d} {} nodes and {:d} polynomials, expanded by {}".format(
            self.N, nodetype.upper(), self.M, self.opts.method)
        bstr += '\n' + '_' * 60 + '\n'
        for k in range(self.d):
            bstr += "\t{:12s}: {:d} nodes in [{:6.2f}, {:6.2f}]\n".format(vnames[k], n[k], a[k], b[k])

        bstr += '\n' + '=' * 60 + '\n'
        bstr += "WARNING! Class Basis is still work in progress"
        return bstr


class OptBasis(object):

    def __init__(self, d):
        # Make default options dictionary
        self._d = d
        self._type = 'chebyshev'
        self._nodetype = 'gaussian'
        self._k = 3
        self._method = 'tensor'
        self._qn = None
        self._qp = None
        self._varnames = ["V{}".format(dim) for dim in range(d)]
        self._validX = []
        self._validPhi = []

    ''' Properties'''
    @property
    def type(self):
        return self._type

    @property
    def nodetype(self):
        return self._nodetype

    @property
    def method(self):
        return self._method

    @property
    def validX(self):
        return self._validX

    @property
    def validPhi(self):
        return self._validPhi

    @property
    def validPhi(self):
        return self._validPhi

    @property
    def varnames(self):
        return self._varnames

    @property
    def keys(self):
        return([name for name in OptBasis.__dict__ if not name.startswith('_')])

    @property
    def qn(self):
        return self._qn

    @property
    def qp(self):
        return self._qp


    ''' Setters '''
    @type.setter
    def type(self, value):
        valid = ['chebyshev', 'spline']
        print(value, valid, value in valid)
        if value in valid:
            self._type = value
        else:
            raise ValueError('type value must be in ' + str(valid))


    @nodetype.setter
    def nodetype(self, value):
        valid = ['gaussian', 'lobatto', 'endpoint', 'cardinal']
        if value in valid:
            self._nodetype = value
        else:
            raise ValueError('nodetype value must be in ' + str(valid))

    @method.setter
    def method(self, value):
        valid = ['tensor', 'smolyak', 'complete', 'cluster', 'zcluster']
        if value in valid:
            self._method = value
        else:
            raise ValueError('method value must be in ' + str(valid))

    @validX.setter
    def validX(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == self._d:
            self._validX = value
        else:
            raise ValueError('validX must be a 2-dimensional numpy array with {} rows'.format(self._d))

    @validPhi.setter
    def validPhi(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == self._d:
            self._validPhi = value
        else:
            raise ValueError('validPhi must be a 2-dimensional numpy array with {} rows'.format(self._d))

    @varnames.setter
    def varnames(self, value):
        if isinstance(value, list) and len(value) == self._d:
            self._varnames = value
        else:
            raise ValueError('varnames must be a list of {} strings'.format(self._d))

    @qn.setter
    def qn(self, value):
        self._qn = value

    @qp.setter
    def qp(self, value):
        self._qp = value



    def validateChebyshev(self, n):
        if self._d == 1:
            return

        if self.method in ['complete', 'cluster', 'zcluster']:
            if self.qn is None:
                self.qn = 0
            if self.qp is None:
                self.qp = max(n) - 1
        elif self.method == 'smolyak':
            n_valid = 2 ** np.ceil(np.log2(n - 1)) + 1
            if np.any(n != n_valid):
                # todo: make this a warning
                print('Warning: For smolyak expansion, number of nodes should be n = 2^k+1 for some k=1,2,...')
                print('Adjusting nodes\n {:7s}  {:7s}'.format('old n', 'new n'))
                for n1, n2 in zip(n, n_valid):
                    print('{:7.0f} {:7.0f}'.format(n1, n2))
                n = np.array(n_valid,'int')
            if self.nodetype != 'lobatto':
                self.nodetype = 'lobatto'  # todo issue warning
            if self.qn is None:
                self.qn = asArray(2)
            else:
                self.qn = asArray(self.qn)
            if self.qp is None:
                self.qp = self.qn
            else:
                self.qp = asArray(self.qp)


    def expandGrid(self, n):
        """
             Compute nodes for multidimensional basis and other auxiliary fields required to keep track
             of the basis (how to combine unidimensional nodes bases). It is called by the constructor method, and as
             such is not directly needed by the user. ExpandBasis updates the following fields

             * nodes: matrix with all nodes, one column by dimension
             * opts.validPhi: indices to combine unidimensional bases
             * B.opts.validX: indices to combine unidimensional nodes

             Combining polynomials depends on value of input opts.method:

             * 'tensor' takes all possible combinations,
             * 'smolyak' computes Smolyak basis, given |opts.degreeParam|,
             * 'complete', 'cluster', and 'zcluster' choose polynomials with degrees not exceeding opts.qp

             Expanding nodes depends on value of field opts.method

             * 'tensor' and 'complete' take all possible combinations,
             * 'smolyak' computes Smolyak basis, given opts.qn
             * 'cluster' and 'zcluster' compute clusters of the tensor nodes based on opts.qn

        :return: None
        """
        if self._d == 1:
            self.validX = np.arange(n).reshape(1, -1)
            self.validPhi = np.arange(n).reshape(1, -1)
            return

        ''' Smolyak interpolation: done by SmolyakGrid function'''
        if self.method == 'smolyak':
            self.validX, self.validPhi = SmolyakGrid(n, self.qn, self.qp)

        ''' All other methods'''
        degs = n - 1 # degree of polynomials
        ldeg = [np.arange(degs[ni] + 1) for ni in range(self._d)]

        idxAll = gridmake(*ldeg).T   # degree of polynomials = index

        ''' Expanding the polynomials'''
        if self.method == 'tensor':
            self.validPhi = idxAll
        else:
            degValid = np.sum(idxAll, axis=0) <= self.qp
            self.validPhi = idxAll[:, degValid]

        ''' Expanding the nodes'''
        if self.method in ['tensor', 'complete']:
            self.validX = idxAll
        elif self.method in ['cluster', 'zcluster']:
            raise NotImplementedError # todo: implement this method


def SmolyakGrid(n, qn, qp=None):
    """

    :param n: number of nodes per dimension
    :param qn: cut-off parameters for node selection (array)
    :param qp: cut-off parameters for polynomial selection(array)
    :return: a (node_indices, polynomial_indices) tuple to form the Smolyak grid from univariate nodes and polynomials
    """

    if qp is None:
        qp = qn

    if isinstance(qn,(float, int)):
        qn = np.array([qn])

    if isinstance(qp, (float, int)):
        qp = np.array([qp])




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

    # make disjoint sets
    nodeMapping = []
    polyMapping = []
    nodeIndex = []

    for i in range(d):
        nodeMapping.append(g[g <= ngroups[i]])
        polyMapping.append(np.sort(nodeMapping[i]))
        nodeIndex.append(np.arange(n[i]))

    # set up nodes for first dimension
    nodeSum = nodeMapping[0]
    theNodes = np.mat(nodeIndex[0]).T
    if not node_isotropic:
        isValid = nodeSum <= (qn[0] + 1)  # todo: not sure this index is ok
        nodeSum = nodeSum[isValid]
        theNodes = theNodes[isValid,:]

    # set up polynomials for first dimension
    polySum = polyMapping[0]
    thePolys = np.mat(nodeIndex[0]).T
    if not poly_isotropic:
        isValid = polySum <= (qp[0] + 1)  # todo: not sure this index is ok
        polySum = polySum[isValid]
        thePolys = thePolys[isValid,:]

    # compute the grid
    for k in range(1,d):
        theNodes, nodeSum = ndgrid2(theNodes,nodeSum,nodeMapping[k], 1 + k + node_q, qn[k])
        thePolys, polySum = ndgrid2(thePolys,polySum,polyMapping[k], 1 + k + poly_q, qp[k])

    return theNodes.T, thePolys.T


def ndgrid2(Indices,groupsum,newGroup,q,qk):
    """
    Expanding a Smolyak grid, 2 dimensions

    :param Indices: Previous iteration smolyak grid
    :param groupsum: sum of indices from previous iteration
    :param newGroup: new indices to be combined with Indices
    :param q: cutt-off parameter for new sum of indices
    :param qk: adjustment for anisotropic grids
    :return: Updated "Indices" and "groupsum"
    """
    nx, ny = np.arange(groupsum.size), np.arange(newGroup.size)
    idx, idy = np.meshgrid(nx,ny)
    idx, idy = idx.flatten(0), idy.flatten(0)
    newSum = groupsum[idx] + newGroup[idy]
    isValid = newSum <= q
    if qk != 0: #anisotropic
        isValid = np.logical_and(isValid,newGroup[idy] <= qk + 1)

    newSum = newSum[isValid]
    idxv = idx[isValid]
    idyv = np.mat(idy[isValid]).T
    newIndices = np.concatenate((Indices[idxv, :], idyv), axis=1)
    return newIndices, newSum


def asArray(param):
            if np.isscalar(param):
                param = [param]
            return np.asarray(param)