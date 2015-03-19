import numpy as np
from .basisChebyshev import BasisChebyshev
from .ce_util import gridmake, ckron

__author__ = 'Randall'



class Basis:
    """
      A multivariate interpolation basis
    """

    def __init__(self, n, a, b, **options):
        """
        A multivariate basis

        :param n: number of nodes (:math:`1\\times d` vector)
        :param a: lower bounds (:math:`1\\times d` vector)
        :param b: upper bounds (:math:`1\\times d` vector)
        :param options: keyword-separated list of additional options (see below)
        :return: a Basis object
        """
        n = np.array([n] if np.isscalar(n) else n, 'int')
        a = np.array([a] if np.isscalar(a) else a)
        b = np.array([b] if np.isscalar(b) else b)

        d = a.size  # dimension of basis

        # todo use same number of nodes in all dimension if n is scalar
        if not np.all(a < b):
            raise ValueError('All lower bounds must be less than their corresponding upper bound')

        # Make default options dictionary
        opts = dict(
            type='chebyshev',
            nodetype='gaussian',
            k=3,
            method='tensor',
            qn=None,
            qp=None,
            varnames=["V{}".format(dim) for dim in range(d)]
        )

        valid_opts = dict(
            type={'chebyshev', 'spline'},
            nodetype={'gaussian', 'lobatto', 'endpoint', 'cardinal'},   # todo: cardinal is only valid in spline basis
            k=range(min(n)),
            method={'tensor', 'smolyak', 'complete', 'cluster', 'zcluster'},
            qn=range(np.prod(n)),
            qp=range(np.prod(n))
        )

        # get variable names, if provided
        if 'varnames' in options.keys():
            if len(options['varnames']) == d:
                opts["varnames"] = options['varnames']
            else:
                print('If provided, option varnames must have {} elements'.format(d))
            del options['varnames']

        # Read user options
        for opt, val in options.items():
            if opt not in opts.keys():
                print('Unknown option {} with value {}; ignoring'.format(opt, val))  # todo: make this a warning
            elif val not in valid_opts[opt]:
                print('Value {} is not a valid option for {}; ignoring'.format(val, opt))  # todo: make this a warning
            else:
                opts[opt] = val

        # Validate options for chebyshev basis of several dimensions
        if d > 1 and opts['type'] == 'chebyshev':
            if opts['method'] in ['complete', 'cluster', 'zcluster']:
                if opts['qn'] is None:
                    opts['qn'] = 0
                if opts['qp'] is None:
                    opts['qp'] = max(n) - 1
            elif opts['method'] == 'smolyak':
                n_valid = 2 ** np.ceil(np.log2(n - 1)) + 1
                if np.any(n != n_valid):
                    # todo: make this a warning
                    print('Warning: For smolyak expansion, number of nodes should be n = 2^k+1 for some k=1,2,...')
                    print('Adjusting nodes\n {:7s}  {:7s}'.format('old n', 'new n'))
                    for n1, n2 in zip(n,n_valid):
                        print('{:7.0f} {:7.0f}'.format(n1, n2))
                    n = np.array(n_valid,'int')
                if opts['nodetype'] != 'lobatto':
                    opts['nodetype'] = 'lobatto'  # todo issue warning
                if opts['qn'] is None:
                    opts['qn'] = np.array([2])
                if opts['qp'] is None:
                    opts['qp'] = opts['qn']

        # make list of 1-basis
        B1 = []
        nodetype, varnames = opts['nodetype'], opts['varnames']

        if opts['type'] == 'chebyshev':
            for i in range(d):
                B1.append(BasisChebyshev(n[i], a[i], b[i], nodetype, varnames[i]))

        # Pack value in object
        self.a = a
        self.b = b
        self.n = n
        self.d = d
        self.opts = opts
        self._B1 = B1
        self.type = opts['type']
        self._expandBasis()

        # todo: expand basis

    def _expandBasis(self):
        """
             ExpandBasis computes nodes for multidimensional basis and other auxiliary fields required to keep track
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
        if self.d == 1:
            return

        ''' Smolyak interpolation: Now it is done by SmolyakGrid function'''
        n, qn, qp = self.n, self.opts['qn'], self.opts['qp']

        if self.opts['method'] == 'smolyak':
            self.opts['validX'], self.opts['validPhi'] = SmolyakGrid(n, qn, qp)
            self.nodes = np.zeros(self.opts['validX'].shape)
            for k in range(self.d):
                self.nodes[:,k] = self._B1[k].nodes[self.opts['validX'][:,k]].flatten()
            return

        ''' All other methods'''
        degs = self.n - 1 # degree of polynomials
        ldeg = [np.arange(degs[ni] + 1) for ni in range(self.d)]

        idxAll = gridmake(*ldeg)   # degree of polynomials = index


        ''' Expanding the polynomials'''
        if self.opts['method'] == 'tensor':
            self.opts['validPhi'] = idxAll
        else:
            degValid = np.sum(idxAll, axis=1) <= self.opts['qp']
            self.opts['validPhi'] = idxAll[degValid, :]

        ''' Expanding the nodes'''
        nodes1 = [self._B1[k].nodes for k in range(self.d)]
        nodes_tensor = gridmake(*nodes1)

        if self.opts['method'] in ['tensor', 'complete']:
            self.nodes = nodes_tensor
            self.opts['validX'] = idxAll
        elif self.opts['method'] in ['cluster', 'zcluster']:
            H = self.opts['validPhi'].size[0] + self.opts['qn']
            raise NotImplementedError # todo: implement this method


    def interpolation(self, x=None, order=None):
        if self.d == 1:
            return self._B1(x, order)

        if order is None:
            order = np.zeros(1, self.d)
        else:
            assert (order.shape(1) == self.d)

        # check what type of input x is provided
        if x is None:
            return self._interp_default(order)
        elif type(x) == list:
            return self._interp_list(x,order)
        elif type(x) == np.array:
            return self._interp_matrix(x,order)
        else:
            raise Exception('wrong x input')




        # HANDLE POLYNOMIALS DIMENSION



    def _interp_default(self, order):
        """

        :param x:
        :param order:
        :return:
        """
        x = self.nodes
        nrows = self.N
        if self.opts['method'] not in {'cluster', 'zcluster'}:
            r = self.opts['validX']  # combine default basis nodes

        c = self.opts['validPhi']
        ncols = c.shape(0)
        Norders = order.shape(0)

        if self.opts['type'] == 'Chebyshev':
            PHI = np.zeros([nrows, ncols, Norders, self.d])
            # Compute interpolation matrices for each dimension
            for k in range(self.d):
                Phij = self._B1[k](order=order[:, k])
                PHI[:, :, :, k] = Phij[r[:, k], c[:, k], :]
        else:
            raise NotImplemented

        # todo multiply 4th dimension
        return PHI






    def _interp_matrix(self, x, order):
        """

        :param x:
        :param order:
        :return:
        """
        assert (x.shape(1) == self.d)  # 'In Interpolation, class basis: x must have d columns')
        nrows = x.shape(0)
        r = np.tile(np.arange(x.shape(0)), [self.d, 1])

        c = self.opts['validPhi']
        ncols = c.shape(0)
        Norders = order.shape(0)

        if self.opts['type'] == 'Chebyshev':
            PHI = np.zeros([nrows, ncols, Norders, self.d])
            # Compute interpolation matrices for each dimension
            for k in range(self.d):
                Phij = self._B1[k](x[:, k], order[:, k])
                PHI[:, :, :, k] = Phij[:, c[:,k], :]
        else:
            raise NotImplemented

        # todo multiply 4th dimension
        return PHI





    def _interp_list(self, x, order):
        """

        :param x:
        :param order:
        :return:
        """
        assert (len(x) == self.d)
        nrows = np.array([xi.size for xi in x]).prod()
        r = self.opts['validX']  # combine default basis nodes

        c = self.opts['validPhi']
        ncols = c.shape(0)
        Norders = order.shape(0)

        if self.opts['type'] == 'Chebyshev':
            PHI = np.zeros([nrows, ncols, Norders, self.d])
            # Compute interpolation matrices for each dimension
            for k in range(self.d):
                Phij = self._B1[k](x[k], order[:, k])
                PHI[:, :, :, k] = Phij[r[:, k], c[:, k], :]
        else:
            raise NotImplemented

        # todo multiply 4th dimension
        return PHI











    def plot(self):
        raise NotImplementedError # todo: implement this method

    @property
    def N(self):
        """ Total number of nodes"""
        return self.opts['validX'].shape[0]

    @property
    def M(self):
        """ Total number of polynomials"""
        return self.opts['validPhi'].shape[0]


    def __repr__(self):
        n, a, b = self.n, self.a, self.b
        nodetype = self.opts['nodetype']
        vnames = self.opts['varnames']

        bstr = "A {}-dimension basis function:  ".format(self.d)
        bstr += "using {:d} {} nodes and {:d} polynomials, expanded by {}".format(
            self.N, nodetype.upper(), self.M, self.opts['method'])
        bstr += '\n' + '_' * 60 + '\n'
        for k in range(self.d):
            bstr += "\t{:12s}: {:d} nodes in [{:6.2f}, {:6.2f}]\n".format(vnames[k], n[k], a[k], b[k])

        bstr += '\n' + '=' * 60 + '\n'
        bstr += "WARNING! Class Basis is still work in progress"
        return bstr





def SmolyakGrid(n, qn, qp=None):
    """

    :param n: number of nodes per dimension
    :param qn: cut-off parameters for node selection (array)
    :param qp: cut-off parameters for polynomial selection(array)
    :return: a (node_indices, polynomial_indices) tuple to form the Smolyak grid from univariate nodes and polynomials
    """

    if qp is None:
        qp = qn

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

    return theNodes, thePolys




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