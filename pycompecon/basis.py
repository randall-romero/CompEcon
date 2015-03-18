import numpy as np
from .basisChebyshev import BasisChebyshev


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
            qnode=None,
            qdegree=None,
            varnames=["V{}".format(dim) for dim in range(d)]
        )

        valid_opts = dict(
            type={'chebyshev', 'spline'},
            nodetype={'gaussian', 'lobatto', 'endpoint','cardinal'},   #todo: cardinal is only valid in spline basis
            k=range(min(n)),
            method={'tensor', 'smolyak', 'complete', 'cluster', 'zcluster'},
            qnode=range(min(n)),
            qdegree=range(min(n)),
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
                print('Value {} is not a valid option for {}; ignoring'.format(val,opt))  # todo: make this a warning
            else:
                opts[opt] = val

        # Validate options for chebyshev basis of several dimensions
        if d > 1 and opts['type'] == 'chebyshev':
            if opts['method'] in ['complete', 'cluster', 'zcluster']:
                if opts['qnode'] is None:
                    opts['qnode'] = 0
                if opts['qdegree'] is None:
                    opts['qdegree'] = max(n) - 1
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
                if opts['qnode'] is None:
                    opts['qnode'] = 2
                if opts['qdegree'] is None:
                    opts['qdegree'] = opts['qnode']

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
             * 'complete', 'cluster', and 'zcluster' choose polynomials with degrees not exceeding opts.qdegree
            
             Expanding nodes depends on value of field opts.method
            
             * 'tensor' and 'complete' take all possible combinations,
             * 'smolyak' computes Smolyak basis, given opts.qnode
             * 'cluster' and 'zcluster' compute clusters of the tensor nodes based on opts.qnode

        :return: None
        """
        if self.d == 1:
            return

        # Smolyak interpolation: Now it is done by SmolyakGrid function.
        n, qnode, qdegree = self.n, self.opts.qnode, self.opts.qdegree

        if self.opts.met == 'smolyak':
            self.opts['validX'], self.opts['validPhi'] = SmolyakGrid(n, qnode, qdegree)
            self.nodes = np.zeros(self.opts.validX.shape)
            for k in range(self.d):
                self.nodes[:,k] = self.B1[k].nodes[self.opts.validX[:,k]]
            return





        raise NotImplementedError # todo: implement this method






    def interpolation(self,x,order):
        raise NotImplementedError # todo: implement this method

    def plot(self):
        raise NotImplementedError # todo: implement this method

    def __repr__(self):
        return "WARNING! Class Basis is still work in progress"


def SmolyakGrid(n, qnode, qdegree=None):
    """

    :param n: number of nodes per dimension
    :param qnode: cut-off parameters for node selection (array)
    :param qdegree: cut-off parameters for polynomial selection(array)
    :return: a (node_indices, polynomial_indices) tuple to form the Smolyak grid from univariate nodes and polynomials
    """

    if qdegree is None:
        qdegree = qnode

    # Dimensions
    d = n.size
    ngroups = np.log2(n - 1) + 1
    N = max(ngroups)

    # node parameter
    node_q = max(qnode)
    node_isotropic = qnode.size == 1
    if node_isotropic:
        qnode = np.zeros(d)

    # polynomial parameter
    poly_q = max(qdegree)
    poly_isotropic = qdegree.size == 1
    if poly_isotropic:
        qdegree = np.zeros(d)

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
        isValid = nodeSum <= (qnode[0] + 1)  # todo: not sure this index is ok
        nodeSum = nodeSum[isValid]
        theNodes = theNodes[isValid,:]

    # set up polynomials for first dimension
    polySum = polyMapping[0]
    thePolys = np.mat(nodeIndex[0]).T
    if not poly_isotropic:
        isValid = polySum <= (qdegree[0] + 1)  # todo: not sure this index is ok
        polySum = polySum[isValid]
        thePolys = thePolys[isValid,:]

    # compute the grid
    for k in range(1,d):
        theNodes, nodeSum = ndgrid2(theNodes,nodeSum,nodeMapping[k], 1 + k + node_q, qnode[k])
        thePolys, polySum = ndgrid2(thePolys,polySum,polyMapping[k], 1 + k + poly_q, qdegree[k])

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