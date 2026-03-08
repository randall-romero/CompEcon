__author__ = 'Randall'
import numpy as np


class Smolyak(object):
    """
    A class to generate Smolyak grids
    """
    def __init__(self, n, qn, qp=None):
        self.n = np.asarray(n, int)
        self.d = self.n.size

        if qp is None:
            qp = qn

        self.qn = np.array([qn] * self.d, int) if isinstance(qn, (float, int)) else np.asarray(qn, int)
        self.qp = np.array([qp] * self.d, int) if isinstance(qp, (float, int)) else np.asarray(qp, int)












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