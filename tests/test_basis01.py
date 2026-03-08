from nose.tools import *

from compecon.basis import Basis, SmolyakGrid
import numpy as np









# BASIS 1: tensor product
n = [3, 5]
a = [-1, 0]
b = [1, 3]
oo = np.array([[0, 1, 0], [0, 0, 1]])  # first derivatives
Phi = Basis(n, a, b)
xlist = [np.linspace(a[0], b[0], 7), np.linspace(a[1], b[1], 9)]

N = 45
xmatrix = np.array([np.linspace(a[0], b[0], N), np.linspace(a[1], b[1], N)])
xmatrix.shape = (2, N)


def test_interpolation():
    assert_equal(Phi().shape, (15, 15))
    assert_equal(Phi(order=oo).shape, (3, 15, 15))
    assert_equal(Phi(xlist).shape, (63, 15))
    assert_equal(Phi(xlist, oo).shape, (3, 63, 15))
    assert_equal(Phi(xmatrix).shape, (N, 15))
    assert_equal(Phi(xmatrix,oo).shape, (3, N, 15))

# BASIS 2: smolyak
n = [9, 9, 5]
qn = 2
Phi2 = Basis(n, np.zeros(3), np.ones(3), method='smolyak')
nodes, polys = SmolyakGrid(np.array(n), qn)


def test_smolyak():
    assert_equal(nodes.shape, (3, 25))
    assert_equal(polys.shape, (3, 25))
    assert_equal(Phi2().shape, (25, 25))


# anisotropic grid
qna = [3, 2, 1]
nodes2, polys2 = SmolyakGrid(n, qna)


def test_smolyak():
    assert_equal(nodes2.shape, (3, 51))
    assert_equal(polys2.shape, (3, 51))

#Phi3 = Basis(n, a, b, method='complete', qp=6)
