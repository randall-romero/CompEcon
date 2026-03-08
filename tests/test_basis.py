from nose.tools import *
from compecon.basis import Basis, SmolyakGrid
from compecon import BasisChebyshev, BasisSpline, BasisLinear
from compecon.tools import gridmake

import numpy as np

__author__ = 'Randall'

class TestBasisShapes:
    def test_cheby_shape(self):
        n, a, b = 7, 0, 2
        basis = BasisChebyshev(n, a, b)
        assert_equal(basis.nodes.size, n)
        assert_equal(basis.Phi().shape, (n, n))
        assert_equal(basis._diff(0, 2).shape, (n - 2, n))
        assert_equal(basis._diff(0, -3).shape, (n + 3, n))
        assert_equal(basis().shape, (n,))

        nx = 24
        x = np.linspace(a, b, nx)
        assert_equal(basis.Phi(x).shape, (nx, n))
        assert_equal(basis.Phi(x, 1).shape, (nx, n))
        assert_equal(basis.Phi(x, -1).shape, (nx, n))
        assert_equal(basis(x).shape, (nx,))


    def test_spli_shape(self):
        n, a, b = 35, -3, 3
        basis = BasisSpline(n, a, b)
        assert_equal(basis.nodes.size, n)
        assert_equal(basis.Phi().shape, (n, n))
        assert_equal(basis._diff(0, 2).shape, (n - 2, n))
        assert_equal(basis._diff(0, -3).shape, (n + 3, n))
        assert_equal(basis().shape, (n,))

        nx = 24
        x = np.linspace(a, b, nx)
        assert_equal(basis.Phi(x).shape, (nx, n))
        assert_equal(basis.Phi(x, 1).shape, (nx, n))
        assert_equal(basis.Phi(x, -1).shape, (nx, n))
        assert_equal(basis(x).shape, (nx,))

    def test_lin_shape(self):
        n, a, b = 35, -3, 3
        basis = BasisLinear(n, a, b)
        assert_equal(basis.nodes.size, n)
        assert_equal(basis.Phi().shape, (n, n))
        assert_equal(basis._diff(0, 2).shape, (n - 2, n))
        assert_equal(basis._diff(0, -3).shape, (n + 3, n))
        assert_equal(basis().shape, (n,))

        nx = 50
        x = np.linspace(a, b, nx)
        assert_equal(basis.Phi(x).shape, (nx, n))
        assert_equal(basis.Phi(x, 1).shape, (nx, n))
        assert_equal(basis.Phi(x, -1).shape, (nx, n))
        assert_equal(basis(x).shape, (nx,))

    def test_cheb_2d(self):
        n, a, b = [5, 9], -3, 3
        s = 2
        nn = n[0] * n[1]
        basis = BasisSpline(n, a, b, s=s)
        assert_equal(basis.nodes.shape, (len(n), nn))
        assert_equal(basis.Phi().shape, (nn, nn))
        assert_equal(basis._diff(0, 2).shape, (n[0] - 2, n[0]))
        assert_equal(basis._diff(1, -3).shape, (n[1] + 3, n[1]))
        assert_equal(basis().shape, (s, nn))

        nx = [15, 16]
        nnx = nx[0] * nx[1]
        x = gridmake([np.linspace(a, b, j) for j in nx])
        assert_equal(basis.Phi(x).shape, (nnx, nn))
        assert_equal(basis.Phi(x, [[1], [0]]).shape, (nnx, nn))
        assert_equal(basis.Phi(x, [[0], [-1]]).shape, (nnx, nn))
        assert_equal(basis(x).shape, (s, nnx))
