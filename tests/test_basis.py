from nose.tools import *
from compecon.basis import Basis, SmolyakGrid
from compecon.basisChebyshev import BasisChebyshev

import numpy as np

__author__ = 'Randall'

class TestBasisChebyshev:
    def test_size(self):
        n, a, b = 7, 0, 2
        Phi = BasisChebyshev(n, a, b)
        assert_equal(Phi.nodes.size, n)
        assert_equal(Phi().shape, (n, n))
        assert_equal(Phi._Diff_(2).shape, (n - 2, n))
        assert_equal(Phi._Diff_(-3).shape, (n + 3, n))

        nx = 24
        x = np.linspace(a, b, nx)
        assert_equal(Phi(x).shape, (nx, n))
        assert_equal(Phi(x, 1).shape, (nx, n))
        assert_equal(Phi(x, -1).shape, (nx, n))

