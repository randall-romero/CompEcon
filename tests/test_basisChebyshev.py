from nose.tools import *
import numpy as np
from pycompecon.basisChebyshev import BasisChebyshev


''' test for size of Diff operator '''
n, a, b = 7, 0, 2
Phi = BasisChebyshev(n, a, b)


def test_Diff():
    assert_equal(Phi.Diff(2).shape, (n - 2, n))
    assert_equal(Phi.Diff(-3).shape, (n + 3, n))


''' test for shape of interpolation matrix '''
nx = 24
x = np.linspace(a, b, nx)


def test_Phi():
    assert_equal(Phi(x).shape, (nx, n))
    assert_equal(Phi().shape, (n, n))
