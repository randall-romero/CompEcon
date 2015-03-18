from nose.tools import *

import pycompecon as econ
import numpy as np


def test_smolyak():
    n = np.array([9, 9, 5])
    qn = np.array([2])
    nodes, polys = econ.basis.SmolyakGrid(n, qn)
    nn, d = nodes.shape
    assert_equal(nodes.shape[0], 25)
    assert_equal(nodes.shape[1], 3)
    # anisotropic grid
    qn = np.array([3,2,1])
    nodes, polys = econ.basis.SmolyakGrid(n, qn)
    assert_equal(polys.shape[0], 51)