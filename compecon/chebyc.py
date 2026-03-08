# __author__ = 'Randall'
# import numpy as np
# from numba import jit, float64, void
#
#
# ''' Usual Python version '''
#
# def cheby(z, bas):
#     #bas[0] = 1
#     bas[1] = z
#     z *= 2
#     for k in range(2, bas.shape[0]):
#         bas[k] = np.multiply(z, bas[k - 1]) - bas[k - 2]
#     return None
#
# ''' Numba optimized version'''
#
# @jit(void(float64[:], float64[:, :]), nopython=True)
# def cheby_numba(z, bas):
#     for node in range(z.size):
#         bas[0, node] = 1
#         bas[1, node] = z[node]
#         z[node] *= 2
#         for k in range(bas.shape[0]):
#             bas[k, node] = z[node] * bas[k - 1, node] - bas[k - 2, node]
#     return None
#
#
# def test(func, n):
#     zval = np.linspace(-1.0, 1.0, n)
#     npolys = n
#     phi = np.ones((npolys, n))
#     func(zval, phi)
#     return None
#
# test(cheby, 18)
# test(cheby_numba, 18)