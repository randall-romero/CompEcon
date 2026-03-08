# from .basis import Basis
# from .basisChebyshev import BasisChebyshev
#
# import numpy as np
# import copy
# '''
#         just the template from Matlab version.
#         Work in progress
#
#
#
#
#
#
# ## interpolator class
# # Defines a class to represent an approximated function
# #
# # Objects created by this class are a subclass of <basis.m basis>, adding fields to
# # identify a function and methods to interpolate, compute Jacobian and Hessian.
# #
# #
# # Apart from the properties inherited from <basis.m basis>, objects of class |interpolator|
# # have the following properties:
# #
# # * |y|: value of interpolated function at basis nodes
# # * |c|:  Phi coefficients
# # * |Phi|:   Phi matrix, evaluated at basis nodes
# # * |Phiinv|: inverse of |Phi|
# #
# # Object of class |funcApprox| have the following methods:
# #
# # * |funcApprox|: class constructor
# # * |updateCoef|: computes Phi coefficients if |y| is modified
# # * |Interpolate|: interpolates the function
# # * |Jacobian|: computes the Jacobian of the function
# # * |Hessian|: computes the Hessian of the function
# #
# #
# # *To date, only basisChebyshev has been implemented*, so calling the function with |type|
# # 'spli' or 'lin' returns an error.
# #
# # Last updated: November 24, 2014.
# #
# #
# # Copyright (C) 2015 Randall Romero-Aguilar
# #
# # Licensed under the MIT license, see LICENSE.txt
#
#
#
#     #properties (Dependent)
#     y = None   # value of functions at nodes
#     c = None   # Phi coefficients
#     x = None   # basis nodes
#     #properties (Access = protected)
#     fnodes_ = None # stored values for function at nodes
#     coef_ = None # stored coefficients
#     fnodes_is_outdated = None# if true, calling "y" updates fnodes_ before returning values
#     coef_is_outdated = None# if true, calling "c" updates coef_ before returning values
#     #properties (SetAccess = protected)
#     Phi = None  # Phi matrix
#     Phiinv = None # inverse of Phi matrix
# '''
#
#
# class Interpolator(object):
#     def __init__(self, B, y=None, c=None, f=None, s=1):
#
#         if type(B) is Basis:
#             self.B = B
#         else:
#             raise ValueError('B must be a Basis object')
#
#         # add data
#         self._y = None
#         self._c = None
#         self._yIsOutdated = None
#         self._cIsOutdated = None
#
#         if y is not None:
#             self.y = np.atleast_2d(y)
#         elif c is not None:
#             self.c = c
#         elif callable(f):
#             self.y = f(B.nodes)
#         else:
#             if type(s) is int:
#                 s = [s]
#             elif type(s) is np.ndarray:
#                 s = s.tolist()
#             s.append(B.N)
#             self.y = np.zeros(s)
#
#
#     """ setter and getter methods """
#
#     @property
#     def N(self):
#         """  :return: number of nodes """
#         return self.B.N
#
#     @property
#     def growth_model(self):
#         """  :return: number of polynomials """
#         return self.B.growth_model
#
#     @property
#     def x(self):
#         """  :return: interpolation nodes  """
#         return self.B.nodes
#
#     @property
#     def nodes(self):
#         """  :return: interpolation nodes  """
#         return self.B.nodes
#
#     @property
#     def Phi(self):
#         """ Interpolation matrix """
#         return self.B.Phi
#
#
#     def update_y(self):
#         self._y = np.dot(self._c, self.B._PhiT)
#         self._yIsOutdated = False
#
#     def update_c(self):
#         self._c = np.dot(self._y, self.B._PhiInvT)
#         self._cIsOutdated = False
#
#     @property
#     def shape(self):
#         if self._yIsOutdated:
#             return self._c.shape[:-1]
#         else:
#             return self._y.shape[:-1]
#
#     @property
#     def size(self):
#         return np.prod(self.shape)
#
#     @property
#     def ndim(self):
#         return len(self.shape) - 1
#
#     @property
#     def shape_N(self):
#         temp = list(self.shape)
#         temp.append(self.N)
#         return temp
#
#     def copy(self):
#         return copy.copy(self)
#
#     @property
#     def y(self):
#         """ :return: function values at nodes """
#         if self._yIsOutdated:
#             self.update_y()
#         return self._y
#
#     @property
#     def c(self):
#         """ :return: interpolation coefficients """
#         if self._cIsOutdated:
#             self.update_c()
#         return self._c
#
#     @y.setter
#     def y(self, val):
#         val = np.atleast_2d(np.asarray(val))
#         if val.shape[-1] != self.N:
#             raise ValueError('y must be an array with {} elements in its last dimension.'.format(self.N))
#         self._y = val
#         self._yIsOutdated = False
#         self._cIsOutdated = True
#
#     @c.setter
#     def c(self, val):
#         val = np.atleast_2d(np.asarray(val))
#         if val.shape[-1] != self.growth_model:
#             raise ValueError('c must be an array with {} elements in its last dimension.'.format(self.growth_model))
#         self._c = val
#         self._cIsOutdated = False
#         self._yIsOutdated = True
#
#     def __getitem__(self, item):
#         other = self.copy()
#         other._y = None if self._y is None else self._y[item]
#         other._c = None if self._c is None else self._c[item]
#         return other
#
#     def __setitem__(self, key, value):
#         if isinstance(value, Interpolator) and value.B.N == self.N:
#             self._y[key] = value.y
#             self._c[key] = value.c
#         else:
#             raise ValueError('value must be an Interpolator with {} nodes'.format(self.N))
#
#
#
#     """  Interpolation method """
#
#     def __call__(self, x=None, order=None):
#         """
#
#         :param x:
#         :param order:
#         :return:
#         """
#         d = self.B.d
#
#         if type(order) is str:
#             if order is 'jac':
#                 order_array = np.identity(d, int)
#             elif order is 'hess':
#                 order_array, mapping = hess_order(d)
#             elif order is 'all':
#                 order_array, mapping = hess_order(d)
#                 order_array = np.hstack([np.zeros([d, 1],int), np.identity(d,int), order_array])
#                 mapping += 1 + d
#             else:
#                 raise NotImplementedError
#         else:
#             order_array = order
#             order = 'none' if (order is None) else 'provided'
#
#         Phix = self.B.Phi(x, order_array)
#         if Phix.ndim == 2:
#             Phix = Phix[np.newaxis]
#
#         cPhix = np.array([np.dot(self.c, phix.T) for phix in Phix])
#
#         if order is 'none':
#             return cPhix[0]
#         elif order in ['provided', 'jac']:
#             return cPhix
#         elif order in ['hess', 'all']:
#             new_shape = [d, d] + list(cPhix.shape[1:])
#             Hess = np.zeros(new_shape)
#             for i in range(d):
#                 for j in range(d):
#                     Hess[i,j] = cPhix[mapping[i,j]]
#             if order is 'hess':
#                 return Hess
#             else:
#                 return cPhix[0], cPhix[1:(1 + d)], Hess
#         else:
#             raise ValueError
#
#
#
#
#
#
#
# ''' ADDITIONAL FUNCTIONS'''
# def hess_order(n):
#     """
#     Returns orders required to evaluate the Hessian matrix for a function with n variables
#     and location of hessian entries in resulting array
#     """
#     A = np.array([a.flatten() for a in np.indices(3*np.ones(n))])
#     A = A[:,A.sum(0)==2]
#
#     C = np.zeros([n,n])
#     for i in range(n):
#         for j in range(n):
#             v = np.zeros(n)
#             v[i] += 1
#             v[j] += 1
#             C[i,j] = (v==A.T).all(1).nonzero()[0]
#
#     return A, C
