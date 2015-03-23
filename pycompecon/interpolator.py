import copy
from compecon import Basis
import numpy as np
from scipy import linalg

'''
        just the template from Matlab version.
        Work in progress






## interpolator class
# Defines a class to represent an approximated function
#
# Objects created by this class are a subclass of <basis.m basis>, adding fields to
# identify a function and methods to interpolate, compute Jacobian and Hessian.
#
#
# Apart from the properties inherited from <basis.m basis>, objects of class |interpolator|
# have the following properties:
#
# * |y|: value of interpolated function at basis nodes
# * |c|:  interpolation coefficients
# * |Phi|:   interpolation matrix, evaluated at basis nodes
# * |Phiinv|: inverse of |Phi|
#
# Object of class |funcApprox| have the following methods:
#
# * |funcApprox|: class constructor
# * |updateCoef|: computes interpolation coefficients if |y| is modified
# * |Interpolate|: interpolates the function
# * |Jacobian|: computes the Jacobian of the function
# * |Hessian|: computes the Hessian of the function
#
#
# *To date, only basisChebyshev has been implemented*, so calling the function with |type|
# 'spli' or 'lin' returns an error.
#
# Last updated: November 24, 2014.
#
#
# Copyright (C) 2014 Randall Romero-Aguilar
#
# Licensed under the MIT license, see LICENSE.txt



    #properties (Dependent)
    y = None   # value of functions at nodes
    c = None   # interpolation coefficients
    x = None   # basis nodes
    #properties (Access = protected)
    fnodes_ = None # stored values for function at nodes
    coef_ = None # stored coefficients
    fnodes_is_outdated = None# if true, calling "y" updates fnodes_ before returning values
    coef_is_outdated = None# if true, calling "c" updates coef_ before returning values
    #properties (SetAccess = protected)
    Phi = None  # interpolation matrix
    Phiinv = None # inverse of interpolation matrix
'''



class Interpolator(Basis):
    def __init__(self, *args, y=None, **kwargs):
        if type(args[0]) == Basis:
            B = args[0]
            Basis.__init__(self, B.n, B.a, B.b, **B.opts)
        else:
            Basis.__init__(self,*args, **kwargs)

        # Compute interpolation matrix at nodes
        _PhiT = self.interpolation().T
        self._PhiT = _PhiT

        # Compute inverse if not spline
        if self.type == 'chebyshev':
            self._PhiInvT = linalg.solve(np.dot(_PhiT, _PhiT.T), _PhiT).T
        else:
            raise NotImplementedError

        # add data
        self._y = np.zeros([self.N]) if y is None else y
        self._c = np.dot(self._y, self._PhiInvT)
        self._oldy = copy.copy(self._y)
        self._oldc = copy.copy(self._c)



    """ setter and getter methods """

    @property
    def _yIsOutdated(self):
        """ :return: True iff _c and _oldc have same shape and values """
        return not np.array_equal(self._c, self._oldc)

    @property
    def _cIsOutdated(self):
        """ :return: True iff _y and _oldy have same shape and values """
        return not np.array_equal(self._y, self._oldy)

    @property
    def y(self):
        """ :return: function values at nodes """
        if self._yIsOutdated:
            self._y = np.dot(self._c, self._PhiT)
            self._oldy = copy.copy(self._y)
            self._oldc = copy.copy(self._c)
        return self._y

    @property
    def c(self):
        """ :return: interpolation coefficients """
        if self._cIsOutdated:
            self._c = np.dot(self._y, self._PhiInvT)
            self._oldc = copy.copy(self._c)
            self._oldy = copy.copy(self._y)
        return self._c

    @property
    def x(self):
        """  :return: interpolation nodes  """
        return self.nodes


    @y.setter
    def y(self, val):
        print('calling the y setter')
        if val.ndim == 1:
            assert(val.size == self.N)
        elif val.shape[-1] != self.N:
            raise ValueError('last index of value must have {} values (one for each node)'.format(self.N))
        self._y = val

    @c.setter
    def c(self, val):
        print('calling the c setter')
        assert (val.shape[-1] == self.M)  # one value per polynomial
        self._c = val




    """  Interpolation method """

    def __call__(self,x = None, order = None):
        """

        :param x:
        :param order:
        :return:
        """

        if order is None:
            order = np.zeros([self.d, 1], 'int')
        else:
            assert (order.shape[0] == self.d)

        Phix = self.interpolation(x, order)

        orderIsScalar = order.shape[1] == 1
        if orderIsScalar:
            return np.dot(self.c, Phix.T)
        else:
            return np.array([np.dot(self.c, Phix[k].T) for k in range(order.shape[1])])



    def __getitem__(self, item):
        Fij = copy.copy(self)
        Fij.y = self.y[item]
        Fij.c = self.c[item]
        return Fij

    def __setitem__(self, key, value):
        if isinstance(value,(int,float)):
            raise TypeError('new value must be a numpy array')
        vsh = value.shape
        if vsh[-1] == self.N:
            self.y[key] = value
        else:
            raise ValueError('last index of value must have {} values (one for each node)'.format(self.N))


# todo: need to fix the setter methods: they do not adjust the other value when set with a subset