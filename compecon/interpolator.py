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
        else:
            B = Basis()
            Basis.__init__(B, *args, **kwargs)

        # Compute interpolation matrix at nodes
        _PhiT = B.interpolation().T
        B._PhiT = _PhiT

        # Compute inverse if not spline
        if B.type == 'chebyshev':
            B._PhiInvT = linalg.solve(np.dot(_PhiT, _PhiT.T), _PhiT).T
        else:
            raise NotImplementedError


        # share data in this basis with all instances
        self.__dict__ = copy.copy(B.__dict__)



        # add data
        if y is None:
            y = np.zeros([self.N])
        elif isinstance(y, (list, np.ndarray)):
            y = np.asarray(y)
            y = y.reshape([y.size])
        else:
            raise ValueError('y must be a list or numpy array with {} elements'.format(self.N))

        if y.size != self.N:
            raise ValueError('y must be a list or numpy array with {} elements'.format(self.N))

        self._y = y
        self._c = np.dot(self._y, self._PhiInvT)
        self._yIsOutdated = False
        self._cIsOutdated = False



    """ setter and getter methods """

    @property
    def y(self):
        """ :return: function values at nodes """
        if self._yIsOutdated:
            self._y = np.dot(self._c, self._PhiT)
            self._yIsOutdated = False
        return self._y

    @property
    def c(self):
        """ :return: interpolation coefficients """
        if self._cIsOutdated:
            self._c = np.dot(self._y, self._PhiInvT)
            self._cIsOutdated = False
        return self._c

    @property
    def x(self):
        """  :return: interpolation nodes  """
        return self.nodes


    @y.setter
    def y(self, val):
        if isinstance(val, (list, np.ndarray)):
            val = np.asarray(val)
            val = val.reshape([val.size])
        else:
            raise ValueError('y must be a list or numpy array with {} elements'.format(self.N))

        if val.size != self.N:
            raise ValueError('val must be a list or numpy array with {} elements'.format(self.N))
        self._y = val
        self._yIsOutdated = False
        self._cIsOutdated = True

    @c.setter
    def c(self, val):
        assert (val.size == self.M)  # one value per polynomial
        self._c = val
        self._cIsOutdated = False
        self._yIsOutdated = True



    """  Interpolation method """

    def __call__(self,x=None, order=None):
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

"""
        NO LONGER NEEDED: INTERPOLATOR ONLY HANDLES ONE FUNCTION AT A TIME:
        For more dimension, just make a numpy.array from a list comprehension

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

"""


def Interpolator_array(basis, dims):
    """
    Creates an array of Interpolator objects

    :param basis: a Basis instance common to all functions in the array
    :param dims: the shape of the array
    :return: a numpy array of Interpolator instances
    """

    A = np.array([Interpolator(basis) for k in range(np.prod(dims))])
    return A.reshape(dims)


class Hola(Interpolator):
    def __init__(self, basis, dims):
        A = np.array([Interpolator(basis) for k in range(np.prod(dims))])
        self.F = A.reshape(dims)
        self.shape = self.F.shape
        self.size = self.F.size
        self.ndim = self.F.ndim
        self.idx = [np.unravel_index(k, self.shape) for k in range(self.size)]
        Shape = list(self.shape)
        Shape.append(self.N)
        self.Shape = Shape


    def __getitem__(self, item):
        return self.F[item]

    def __setitem__(self, key, value):
        if isinstance(value, (list, np.ndarray)):
            value = np.asarray(value)
            value = value.reshape([value.size])
        else:
            raise ValueError('y must be a list or numpy array with {} elements'.format(self.N))

        if value.size != self.N:
            raise ValueError('val must be a list or numpy array with {} elements'.format(self.N))
        self.F[key].y = value

    @property
    def N(self):
        """  :return: number of nodes """
        return self.F[self.idx[0]].N

    @property
    def y(self):
        """ :return: function values at nodes """
        y = np.array([self.F[k].y for k in self.idx])
        return y.reshape(self.Shape)

    @property
    def c(self):
        """ :return: interpolation coefficients """
        c = np.array([self.F[k].c for k in self.idx])
        return c.reshape(self.Shape)

    @property
    def x(self):
        """  :return: interpolation nodes  """
        return self.F[self.idx[0]].x
