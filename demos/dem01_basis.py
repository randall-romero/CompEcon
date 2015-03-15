__author__ = 'Randall'

"""
    DEMO: basisChebyshev
    This script provides a few examples on using the basis classes.
    Last updated: March 15, 2015.
    Copyright (C) 2014 Randall Romero-Aguilar
    Licensed under the MIT license, see LICENSE.txt
"""

import basisChebyshev as bc
import numpy as np
import matplotlib.pyplot as plt

"""
    EXAMPLE 1:
        Using BasisChebyshev to interpolate a 1-D function with a Chebyshev basis
        PROBLEM: Interpolate the function y = f(x) = 1 + sin(2*x) on the domain [0,pi], using 5 Gaussian nodes.
"""

# First, create the function f
def f(x):
    return np.mat(1 + np.sin(2 * x)).T

# and the Chebyshev basis B. If the type of nodes is unspecified, Gaussian is computed by default
B = bc.BasisChebyshev(5,0,np.pi)

# Interpolation matrix and nodes: Obtain the interpolation matrix Phi, evaluated at the basis nodes.
Phi = B.interpolation()

# The basis nodes are:
xnodes = B.nodes

# Fitting a function: set the interpolation coefficients c:
c = np.mat(np.linalg.solve(Phi[0], f(xnodes)))  # notice that Phi is a dictionary, the matrix is indexed by 0

# Next plot the function f and its approximation. To evaluate the function defined by the basis B and coefficients c
# at values xx we use the interpolation method:

# We also plot the residuals, showing the residuals at the interpolating nodes (zero by construction)
xx = np.linspace(B.a, B.b, 121)
Phi_x = B.interpolation(xx)[0]
f_approx = Phi_x * c
fxx = f(xx)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xx.T, fxx)
plt.plot(xx.T, f_approx)
plt.legend(['f = 1 + sin(2x)','approx.'])
plt.title('Chebyshev approximation with 5 nodes')

plt.subplot(2, 1, 2)
plt.plot(xx.T, f_approx - fxx)
plt.title('Residuals using 5 nodes')

# Adjusting the number of nodes: to increase accuracy, we increase the number of nodes in B to 25.
B.n = 25
c2 = np.mat(np.linalg.solve(B.interpolation(B.nodes)[0], f(B.nodes)))
plt.figure()
plt.subplot(2,1,1)
plt.plot(xx.T, f(xx))
plt.plot(xx.T, B.interpolation(xx)[0] * c2)
plt.legend(['f = 1 + sin(2x)','approx.'])
plt.title('Chebyshev approximation with 25 nodes')

plt.subplot(2,1,2)
plt.plot(xx.T, B.interpolation(xx)[0] * c2 - f(xx),'r')
plt.title('Residuals using 25 nodes')
