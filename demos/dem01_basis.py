__author__ = 'Randall'

"""
    DEMO: basisChebyshev
    This script provides a few examples on using the basis classes.
    Last updated: March 15, 2015.
    Copyright (C) 2014 Randall Romero-Aguilar
    Licensed under the MIT license, see LICENSE.txt
"""

from compecon import BasisChebyshev
import numpy as np
import matplotlib.pyplot as plt

"""
    EXAMPLE 1:
        Using BasisChebyshev to interpolate a 1-D function with a Chebyshev basis
        PROBLEM: Interpolate the function y = f(x) = 1 + sin(2*x) on the domain [0,pi], using 5 Gaussian nodes.
"""

# First, create the function f
f = lambda x: np.mat(1 + np.sin(2 * x)).T

# and the Chebyshev basis B. If the type of nodes is unspecified, Gaussian is computed by default
Phi = BasisChebyshev(5,0,np.pi)

# Interpolation matrix and nodes: Obtain the interpolation matrix Phi, evaluated at the basis nodes.


# The basis nodes are:
xnodes = Phi.nodes

# Fitting a function: set the interpolation coefficients c:
c = np.mat(np.linalg.solve(Phi(), f(xnodes)))  # notice that Phi is a dictionary, the matrix is indexed by 0

# Next plot the function f and its approximation. To evaluate the function defined by the basis B and coefficients c
# at values xx we use the interpolation method:

# We also plot the residuals, showing the residuals at the interpolating nodes (zero by construction)
xx = np.linspace(Phi.a, Phi.b, 121)
Phi_x = Phi(xx)
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
Phi.n = 25
c2 = np.mat(np.linalg.solve(Phi(), f(Phi.nodes)))
plt.figure()
plt.subplot(2,1,1)
plt.plot(xx.T, f(xx))
plt.plot(xx.T, Phi(xx) * c2)
plt.legend(['f = 1 + sin(2x)','approx.'])
plt.title('Chebyshev approximation with 25 nodes')

plt.subplot(2,1,2)
plt.plot(xx.T, Phi(xx) * c2 - f(xx),'r')
plt.title('Residuals using 25 nodes')

# With previous basis, now compute derivative.
df = lambda x: np.mat(2 * np.cos(2 * x)).T
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(xx.T, df(xx))
plt.plot(xx.T, Phi(xx, 1) * c2)  # notice the 1 in Phi(xx,1) to compute first derivative
plt.legend(['df/dx = 2cos(2x)', 'approx.'])
plt.title('Chebyshev approximation with 25 nodes')

plt.subplot(2, 1, 2)
plt.plot(xx.T, Phi(xx, 1) * c2 - df(xx), 'r')
plt.title('Residuals using 25 nodes')

plt.show()