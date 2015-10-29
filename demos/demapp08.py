from demos.setup import np, plt, demo
from compecon import BasisChebyshev, NLP
from compecon.tools import nodeunif

__author__ = 'Randall'

# DEMAPP08 Compute function inverse via collocation

# Residual function


def resid(c, F, x):
    F.c = c
    f = F(x)
    return f ** -2 + f ** -5 - 2 * x

# Approximation structure
n, a, b = 31, 1, 5
f = BasisChebyshev(n, a, b)  # define basis functions
xnode = f.nodes              # compute standard nodes

# Compute function inverse
c0 = np.zeros(n)
c0[0] = 0.2
problem = NLP(resid, c0, f, xnode)
f.c = problem.broyden()   # call rootfinding routine to compute coefficients

# Plot setup
n = 1000
x = nodeunif(n, a, b)
r = resid(f.c, f, x)

# Plot function inverse
demo.figure('Implicit Function', 'x', 'f(x)')
plt.plot(x, f(x))

# Plot residual
demo.figure('Functional Equation Residual', 'x', 'Residual')
plt.plot(x, np.zeros_like(x), 'k--')
plt.plot(x, r)

plt.show()

