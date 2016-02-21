from demos.setup import np, plt, demo
from compecon import BasisChebyshev, NLP
from compecon.tools import nodeunif

__author__ = 'Randall'

# DEMAPP10 Monopolist's Effective Supply Function

# Residual Function
def resid(c):
    Q.c = c
    q = Q(p)
    return p + q / (-3.5 * p **(-4.5)) - np.sqrt(q) - q ** 2

# Approximation structure
n, a, b = 21, 0.5, 2.5
Q = BasisChebyshev(n, a, b)
c0 = np.zeros(n)
c0[0] = 2
p = Q.nodes

# Solve for effective supply function
monopoly = NLP(resid)
Q.c = monopoly.broyden(c0)

# Setup plot
nplot = 1000
p = nodeunif(nplot, a, b)
rplot = resid(Q.c)

# Plot effective supply
demo.figure("Monopolist's Effective Supply Curve", 'Quantity', 'Price')
plt.plot(Q(p), p)


# Plot residual
demo.figure('Functional Equation Residual', 'Price', 'Residual')
plt.hlines(0, a, b, 'k', '--')
plt.plot(p, rplot)

plt.show()
