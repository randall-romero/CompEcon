from demos.setup import np, plt
from compecon import BasisChebyshev, NLP
from compecon.tools import nodeunif

__author__ = 'Randall'

# DEMAPP10 Monopolist's Effective Supply Function

# Residual Function
def resid(c, Q, p):
    Q.c = c
    q = Q(p)
    return p + q / (-3.5 * p **(-4.5)) - np.sqrt(q) - q ** 2

# Approximation structure
n, a, b = 21, 0.5, 2.5
Q = BasisChebyshev(n, a, b)
c0 = np.zeros(n)
c0[0] = 2
pnode = Q.nodes

# Solve for effective supply function
monopoly = NLP(resid, c0, Q, pnode)
Q.c = monopoly.broyden()

# Setup plot
nplot = 1000
pplot = nodeunif(nplot, a, b)
rplot = resid(Q.c, Q, pplot)

# Plot effective supply
plt.figure()
plt.plot(Q(pplot), pplot)
plt.title("Monopolist's Effective Supply Curve")
plt.xlabel('Quantity')
plt.ylabel('Price')


# Plot residual
plt.figure()
plt.plot(pplot, rplot)
plt.plot(pplot, np.zeros_like(pplot), 'k--')
plt.title('Functional Equation Residual')
plt.xlabel('Price')
plt.ylabel('Residual')

plt.show()
