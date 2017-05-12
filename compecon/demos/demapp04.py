from demos.setup import np, plt
from numpy.linalg import norm, cond
from compecon import BasisChebyshev
import warnings

warnings.simplefilter('ignore')


""" Uniform-node and Chebyshev-node polynomial approximation of Runge's function
and compute condition numbers of associated interpolation matrices
"""


# Runge function
runge = lambda x: 1 / (1 + 25 * x ** 2)

# Set points of approximation interval
a, b = -1, 1

# Construct plotting grid
nplot = 1001
x = np.linspace(a, b, nplot)
y = runge(x)

# Plot Runge's Function
fig1 = plt.figure(figsize=[6, 9])
ax1 = fig1.add_subplot(211, title="Runge's Function", xlabel='', ylabel='y', xticks=[])
ax1.plot(x, y)
ax1.text(-0.8, 0.8, r'$y = \frac{1}{1+25x^2}$', fontsize=18)

# Initialize data matrices
n = np.arange(3, 33, 2)
nn = n.size
errunif, errcheb = (np.zeros([nn, nplot]) for k in range(2))
nrmunif, nrmcheb, conunif, concheb = (np.zeros(nn) for k in range(4))

# Compute approximation errors on refined grid and interpolation matrix condition numbers
for i in range(nn):
    # Uniform-node monomial-basis approximant
    xnodes = np.linspace(a, b, n[i])
    c = np.polyfit(xnodes, runge(xnodes), n[i])
    yfit = np.polyval(c, x)
    phi = xnodes.reshape(-1, 1) ** np.arange(n[i])

    errunif[i] = yfit - y
    nrmunif[i] = np.log10(norm(yfit - y, np.inf))
    conunif[i] = np.log10(cond(phi, 2))

    # Chebychev-node Chebychev-basis approximant
    yapprox = BasisChebyshev(n[i], a, b, f=runge)
    yfit = yapprox(x)  # [0] no longer needed?  # index zero is to eliminate one dimension
    phi = yapprox.Phi()
    errcheb[i] = yfit - y
    nrmcheb[i] = np.log10(norm(yfit - y, np.inf))
    concheb[i] = np.log10(cond(phi, 2))

# Plot Chebychev- and uniform node polynomial approximation errors
ax2 = fig1.add_subplot(212, title="Runge's Function $11^{th}$-Degree\nPolynomial Approximation Error.",
                       xlabel='x',ylabel='Error')
ax2.axhline(color='gray', linestyle='--')
ax2.plot(x, errcheb[4], label='Chebychev Nodes')
ax2.plot(x, errunif[4], label='Uniform Nodes')
ax2.legend(loc='upper center', frameon=False)


# Plot approximation error per degree of approximation
fig2 = plt.figure(figsize=[6, 9])
ax3 = fig2.add_subplot(211, title="Log10 Polynomial Approximation Error for Runge''s Function",
                       xticks=[], xlabel='',ylabel='Log10 Error')
ax3.plot(n, nrmcheb, label='Chebychev Nodes')
ax3.plot(n, nrmunif, label='Uniform Nodes')
ax3.legend(loc='upper left', frameon=False)

ax4 = fig2.add_subplot(212, title="Log10 Interpolation Matrix Condition Number",
                       xlabel='Degree of Approximating Polynomial',ylabel='Log10 Condition Number')
ax4.plot(n, concheb, label='Chebychev Polynomial Basis')
ax4.plot(n, conunif, label='Mononomial Basis')
ax4.legend(loc='upper left', frameon=False)
plt.show()
