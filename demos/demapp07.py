from demos.setup import np, plt
from compecon import BasisChebyshev, NLP
from compecon.tools import nodeunif

__author__ = 'Randall'


# DEMAPP07 Solve Cournot oligopoly model via collocation


# Residual function


def resid(c, Q, p, alpha, eta):
    Q.c = c
    q = Q(p)
    return p + q * ((-1 / eta) * p ** (eta+1)) - alpha * np.sqrt(q) - q ** 2


# Approximation structure
n =  21
a = 0.5
b = 2.0
c0 = np.zeros(n)
c0[0] = 1

Q = BasisChebyshev(n, a, b, c=c0)



# Model parameters
alpha = 1.0
eta = 3.5


# Solve for effective supply function
p = Q.nodes
cournot = NLP(resid, c0, Q, p, alpha, eta, tol=1e-12)
Q.c = cournot.broyden()

# Plot demand and effective supply for m=5 firms
plt.figure()
pplot = nodeunif(501, a, b)
splot = Q(pplot)
dplot = pplot ** -eta
plt.plot(5 * splot, pplot, dplot, pplot)
plt.xlim([0, 4])
plt.ylim([0.5, 2])
plt.legend(('Supply','Demand'))
plt.title('Cournot Effective Firm Supply Function')
plt.xlabel('Quantity')
plt.ylabel('Price')



# Plot residual
plt.figure()
rplot = resid(Q.c, Q, pplot, alpha, eta)
plt.plot(pplot, rplot)
plt.plot(pplot, np.zeros_like(pplot), 'k--', lw=2)
plt.title('Residual Function for Cournot Problem')
plt.xlabel('Quantity')
plt.ylabel('Residual')


# Plot demand and effective supply for m=1,3,5,10,15,20 firms
plt.figure()
m = np.array([1, 3, 5, 10, 15, 20])
plt.plot(np.outer(splot, m), pplot, dplot, pplot)
plt.title('Industry Supply and Demand Functions')
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.legend(['m=1','m=3','m=5','m=10','m=15','m=20'])
plt.xlim([0, 13])


# Plot equilibrium price as a function of number of firms m
plt.figure()
pp = (b + a) / 2
dp = (b - a) / 2
m  = np.arange(1, 26)
for i in range(50):
    dp /= 2
    pp = pp - np.sign(Q(pp) * m - pp ** (-eta)) * dp

plt.plot(m, pp)
plt.title('Cournot Equilibrium Price as Function of Industry Size')
plt.xlabel('Number of Firms')
plt.ylabel('Price')

plt.show()
