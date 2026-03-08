from demos.setup import np, plt
from compecon import BasisChebyshev, BasisSpline, NLP
from compecon.quad import qnwsimp
import warnings
warnings.simplefilter('ignore')

__author__ = 'Randall'
'''

# Table 6.1
runge = lambda x: 1 / (1 + 25 * x ** 2)

a, b = -5, 5

xx = np.linspace(a, b, 1001)
fxx = runge(xx)


def precision(fhat):
    return np.log10(np.linalg.norm(fhat - fxx, np.inf))

print('log10( norm(f(x) - fapprox(x0))) when using UNIFORM nodes')
print('\tn   monomial  chebyshev')

for n in [10, 20, 30, 40, 50]:
    # Chebyshev polynomials, uniform nodes
    Bunif = BasisChebyshev(n, a, b, f=runge, nodetype='uniform')

    # Uniform-node monomial-basis approximant
    xnodes = Bunif.nodes[0]
    c = np.polyfit(xnodes, runge(xnodes), n)
    yfit = np.polyval(c, xx)
    print('\t{:d}    {:.2f}    {:.2f}'.format(n, precision(yfit), precision(Bunif(xx))))


# Figure 6.1
nodes = BasisChebyshev(9, 0, 1).nodes
plt.figure()
plt.plot(nodes, np.zeros_like(nodes), 'r*')


# Figure 6.2
f = lambda x: np.exp(-x)
xx = np.linspace(a, b, 1001)
yy = f(xx)

B = BasisChebyshev(10, -5, 5, f=f)
res_cheb = B(xx) - yy

xnodes = np.linspace(-5, 5, 10)
c = np.polyfit(xnodes, f(xnodes), 10)
res_unif = np.polyval(c, xx) - yy
plt.figure()
plt.plot(xx, res_cheb, xx, res_unif)
plt.legend(['Chebyshev nodes', 'Uniform nodes'])
# todo: i get smaller errors!?


# Table 6.2
# Errors for seletected interpolation methods
#   1: y = exp(-x)
#   2: y = 1./(1+25*x.^2).
#   3: y = sqrt(abs(x))

print('\n\n')

## Functions to be approximated
funcs = [lambda x: np.exp(-x),
         lambda x: 1 / ( 1 + 25 * x ** 2),
         lambda x: np.sqrt(np.abs(x))]

# Set degree of approximation and endpoints of approximation interval
a, b = -5, 5
x = np.linspace(a, b, 2001)  # to evaluate precision

precision = lambda y, yhat: np.max(np.abs(yhat - y))

func_names = ['exp(-x)', '1/(1+25x^2)', 'sqrt(abs(x))']


print('\t\t{:s}\t{:10s}\t{:10s}\t{:10s}'.format('n', 'Linear', 'Cubic', 'Chebyshev'))
for ii, ff in enumerate(funcs):
    fx = ff(x)
    print('\n', func_names[ii])
    for n in [10, 20, 30]:
        C = precision(BasisChebyshev(n, a, b, f=ff)(x), fx)
        S = precision(BasisSpline(n, a, b, f=ff)(x), fx)
        L = precision(BasisSpline(n, a, b, k=1, f=ff)(x), fx)

        print('\t\t{:d}\t{:.2e}\t{:.2e}\t{:.2e}'.format(n, L, S, C))


# Example page 139-140
alpha = 2
f = lambda x: np.exp(-alpha * x)

F = BasisChebyshev(10, -1, 1, f=f)
x = np.linspace(-1, 1, 1001)
plt.figure()
plt.plot(x, F(x) - f(x))
plt.title('Figure 6.11  Approximation Error')
plt.show()

'''
# Example p147
r, k, eta, s0 = 0.1, 0.5, 5 ,1


T, n = 1, 15
tnodes = BasisChebyshev(n - 1, 0, T).nodes
F = BasisChebyshev(n, 0, T, y=np.ones((2, n)))


def resid(c, tnodes, T, n, F, r, k, eta, s0):
    F.c = np.reshape(c[:], (2, n))
    (p, s), d = F(tnodes, [[0, 1]])
    d[0] -= (r * p + k)
    d[1] += p ** -eta
    (p_0, p_T), (s_0, s_T) = F([0, T])
    return np.r_[d.flatten(), s_0 - s0, s_T]


storage = NLP(resid, F.c.flatten(), tnodes, T, n, F, r, k, eta, s0)
c = storage.broyden(print=True)
F.c = np.reshape(c, (2, n))

nplot = 501
t = np.linspace(0, T, nplot)
(p, s), (dp, ds) = F(t, [[0, 1]])
res_p = dp - r * p - k
res_s = ds + p ** -eta
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, res_p)
plt.title('Residuals')
plt.ylabel('d(price) residual')

plt.subplot(2, 1, 2)
plt.plot(t, res_s)
plt.xlabel('time')
plt.ylabel('d(storage) residual')


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, p)
plt.title('Solution')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
plt.plot(t, s)
plt.xlabel('time')
plt.ylabel('Stock')


plt.show()