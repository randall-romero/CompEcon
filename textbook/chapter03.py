__author__ = 'Randall'

import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm as Normal_distribution
from compecon import NLP, MCP, LCP
from compecon.tools import example, exercise

''' Example page 32 '''
example(32)
f = NLP(lambda x: x**3 - 2)
x = f.bisect(1, 2)
print('x = ', x)

''' Example page 33 '''
example(33)
g = NLP(lambda x: x ** 0.5)
x = g.fixpoint(0.4)
print('x = ', x)

''' Example page 35 '''
example(35)
def cournot(q):
    c = np.array([0.6, 0.8])
    eta = 1.6
    e = -1 / eta
    s = q.sum()
    fval = s ** e + e * s ** (e-1)*q - c * q
    fjac = e*s**(e-1) * np.ones([2,2]) + e * s ** (e-1) * np.identity(2) +\
           (e-1)*e*s **(e-2)* np.outer(q, [1,1]) - np.diag(c)
    return fval, fjac

market = NLP(cournot)
x = market.newton([0.2, 0.2])
print('q1 = ', x[0], '\nq2 = ', x[1])

''' Example page 39 '''
# continuation of example page 35
example(39)
x = market.broyden([0.2, 0.2])
print('q1 = ', x[0], '\nq2 = ', x[1])


''' Example page 43 '''
# numbers don't match those of Table 3.1, but CompEcon2014' broyden gives same answer as this code
example(43)
opts = {'maxit': 30, 'all_x': True, 'print': True}

g = NLP(lambda x: sqrt(x), 0.5)
f = NLP(lambda x: (x - sqrt(x), 1 - 0.5/sqrt(x)), 0.5)

x_fp = g.fixpoint(**opts)
x_br = f.broyden(**opts)
x_nw = f.newton(**opts)


''' Example page 51 '''
example(51)
f = MCP(lambda x: (1.01 - (1 - x) ** 2, 2 * (1 - x)), 0, np.inf)
x = f.zero(0.0)  # ssmooth transform by default
print('Using', f.opts.transform, 'transformation,  x = ', x)
x = f.zero(0.0, transform='minmax')
print('Using', f.opts.transform, 'transformation, x = ', x, 'FAILED TO CONVERGE')


# ==============================================================
''' Exercise 3.1 '''
exercise('3.1')

def newtroot(c):
    x = c / 2
    for it in range(150):
        fx = x ** 2 - c
        if abs(fx) < 1.e-9:
            break
        else:
            x -= fx / (2 * x)

    return x

# testing with some known values
for c in [1.0, 25.0, 49.0, 30.25, 100]:
    print('newtroot({:g}) = {:g}'.format(c, newtroot(c)))

''' Exercise 3.2 '''
exercise('3.2')
def newtroot2(c):
    x = abs(c)
    for it in range(150):
        step = (x + 1 - ((1 + c) / (1 + x)) * (1 + c) + 2 * (c / (1 + x))) / 2
        if abs(step) < 1.e-9:
            break
        else:
            x -= step

    return x

# testing with some extreme values
for c in [0.0, 1.0, 1.e-12, 1.e250]:
    print('newtroot2({:g}) = {:g}'.format(c, newtroot2(c)))

''' Exercise 3.3 '''
exercise('3.3')
Phi = Normal_distribution.cdf


def BSVal(S, K, tau, r, delta, sigma):
    edtS = exp(-delta * tau) * S
    ertK = exp(-r* tau) * K
    sigmat = sigma * sqrt(tau)

    d = (log(edtS) - log(ertK)) / sigmat + sigmat / 2
    value = edtS * Phi(d) - ertK * Phi(d - sigmat)
    value_sigma = edtS * sqrt(tau / (2 * np.pi)) * exp(-0.5 * d ** 2)
    return value, value_sigma

def ImpVol(S, K, tau, r, delta, V):
    sigma = 1.0
    for it in range(150):
        value, dvalue = BSVal(S, K, tau, r, delta, sigma)
        f = V - value
        if abs(f) < 1.e-9:
            break
        else:
            step = (V - value) / dvalue
            sigma += step

    return sigma

# Testing the formula with parameters from demfin02
S, K, tau, r, delta = 1.12, 1.0, 1, 0.05, 0.03
sigma = 0.2

market_value = BSVal(S, K, tau, r, delta, sigma)[0]
implied_sigma = ImpVol(S, K, tau, r, delta, market_value)
print('original sigma = ', sigma)
print('implied sigma = ', implied_sigma)
print('market value = ', market_value)

''' Exercise 3.4 '''
# It's an analytical exercise, no code required

''' Exercise 3.5 '''
exercise('3.5')


def f35(z):
    x, y = z
    fval = [200 * x * (y - x ** 2) - x + 1, 100 * (x ** 2 - y)]
    fjac = [[200 * (y - 3*x**2) - 1, 200*x],
            [200 * x, -100]]
    return np.array(fval), np.array(fjac)

f35problem = NLP(f35, [0.4, 1.4])
f35problem.check_jacobian()
print('Solution by Newton:', f35problem.newton())
print('Solution by Broyden:', f35problem.broyden())

''' Exercise 3.6 '''
exercise('3.6')

def icdf(p, F, x0, *args):
    x = x0
    for it in range(150):
        cdf, pdf = F(x, *args)
        if abs(cdf - p) < 1.e-9:
            break
        else:
            x += (p - cdf) / pdf

    return x

# testing with a normal distribution
phi = Normal_distribution.pdf

def cdfnormal(x, mu, sigma):
    z = (x-mu) / sigma
    return Phi(z), phi(z)

for p in [0.01, 0.05, 0.5, 0.95, 0.99]:
    print('icdf({:g}) = {:g}'.format(p, icdf(p, cdfnormal,0.0,0.0,1.0)))


''' Exercise 3.7 '''
exercise('3.7')
a = np.array([[2.0, 1.5], [1.5, 2.0], [1.5, 2.0]])
e = np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 0.0]])
v = np.array([[-2.0, -0.5],[-1.5, -0.5],[-0.5, -1.5]])

def unpack_market(z):
    x, p1, la =  np.split(z, [6, 7])
    x.shape = 3, 2
    p = np.zeros(2)
    p[0], p[1] = p1, 1-p1
    return x, p, la


def equilibrium(z):
    x, p, la =  unpack_market(z)
    foc = a * x ** v - np.outer(la, p)
    budget =  (p * (x - e)).sum(1)
    clearance = (e - x).sum(0)
    fval = np.zeros(10)
    fval[:6] = foc.flatten()
    fval[6:9] = budget
    fval[9] = clearance[0]
    return fval

guess = np.zeros(10)
guess[:6] = e.flatten() + 0.25
guess[6] = 0.5
guess[7:] = 0.0

market = NLP(equilibrium, guess)
z = market.broyden(print=True)
x, p, la = unpack_market(z)
print('Consumption = \n', x)
print('Prices =\n', p)
print('Multipliers=\n', la)
print('Total consumption = ', x.sum(0), ' = total endowment = ', e.sum(0))


''' Exercise 3.8 '''
exercise('3.8')

kappa = 0.2
delta = 0.95

def arbitrage(x, s):
    c1, c2, p1, p2 = x
    f = np.zeros(4)
    f[0] = s - c1 - c2
    f[1] = delta * p2 - p1 -kappa
    f[2] = p1 - c1 ** -5
    f[3] = p2 - c2 ** -5
    return f

guess = np.ones(4) / 2
txt = 'When s = {:d}, equilibrium is c1 = {:.2f}, c2 = {:.2f}, p1 = {:.2f}, p2 = {:.2f}'

for s in [1, 2, 3]:
    x = NLP(arbitrage,guess, s).zero()
    print(txt.format(s, *x))

''' Exercise 3.10 '''
exercise('3.10')

# SOMETHING IS WRONG WITH THIS ONE!

# growth_model = np.zeros((12, 12))
# q = np.zeros(12)
#
# ad = np.array([42, 54, 51])
# bd = np.array([-2, -3, -1])
# as_ = np.array([9, 3, 18])
# bs = np.array([1, 2, 1])
#
#
# growth_model = [
#     [-2,  0,  0, -2,  0,  0, -2,  0,  0, -1,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0],
#     [ 0, -3,  0,  0, -3,  0,  0, -3,  0,  0, -1,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0, -1],
#     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1],
#     [ 0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0, -1],
#     [-1, -1, -1,  0,  0,  0,  0,  0,  0,  1,  0,  0],
#     [ 0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  1,  0],
#     [ 0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0,  1]]
#
# q = [-42, 3, 9, 3, -54, 3, 6, 3, -51, 9, 3, 18]
#
# # bounds
# a = np.zeros(12)
# b = a + np.inf
# x0 = np.random.rand(12)
#
# trade = LCP(growth_model, q, a, b, x0)
# z = trade.zero()
# x, p = np.split(z, [9])
# x.shape = 3, 3
#
#
#
# print('Consumption = \n', x)
# print('Prices =\n', p)
# print('Demand = ', x.sum(1))
# print('Supply = ', x.sum(0))