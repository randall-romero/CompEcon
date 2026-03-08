__author__ = 'Randall'

import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import gamma as Gamma_Distribution
from scipy.special import psi
from scipy.special import gamma as Gamma_Function

from compecon import OP, NLP, MCP, MLE
from compecon.tools import example, exercise, gridmake, jacobian
np.set_printoptions(4, linewidth=120)
import matplotlib.pyplot as plt

import warnings
# warnings.simplefilter('ignore')

"""
''' Example page 61 '''
example(61)
problem = OP(lambda x: x * np.cos(x ** 2))
xstar = problem.golden(0, 3)
print('x = ', xstar)

#==============================================================
''' Exercise 4.1 '''
exercise('4.1')

# simulate some data
n, k = 500, 3
beta = np.arange(k) + 0.5

X = np.random.rand(n, k)
mu = X.dot(beta)
p = np.random.rand(n)
y = - mu * np.log(1 - p)
# plt.figure()
# plt.hist(y,n/20)
# plt.show()

def logL(beta, X, y):
    u = X.dot(beta)
    l = - (y/u) - log(u)
    return l.sum()


L = OP(logL, np.ones(k),X, y)
beta_hat = L.qnewton()
print('Looking for the maximum likelihood:    beta = ', beta_hat)


def dlogL(beta, X, y):
    u = X.dot(beta)
    temp = ((y - u) / u ** 2)
    dl = temp[:, np.newaxis] * X
    return dl.sum(0)

D = NLP(dlogL, np.ones(k), X, y)
beta_tilde = D.zero()
print('Solving FOC of the maximum likelihood: beta = ', beta_tilde)
print('True population values:                beta =  [ {:6.4f}  {:6.4f}  {:6.4f}]'.format(*beta))

print('The estimator for the variance covariance matrix is\n', np.linalg.inv(-L.hessian(L.x)))

# Solve using MLE
mle = MLE(logL, np.ones(3), X, y)
mle.estimate()
print('\nUsing the MLE class\n\tbeta = ', mle.beta)
print('\tConfidence intervals\n', mle.ci())

''' Exercise 4.2 '''
exercise('4.2')

# simulate some data
n = 500
a = 5.0
b = 2.0
x_data = Gamma_Distribution.rvs(a, scale=1/b, size=n)
Y1 = x_data.mean()
Y2 = exp(log(x_data).mean())

b_hat = lambda a0: a0 / Y1

def dlogL(theta):
    return log(theta) - log(Y1 / Y2) - psi(theta)


a0 = 1.1 # initial guess
estimator = NLP(dlogL, a0, print=True, all_x=True)
# estimator = MCP(dlogL, 0, np.inf, a0, print=True, all_x=True)
a_hat = estimator.zero()
print(estimator.x_sequence)
print(b_hat(estimator.x_sequence))


y1y2 = np.linspace(1.1, 3, 48)
dlogL2 = lambda theta, y12: log(theta) - log(y12) - psi(theta)
ttheta = np.array([NLP(dlogL2, a0, k).zero() for k in y1y2])
plt.figure()
plt.plot(y1y2, ttheta)
plt.xlabel('Y1 / Y2')
plt.ylabel('theta1')
plt.show()



# Solve it using the MLE object
def logL(theta, x):
    n = x.size
    a, b = theta
    return n*a*log(b) + (a-1)*log(x).sum() - b*x.sum() - n*log(Gamma_Function(a))

mle = MLE(logL, np.ones(2), x_data)
mle.estimate()
print('theta1 = {:.4f}, theta1 = {:.4f}'.format(*mle.beta))
print('Estimated Covariance = \n', mle.Sigma)
print('Confidence intervals\n', mle.ci())

"""
''' Exercise 4.3 '''
exercise('4.3')

treasury_tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 30])

treasury_r = np.array(
    [[4.44, 4.49, 4.51, 4.63, 4.63, 4.62, 4.82, 4.77, 5.23],
     [4.45, 4.48, 4.49, 4.61, 4.61, 4.60, 4.84, 4.74, 5.16],
     [4.37, 4.49, 4.53, 4.66, 4.66, 4.65, 4.86, 4.76, 5.18],
     [4.47, 4.47, 4.51, 4.57, 4.57, 4.57, 4.74, 4.68, 5.14]])


def Z(r, t, k, a, s):
    gamma = sqrt(k **2 + 2 * s ** 2)
    egt = exp(gamma * t) - 1

    numA = 2 * gamma * exp((gamma + k) * t / 2)
    numB = 2*egt
    den = (gamma + k) * egt + 2 * gamma
    expA = 2 * k * a / (s ** 2)
    A = (numA / den) ** expA
    B = numB / den
    Z = A * exp(-B * r)
    return Z

def ss(x, r, tau):
    k, a, s = x
    resid = r + 100 * log(Z(r / 100, tau, k, a, s)) / tau
    return -(resid ** 2).sum()


def ss2(x, r, tau):
    tmp = lambda x: ss(x, r, tau)
    return jacobian(tmp, x)[0]





x0 = np.array([0.51, 0.05, 0.12])

hola = MCP(ss2, np.zeros(3), np.ones(3), x0, treasury_r[0], treasury_tau)
x = hola.zero(print=True)
print(x)

objective = OP(ss, x0, treasury_r[0], treasury_tau)
objective.qnewton(print=True, all_x=True)
print(objective.x)
print(objective.fnorm)

