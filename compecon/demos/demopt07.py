# DEMOPT07 Bound constrained optimization via sequential LCP

import numpy as np
from compecon import LCP
__author__ = 'Randall'


def func(x):
    y = np.exp(-x[0]) + x[0] * x[1] ** 2
    d = np.array([-np.exp(-x[0]) + x[1] ** 2, 2 * x[0] * x[1]])
    s = np.array([[np.exp(x[0]), 2 * x[1]],
                  [2*x[1], 2*x[0]]])
    return y, d, s

# Generate problem test data
z = np.random.randn(2, 2) - 1
a = np.min(z, 1)
b = np.max(z, 1)
x = np.random.rand(2)

# Set convergence parameters
maxit = 200
eps = 1e-10

# Perform sequential LCP
print('{:>5s} {:>10s}'.format('it','change'))

for it in range(maxit):
    xold = x
    f, d, s = func(x)
    pro = LCP(s, d - s.dot(xold), a, b, xold)
    x = pro.zero()
    change = np.linalg.norm(x - xold, np.inf)
    print('{:5d} {:10.2e}'.format(it,change))
    if change < eps:
         break

# Print results
if it >= maxit:
    print('Sequential lcp failed in demomax')
else:
    print('\nPerform Bounded Maximization with Random Data\n')
    print('       a         x         b        f''')
    for i in range(2):
        print('{:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}'.format(a[i], x[i], b[i], d[i]))




