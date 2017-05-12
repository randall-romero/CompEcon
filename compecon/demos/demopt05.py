__author__ = 'Randall'

from compecon import OP
import numpy as np

# DEMOPT05 Optimization with qnewton

f = OP(lambda x: x ** 3 - 12 * x ** 2 + 36 * x + 8)

x = f.qnewton(x0=4.0)
J = f.jacobian(x)
E = f.hessian(x)
E = np.linalg.eig(f.hessian(x))[0]

print('x = ', x, '\nJ = ', J, '\nE = ', E)

g = OP(lambda x: 5 - 4 * x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[1])
x = g.qnewton(x0=[-1, 1])
J = g.jacobian(x)
E = np.linalg.eig(g.hessian(x))[0]
print('x = ', x, '\nJ = ', J, '\nE = ', E)

# TODO: Find out why Mario's example gets different values. Mine gets optimal values !!