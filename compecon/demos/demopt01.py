__author__ = 'Randall'

import numpy as np
from compecon import OP
np.set_printoptions(4, linewidth=120)
import matplotlib.pyplot as plt


f = lambda x: x * np.cos(x ** 2)
problem = OP(f)

x1 = problem.golden(0, 1)
x2 = problem.golden(2, 3)

plt.figure()
x = np.linspace(0,3, 500)
plt.plot(x, problem.f(x), 'k')
plt.plot(x1,problem.f(x1),'r', marker='o', markersize=16)
plt.plot(x2,problem.f(x2),'g', marker='o', markersize=16)
plt.title('Maximization of x cos(x^2) via golden search')
plt.show()
