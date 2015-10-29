__author__ = 'Randall'

import numpy as np
from compecon import OP
np.set_printoptions(4, linewidth=120)
import matplotlib.pyplot as plt

''' Set up the problem '''
x0 = [1, 0]
banana = OP(lambda x: -100 * (x[1] - x[0] ** 2)**2 - (1 - x[0]) ** 2,
            x0, maxit=250, print=True, all_x=True)


''' Plots options '''
steps_options = {'marker': 'o',
                 'color': (0.2, 0.2, .81),
                 'linewidth': 1.0,
                 'markersize': 9,
                 'markerfacecolor': 'white',
                 'markeredgecolor': 'red'}

contour_options = {'levels': -np.exp(np.arange(0.25, 20, 0.5)),
                   'colors': '0.25',
                   'linewidths': 0.5}

''' Data for coutours '''
n = 40  # number of grid points for plot per dimension
xmin = [-0.7, -0.2]
xmax = [ 1.2,  1.2]

X0, X1 = np.meshgrid(*[np.linspace(a, b, n) for a, b in zip(xmin, xmax)])
Y = banana.f([X0.flatten(), X1.flatten()])
Y.shape = (n, n)

plt.figure()

for it, method in enumerate(banana.search_methods.keys()):
    ''' Solve problem with given method '''
    print('\n\nMaximization with method ' + method.upper())
    x = banana.qnewton(SearchMeth=method)
    print('x =', x)

    ''' Plot the result '''
    plt.subplot(1, 3, it + 1)
    plt.contour(X0, X1, Y, **contour_options)
    plt.plot(*banana.x_sequence, **steps_options)
    plt.plot(1, 1, 'r*', markersize=15)
    plt.title(method.upper() + " search")
    plt.xlabel('x_0', verticalalignment='top')
    plt.ylabel('x_1', verticalalignment= 'bottom')
    plt.axis((xmin[0], xmax[0], xmin[1], xmax[1]))

plt.show()

#TODO: find out why steepest search method looks different from Miranda's
#TODO: add the Nelder Mead method