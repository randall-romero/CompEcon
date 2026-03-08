from demos.setup import np, plt
from compecon import BasisChebyshev
from compecon.tools import nodeunif
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


""" Approximating using the CompEcon toolbox """

'''Univariate approximation'''
# Define function and derivative
f1 = lambda x: np.exp(-2 * x)
d1 = lambda x: -2 * np.exp(-2 * x)

# Fit approximant
n, a, b = 10, -1, 1
f1fit = BasisChebyshev(n, a, b, f=f1)


# Graph approximation error for function and derivative
axopts = {'xlabel': 'x', 'ylabel': 'Error', 'xticks': [-1, 0, 1]}
x = np.linspace(a, b, 1001)
fig = plt.figure(figsize=[12, 6])

ax1 = fig.add_subplot(121, title='Function approximation error', **axopts)
ax1.axhline(linestyle='--', color='gray', linewidth=2)
ax1.plot(f1fit.nodes, np.zeros_like(f1fit.nodes), 'ro', markersize=12)
ax1.plot(x, f1fit(x) - f1(x))

ax2 = fig.add_subplot(122, title='Derivative approximation error', **axopts)
ax2.plot(x, np.zeros_like(x), '--', color='gray', linewidth=2)
ax2.plot(f1fit.nodes, np.zeros_like(f1fit.nodes), 'ro', markersize=12)
ax2.plot(x, f1fit(x, 1) - d1(x))


''' Bivariate Interpolation '''
# Define function
f2 = lambda x: np.cos(x[0]) / np.exp(x[1])

# Set degree and domain interpolation
n, a, b = 7, 0.0, 1.0
f2fit = BasisChebyshev([n, n], a, b, f=f2)

# Nice plot of function approximation error
nplot = [101, 101]
x = nodeunif(nplot, a, b)
x1, x2 = x
error = f2fit(x) - f2(x)
error.shape = nplot
x1.shape = nplot
x2.shape = nplot

fig = plt.figure(figsize=[15, 6])
ax = fig.gca(projection='3d', title='Chebyshev Approximation Error',
             xlabel='$x_1$', ylabel='$x_2$', zlabel='Error')
ax.plot_surface(x1, x2, error, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.show()

# Compute partial Derivatives
x = np.array([[0.5], [0.5]])
order = [[1, 0, 2, 1, 0],
         [0, 1, 0, 1, 2]]

ff = f2fit(x, order)

print('x   = [0.5, 0.5]\nf1  = {:7.4f}\nf2  = {:7.4f}\nf11 = {:7.4f}\nf12 = {:7.4f}\nf22 = {:7.4f}'.format(*ff))
