from compecon.demos.setup import np, plt

from compecon.quad import qnwlogn
from compecon.tools import nodeunif
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Univariate Taylor approximation

x = np.linspace(-1, 1, 100)
y = (x + 1) * np.exp(2 * x)
y1 = 1 + 3 * x
y2 = 1 + 3 * x + 8 * x ** 2

plt.figure(figsize=[6, 6])
plt.plot(x, y, 'k', linewidth=3, label='Function')
plt.plot(x, y1, 'b', linewidth=3, label='1st order approximation')
plt.plot(x, y2, 'r', linewidth=3, label='2nd order approximation')
plt.legend()
plt.xticks([-1, 0, 1])
plt.show()

## Bivariate Taylor approximation
nplot = [101, 101]
a = [0, -1]
b = [2, 1]
x1, x2 = nodeunif(nplot, a, b)
x1.shape = nplot
x2.shape = nplot

y = np.exp(x2) * x1 ** 2
y1 = 2 * x1 - x2 - 1
y2 = x1 ** 2 - 2 * x1 * x2 + 0.5 * x2 ** 2 + x2

def newPlot(title, Y, k):
    ax = fig.add_subplot(1, 3, k, projection='3d')
    ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    plt.title(title)

fig = plt.figure(figsize=[15, 6])
newPlot('Original function', y, 1)
newPlot('First-order approximation error', y1-y, 2)
newPlot('Second-order approximation error', y2-y, 3)
plt.show()
