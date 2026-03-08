from demos.setup import np, plt
from compecon import BasisChebyshev, BasisSpline
from compecon.tools import nodeunif

__author__ = 'Randall'

# DEMAPP06 Chebychev and cubic spline derivative approximation errors

# Function to be  approximated


def f(x):
    g = np.zeros((3, x.size))
    g[0], g[1], g[2] = np.exp(-x), -np.exp(-x), np.exp(-x)
    return g

# Set degree of approximation and endpoints of approximation interval
a =  -1                            # left endpoint
b =   1                            # right endpoint
n =  10                            # order of interpolatioin

# Construct refined uniform grid for error ploting
x = nodeunif(1001, a, b)

# Compute actual and fitted values on grid
y, d, s = f(x)                     # actual

# Construct and evaluate Chebychev interpolant
C = BasisChebyshev(n, a, b, f=f)    # chose basis functions
yc = C(x)                           # values
dc = C(x, 1)                        # first derivative
sc = C(x, 2)                        # second derivative


# Construct and evaluate cubic spline interpolant
S = BasisSpline(n, a, b, f=f)       # chose basis functions
ys = S(x)                           # values
ds = S(x, 1)                        # first derivative
ss = S(x, 2)                        # second derivative

# Plot function approximation error
plt.figure()
plt.subplot(2, 1, 1),
plt.plot(x, y - yc[0])
plt.ylabel('Chebychev')
plt.title('Function Approximation Error')

plt.subplot(2, 1, 2)
plt.plot(x, y - ys[0])
plt.ylabel('Cubic Spline')
plt.xlabel('x')


# Plot first derivative approximation error
plt.figure()
plt.subplot(2, 1, 1),
plt.plot(x, d - dc[0])
plt.ylabel('Chebychev')
plt.title('First Derivative Approximation Error')

plt.subplot(2, 1, 2)
plt.plot(x, d - ds[0], 'm')
plt.ylabel('Cubic Spline')
plt.xlabel('x')

# Plot second derivative approximation error
plt.figure()
plt.subplot(2, 1, 1),
plt.plot(x, s - sc[0])
plt.ylabel('Chebychev')
plt.title('Second Derivative Approximation Error')

plt.subplot(2, 1, 2)
plt.plot(x, s - ss[0], 'm')
plt.ylabel('Cubic Spline')
plt.xlabel('x')

plt.show()
