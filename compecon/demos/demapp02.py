from demos.setup import np, plt
from compecon import BasisChebyshev, BasisSpline
from compecon.tools import nodeunif
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



""" Approximating functions on R^2

This m-file illustrates how to use CompEcon Toolbox routines to construct and operate with an approximant for a
function defined on a rectangle in R^2.

In particular, we construct an approximant for f(x1,x2)=cos(x1)/exp(x2) on [-1,1]X[-1,1].  The function used in this
illustration posseses a closed-form, which will allow us to measure approximation error precisely. Of course, in
practical applications, the function to be approximated will not possess a known closed-form.

In order to carry out the exercise, one must first code the function to be approximated at arbitrary points.
Let's begin:
"""
# Function to be approximated
f = lambda x: np.sin(x[0]) / np.exp(x[1])

# Set the points of approximation interval:
a = -1  # left points
b =  1  # right points

# Choose an approximation scheme. In this case, let us use an 11 by 11 Chebychev approximation scheme:
n = 11  # order of approximation
basis = BasisChebyshev([n, n], a, b)  # write n twice to indicate the two dimensions. a and b are expanded

# Compute the basis coefficients c.  There are various way to do this:
# One may compute the standard approximation nodes x and corresponding interpolation matrix Phi and function values y
# and use:
x = basis.nodes
Phi = basis.Phi(x)  # input x may be omitted if evaluating at the basis nodes
y = f(x)
c = np.linalg.solve(Phi, y)
print('Interpolation coeff =\n ', c)

# Alternatively, one may compute the standard approximation nodes x and corresponding function values y and use these
# values to create an BasisChebyshev object with keyword argument y:
x = basis.nodes
y = f(x)
fa = BasisChebyshev([n, n], a, b, y=y)
print('Interpolation coeff =\n ', fa.c)  # attribute c returns the coefficients

# ... or one may simply pass the function directly to BasisChebyshev using keyword 'f', which by default
# will evaluate it at the basis nodes
F = BasisChebyshev([n, n], a, b, f=f)
print('Interpolation coeff =\n ', F.c)

# Having created a BasisChebyshev object, one may now evaluate the approximant at any point x by calling the object:
x = np.zeros([2, 1])  # first dimension should match the basis dimension
y = F(x)
print('The exact and approximate value of f at x=[0 0] are')
print('{:4.0f}  {:20.15f}\n'.format(0, y))

# ... one may also evaluate the approximant's first partial derivatives at x:
d1 = F(x, [1, 0])
d2 = F(x, [0, 1])
print('The exact and approximate partial derivatives of f w.r.t. x1 at x=[0 0] are')
print('{:4.0f}  {:20.15f}\n'.format(1, d1))
print('The exact and approximate partial derivatives of f w.r.t. x2 at x=[0 0] are')
print('{:4.0f}  {:20.15f}\n'.format(0, d2))

# ... one may also evaluate the approximant's second own partial and cross
# partial derivatives at x:
d11 = F(x, [2, 0])
d22 = F(x, [0, 2])
d12 = F(x, [1, 1])
print('The exact and approximate second partial derivatives of f w.r.t. x1 at x=[0 0] is')
print('{:4.0f}  {:20.15f}\n'.format(0, d11))
print('The exact and approximate second partial derivatives of f w.r.t. x2 at x=[0 0] is')
print('{:4.0f}  {:20.15f}\n'.format(0, d22))
print('The exact and approximate second cross partial derivatives of f at x=[0 0] is')
print('{:4.0f}  {:20.15f}\n'.format(-1, d12))

# One may evaluate the accuracy of the Chebychev polynomial approximant by computing the approximation error on a
# highly refined grid of points:
nplot = [101, 101]         # chose grid discretization
X = nodeunif(nplot, [a, a], [b, b])  # generate refined grid for plotting
yapp = F(X)        # approximant values at grid nodes
yact = f(X)                      # actual function values at grid points
error = (yapp - yact).reshape(nplot)
X1, X2 = X
X1.shape = nplot
X2.shape = nplot

fig = plt.figure(figsize=[15, 6])
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X1, X2, error, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('error')
plt.title('Chebychev Approximation Error')

# The plot indicates that an order 11 by 11 Chebychev approximation scheme
# produces approximation errors no bigger in magnitude than 1x10^-10.



# Let us repeat the approximation exercise, this time constructing an order
# 21 by 21 cubic spline approximation scheme:
n = [21, 21]                          # order of approximation
S = BasisSpline(n, a, b, f=f)
yapp = S(X)        # approximant values at grid nodes
error = (yapp - yact).reshape(nplot)
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X1, X2, error, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('error')
plt.title('Cubic Spline Approximation Error')
plt.show()

# The plot indicates that an order 21 by 21 cubic spline approximation
# scheme produces approximation errors no bigger in magnitude than 1x10^-6.

