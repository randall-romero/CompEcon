"""
DEMSLV04 Compute fixedpoint of f(x, y)= [x^2 + y^3; xy - 0.5]

Compute fixedpoint of f(x, y)= [x^2 + y^3; x*y - 0.5] using Newton, Broyden, and function iteration methods.
Initial values generated randomly.  Some algorithms may fail to converge, depending on the initial value.
True fixedpoint is x = -0.09 y=-0.46.
"""

from demos.setup import np, tic, toc
from compecon import NLP
np.random.seed(12)



''' Set up the problem '''
def g(z):
    x, y = z
    return np.array([x **2 + y ** 3, x * y - 0.5])

problem_as_fixpoint = NLP(g, maxit=1500)

''' Equivalent Rootfinding Formulation'''
def f(z):
    x, y = z
    fval = [x - x ** 2 - y ** 3,
            y - x * y + 0.5]
    fjac = [[1 - 2 * x, -3 * y **2],
            [-y, 1 - x]]

    return np.array(fval), np.array(fjac)

problem_as_zero = NLP(f, maxit=1500)

'''% Randomly generate starting point'''
xinit = np.random.randn(2)

''' Compute fixed-point using Newton method '''
t0 = tic()
z1 = problem_as_zero.newton(xinit)
t1 = 100 * toc(t0)
n1 = problem_as_zero.fnorm

''' Compute fixed-point using Broyden method '''
t0 = tic()
z2 = problem_as_zero.broyden(xinit)
t2 = 100 * toc(t0)
n2 = problem_as_zero.fnorm

''' Compute fixed-point using function iteration '''
t0 = tic()
z3 = problem_as_fixpoint.fixpoint(xinit)
t3 = 100 * toc(t0)
n3 = np.linalg.norm(problem_as_fixpoint.fx - z3)

''' Print table header '''
print('Hundredths of seconds required to compute fixed-point of g(x1,x2)=[x1^2+x2^3;x1*x2-0.5]')
print('using Newton, Broyden, and function iteration methods, starting at')
print('x1 = {:4.2f}  x2 = {:4.2f}\n\n'.format(*xinit))
print('Method       Time   Norm of f         x1       x2\n', '-' * 45)
print('Newton   {:8.2f}    {:8.0e}     {:5.2f}     {:5.2f}'.format(t1, n1, *z1))
print('Broyden  {:8.2f}    {:8.0e}     {:5.2f}     {:5.2f}'.format(t2, n2, *z2))
print('Function {:8.2f}    {:8.0e}     {:5.2f}     {:5.2f}'.format(t3, n3, *z3))