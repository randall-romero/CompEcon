"""
DEMSLV01 Compute root of f(x)=exp(-x)-1

Compute root of f(x)=exp(-x)-1 using Newton and secant methods. Initial value generated randomly. True root is x=0.
"""
from demos.setup import np, plt, tic, toc
from numpy.linalg import norm
from compecon import NLP


''' Set up the problem '''
def f(x):
    fval = np.exp(-x) - 1
    fjac = -np.exp(-x)
    return fval, fjac

problem = NLP(f, all_x=True)

''' Randomly generate starting point '''
problem.x0 = 10 * np.random.randn(1)

''' Compute root using Newton method '''
t0 = tic()
x1 = problem.newton()
t1 = 100 * toc(t0)
n1, x_newton = problem.fnorm, problem.x_sequence


''' Compute root using Broyden method '''
t0 = tic()
x2 = problem.broyden()
t2 = 100 * toc(t0)
n2, x_broyden = problem.fnorm, problem.x_sequence


''' Print results '''
print('Hundredths of seconds required to compute root of exp(-x)-1,')
print('via Newton and Broyden methods, starting at x = %4.2f.' % problem.x0)
print('\nMethod      Time   Norm of f   Final x')
print('Newton  %8.2f    %8.0e     %5.2f' % (t1, n1, x1))
print('Broyden %8.2f    %8.0e     %5.2f' % (t2, n2, x2))

''' View current options for solver '''
print(problem.opts)

''' Describe the options '''
print(problem.opts.__doc__)

''' Plot the convergence '''
b = -abs(problem.x0)
a = -b
xx = np.linspace(a, b, 100)

fig = plt.figure()
plt.plot(xx, f(xx)[0], 'b-')
plt.plot(x_newton,f(x_newton)[0],'ro:')
plt.show()