"""
DEMSLV02 Compute root of Rosencrantz function

Compute root of f(x1,x2)= [200*x1*(x2-x1 ** 2)+1-x1;100*(x1 ** 2-x2)] using Newton and Broyden
methods. Initial values generated randomly.  True root is x1=1 x2=1.
"""
from demos.setup import np, tic, toc
from compecon import NLP




''' Set up the problem '''
def f(x):
    fval = [200 * x[0] * (x[1] - x[0] ** 2) + 1 - x[0], 100 * (x[0] ** 2 - x[1])]
    fjac = [[200 * (x[1] - x[0] ** 2) - 400 * x[0] ** 2 - 1, 200 * x[0]],
            [200 * x[0], -100]]
    return np.array(fval), np.array(fjac)

problem = NLP(f)

''' Randomly generate starting point '''
problem.x0 = np.random.randn(2)

''' Compute root using Newton method '''
t0 = tic()
x1 = problem.newton()
t1 = 100 * toc(t0)
n1 = problem.fnorm

'''Compute root using Broyden method '''
t0 = tic()
x2 = problem.broyden()
t2 = 100 * toc(t0)
n2 = problem.fnorm


''' Print results '''
print('Hundreds of seconds required to compute root of Rosencrantz function')
print('f(x1,x2)= [200*x1*(x2-x1^2)+1-x1;100*(x1^2-x2)] via Newton and Broyden')
print('methods, starting at x1 = {:4.2f} x2 = {:4.2f}'.format(*problem.x0))
print('\nMethod      Time   Norm of f        x1     x2\n', '-' * 45)
print('Newton  %8.2f    %8.0e     %5.2f  %5.2f' % (t1, n1, x1[0], x1[1]))
print('Broyden %8.2f    %8.0e     %5.2f  %5.2f' % (t2, n2, x2[0], x2[1]))
