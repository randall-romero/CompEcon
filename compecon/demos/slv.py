import warnings
import numpy as np
from numpy.linalg import norm
import time
from compecon import lcpsolve, ncpsolve, minmax, ssmooth, MCP
import matplotlib.pyplot as plt
import seaborn as sns

tic = lambda: time.time()
toc = lambda t: time.time() - t



def demslv07old():
    """Linear complementarity problem methods    """

    # Generate problem test data
    n = 8
    z = np.random.randn(n, 2) - 1
    a = np.min(z, 1)
    b = np.max(z, 1)
    q = np.random.randn(n)
    M = np.random.randn(n, n)
    M = - np.dot(M.T, M)
    x0 = np.random.randn(n)

    print('Hundreds of seconds required to solve randomly generated linear \n',
          'complementarity problem on R^8 using Newton and Lemke methods')
    print('\nAlgorithm           Time      Norm \n{}'.format('-' * 40));

    G = lambda x, z: norm(minmax(x, a, b, z))

    '''Solve my applying Newton method to minmax formulation '''
    t1 = tic()
    x, z = lcpsolve(M, q, a, b, x0, kind='minmax', showiters=True)
    print('Newton minmax     {:6.2f}  {:8.0e}'.format(100 * toc(t1), G(x, z)))

    '''Solve my applying Newton method to semi-smooth formulation '''
    t2 = tic()
    x, z = lcpsolve(M, q, a, b, x0, showiters=True)
    print('Newton semismooth {:6.2f}  {:8.0e}'.format(100 * toc(t2), G(x, z)))


def demslv07():
    """Linear complementarity problem methods    """

    # Generate problem test data
    n = 8
    z = np.random.randn(n, 2) - 1
    a = np.min(z, 1)
    b = np.max(z, 1)
    q = np.random.randn(n)
    M = np.random.randn(n, n)
    M = - np.dot(M.T, M)
    x0 = np.random.randn(n)


    # Define the problem by creating an MCP object
    L = MCP(M, a, b, q, showiters=True)

    # Solve my applying Newton method to semi-smooth formulation
    t1 = tic()
    x1, z1 = L.zero(x0)
    t1 = toc(t1)

    # Solve my applying Newton method to minmax formulation
    t2 = tic()
    x2, z2 = L.zero(x0, transform='minmax')
    t2 = toc(t2)


    # Print results
    G = lambda x, z: norm(minmax(x, a, b, z))

    print('Hundreds of seconds required to solve randomly generated linear \n',
          'complementarity problem on R^8 using Newton and Lemke methods')
    print('\nAlgorithm           Time      Norm \n{}'.format('-' * 40))
    print('Newton semismooth {:6.2f}  {:8.0e}'.format(100 * t1, G(x1, z1)))
    print('Newton minmax     {:6.2f}  {:8.0e}'.format(100 * t2, G(x2, z2)))




def demslv08():
    """Nonlinear complementarity problem methods

    Solve nonlinear complementarity problem on R^2 using semismooth and minmax methods
    """

    ''' function to be solved'''
    def f(z):
        x, y = z
        fval = np.array([200 * x * (y - x ** 2) + 1 - x,
                         100 * (x ** 2 - y)])
        fjac = np.array([[200 * (y - x ** 2) - 400 * x ** 2 - 1, 200 * x],
                         [200 * x, -100]])

        return fval, fjac

    # Generate problem test data
    z = 2 * np.random.randn(2, 2)
    a = 1 + np.min(z, 0)
    b = 1 + np.max(z, 0)
    x0 = np.random.randn(2)

    hdr = 'Hundreds of seconds required to solve nonlinear complementarity \n' \
          'problem on R^2 using minmax and semismooth formulations, with \n' \
          'randomly generated bounds \n\ta = [{:4.2f}, {:4.2f}] \n\tb = [{:4.2f}, {:4.2f}]'
    print(hdr.format(a[0], a[1], b[0], b[1]))
    print('\nAlgorithm           Time      Norm        x1     x2\n{}'.format('-' * 56));

    # Define the problem
    P = MCP(f, a, b, maxit=1500)

    '''Solve my applying Newton method to minmax formulation '''
    t1 = tic()
    x, z = P.zero(x0, transform='minmax')
    print('Newton minmax     {:6.2f}  {:8.0e}     {:5.2f}  {:5.2f}'.format(100*toc(t1), norm(minmax(x, a, b, z)), *x))



    '''Solve my applying Newton method to semismooth formulation '''
    t2 = tic()
    x, z = P.zero(x0)
    print('Newton semismooth {:6.2f}  {:8.0e}     {:5.2f}  {:5.2f}'.format(100*toc(t2), norm(minmax(x, a, b, z)), *x))



def demslv08old():
    """Nonlinear complementarity problem methods

    Solve nonlinear complementarity problem on R^2 using semismooth and minmax methods
    """

    ''' function to be solved'''
    def f(z):
        x, y = z
        fval = np.array([200 * x * (y - x ** 2) + 1 - x,
                         100 * (x ** 2 - y)])
        fjac = np.array([[200 * (y - x ** 2) - 400 * x ** 2 - 1, 200 * x],
                         [200 * x, -100]])

        return fval, fjac

    # Generate problem test data
    z = 2 * np.random.randn(2, 2)
    a = 1 + np.min(z, 0)
    b = 1 + np.max(z, 0)
    x0 = np.random.randn(2)

    hdr = 'Hundreds of seconds required to solve nonlinear complementarity \n' \
          'problem on R^2 using minmax and semismooth formulations, with \n' \
          'randomly generated bounds \n\ta = [{:4.2f}, {:4.2f}] \n\tb = [{:4.2f}, {:4.2f}]'
    print(hdr.format(a[0], a[1], b[0], b[1]))
    print('\nAlgorithm           Time      Norm        x1     x2\n{}'.format('-' * 56));



    '''Solve my applying Newton method to minmax formulation '''
    t1 = tic()
    x, z = ncpsolve(f, a, b, x0, kind='minmax', maxit=1500)
    print('Newton minmax     {:6.2f}  {:8.0e}     {:5.2f}  {:5.2f}'.format(100*toc(t1), norm(minmax(x, a, b, z)), *x))



    '''Solve my applying Newton method to semismooth formulation '''
    t2 = tic()
    x, z = ncpsolve(f, a, b, x0, maxit=1500)
    print('Newton semismooth {:6.2f}  {:8.0e}     {:5.2f}  {:5.2f}'.format(100*toc(t2), norm(minmax(x, a, b, z)), *x))




def demslv09():
    """ Hard nonlinear complementarity problem with Billup's function

     Solve hard nonlinear complementarity problem on R using semismooth and minmax methods.  Problem involves Billup's
     function.  Minmax formulation fails semismooth formulation suceeds.
    """

    warnings.simplefilter("ignore")

    ''' Billups' function'''
    def billups(x):
        fval = 1.01 - (1 - x) ** 2
        fjac = 2 * (1 - x)
        return fval, fjac

    ''' Set up the problem '''
    # Generate problem test data
    x0 = 0.0
    a = 0
    b = np.inf

    Billups = MCP(billups, a, b)

    # Print table header
    print('Hundreds of seconds required to solve hard nonlinear complementarity')
    print('problem using Newton semismooth and minmax formulations')
    print('Algorithm           Time      Norm         x')


    # Solve by applying Newton method to minmax formulation
    t1 = tic()
    x, z = Billups.zero(x0, transform='minmax')
    t1 = 100 * toc(t1)
    print('Newton minmax     {:6.2f}  {:8.0e}     {:5.2f}'.format(t1, norm(minmax(x, a, b, z)), x[0]))

    # Solve by applying Newton method to semismooth formulation
    t2 = tic()
    x, z = Billups.zero(x0)
    t2 = 100 * toc(t2)
    print('Newton semismooth {:6.2f}  {:8.0e}     {:5.2f}'.format(t2, norm(minmax(x, a, b, z)), x[0]))

    ''' Make figure '''
    # Minmax reformulation of Billups' function
    billupsm = lambda x: minmax(x, 0, np.inf, billups(x)[0])

    # Semismooth reformulation of Billups' function
    billupss = lambda x: ssmooth(x, 0, np.inf, billups(x)[0])




    fig = plt.figure()

    x = np.linspace(-0.5, 2.5, 500)
    y = Billups.ssmooth(x)
    ax1 = fig.add_subplot(121, title='Difficult NCP', aspect=1,
                         xlabel='x', xlim=[-0.5, 2.5], ylim=[-1, 1.5])
    ax1.axhline(ls='--', color='gray')
    ax1.plot(x, billupss(x), label='Semismooth')
    ax1.plot(x, billupsm(x), label='Minmax')
    ax1.legend(loc='bottom')

    x = np.linspace(-0.03, 0.03, 500)
    ax2 = fig.add_subplot(122, title='Difficult NCP Magnified', aspect=1,
                          xlabel='x', xlim = [-.035, .035], ylim=[ -.01, .06])
    ax2.axhline(ls='--', color='gray')
    ax2.plot(x, billupss(x), label='Semismooth')
    ax2.plot(x, billupsm(x), label='Minmax')
    ax2.legend(loc='best')
    plt.show()

def demslv09old():
    """ Hard nonlinear complementarity problem with Billup's function

     Solve hard nonlinear complementarity problem on R using semismooth and minmax methods.  Problem involves Billup's
     function.  Minmax formulation fails semismooth formulation suceeds.
    """

    warnings.simplefilter("ignore")

    ''' Billups' function'''
    def billups(x):
        fval = 1.01 - (1 - x) ** 2
        fjac = 2 * (1 - x)
        return fval, fjac

    ''' Define a timed version of ncpsolve '''
    def ncpsolved(*args, **kwargs):
        t = time.time()
        return ncpsolve(*args,**kwargs), 100 * (time.time() - t)

    ''' Set up the problem '''

    # Generate problem test data
    x0 = 0.0
    a = 0
    b = np.inf

    # Print table header
    print('Hundreds of seconds required to solve hard nonlinear complementarity')
    print('problem using Newton semismooth and minmax formulations')
    print('Algorithm           Time      Norm         x')

    # Solve by applying Newton method to minmax formulation
    (x, z), t1 = ncpsolved(billups, a, b, x0, kind='minmax')
    print('Newton minmax     {:6.2f}  {:8.0e}     {:5.2f}'.format(t1, norm(minmax(x, a, b, z)), x[0]))

    # Solve by applying Newton method to semismooth formulation
    (x, z), t2 = ncpsolved(billups, a, b, x0, kind='ssmooth')
    print('Newton semismooth {:6.2f}  {:8.0e}     {:5.2f}'.format(t2, norm(minmax(x, a, b, z)), x[0]))

    ''' Make figure '''
    # Minmax reformulation of Billups' function
    billupsm = lambda x: minmax(x, 0, np.inf, billups(x)[0])

    # Semismooth reformulation of Billups' function
    billupss = lambda x: ssmooth(x, 0, np.inf, billups(x)[0])


    fig = plt.figure()

    x = np.linspace(-0.5, 2.5, 500)
    ax1 = fig.add_subplot(121, title='Difficult NCP', aspect=1,
                         xlabel='x', xlim=[-0.5, 2.5], ylim=[-1, 1.5])
    ax1.axhline(ls='--', color='gray')
    ax1.plot(x, billupss(x), label='Semismooth')
    ax1.plot(x, billupsm(x), label='Minmax')
    ax1.legend(loc='bottom')

    x = np.linspace(-0.03, 0.03, 500)
    ax2 = fig.add_subplot(122, title='Difficult NCP Magnified', aspect=1,
                          xlabel='x', xlim = [-.035, .035], ylim=[ -.01, .06])
    ax2.axhline(ls='--', color='gray')
    ax2.plot(x, billupss(x), label='Semismooth')
    ax2.plot(x, billupsm(x), label='Minmax')
    ax2.legend(loc='best')
    plt.show()


demslv09()
#demslv07old()