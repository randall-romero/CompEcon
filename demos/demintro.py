"""
INTRODUCTORY DEMOS:

This module contains the introductory demos from CompEcon
"""
from compecon.quad import qnwlogn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def demintro01():
    """ Inverse Demand Problem """

    demand = lambda p: 0.5 * p ** -0.2 + 0.5 * p ** -0.5
    derivative = lambda p: -0.01 * p ** -1.2 - 0.25 * p ** -1.5

    p = 0.25
    for it in range(100):
        f = demand(p) - 2
        d = derivative(p)
        s = -f / d
        p += s
        print('iteration {:3d} price {:8.4f}'.format(it, p))
        if np.linalg.norm(s) < 1.0e-8:
            break

    # Generate demand function
    pstar = p
    qstar = demand(pstar)
    n, a, b = 100, 0.02, 0.40
    p = np.linspace(a, b, n)
    q = demand(p)

    # Graph demand function
    plt.subplot(1, 2, 1)
    plt.plot(p, q, 'b')
    plt.xticks([0.0, 0.2, 0.4])
    plt.yticks([0, 2, 4])
    plt.xlim(0, 0.4)
    plt.ylim(0, 4)
    plt.gca().set_aspect(.1)
    plt.title('Demand')
    plt.xlabel('p')
    plt.ylabel('q')

    # Graph inverse demand function
    plt.subplot(1, 2, 2)
    plt.plot(q, p, 'b')
    plt.plot([0, 2, 2], [pstar, pstar, 0], 'r--')
    plt.plot([2], [pstar], 'ro', markersize=12)
    plt.yticks([0.0, pstar, 0.2, 0.4])
    plt.gca().set_yticklabels(['0.0', '$p^{*}$', '0.2', '0.4'])
    plt.xticks([0, 2, 4])
    plt.ylim(0, 0.4)
    plt.xlim(0, 4)
    plt.gca().set_aspect(10)
    plt.title('Inverse Demand')
    plt.ylabel('p')
    plt.xlabel('q')

    plt.show()


def discmoments(w, x):
    """ Generate means and standard deviations of discrete distribution.

    :param w: discrete probabilities
    :param x: values for k distinct discrete variates
    :return: (xavg, xstd) tuple with means and standard deviations
    """
    xavg = np.dot(x, w)
    xstd = np.sqrt(np.dot(x ** 2, w) - xavg ** 2)
    return xavg, xstd

def demintro02():
    """ Rational Expectations Agricultural Market Model """

    # Generate yield distribution
    sigma2 = 0.2 ** 2
    y, w = qnwlogn(25, -0.5 * sigma2, sigma2)

    # Compute rational expectations equilibrium using function iteration, iterating on acreage planted
    A = lambda aa, pp: 0.5 + 0.5 * np.dot(w, np.maximum(1.5 - 0.5 * aa * y, pp))

    ptarg = 1
    a = 1
    for it in range(50):
        aold = a
        a = A(a, ptarg)
        print('{:3d} {:8.4f} {:8.1e}'.format(it, a, np.linalg.norm(a - aold)))
        if np.linalg.norm(a - aold) < 1.e-8:
            break

    # Intermediate outputs
    q = a * y                # quantity produced in each state
    p = 1.5 - 0.5 * a * y    # market price in each state
    f = np.maximum(p, ptarg) # farm price in each state
    r = f * q                # farm revenue in each state
    g = (f - p) * q          #government expenditures

    xavg, xstd = discmoments(w, np.vstack((p, f, r, g)))
    varnames = ['Market Price', 'Farm Price', 'Farm Revenue', 'Government Expenditures']

    # Print results
    print('\n{:24s} {:8s} {:8s}'.format('Variable', 'Expect', 'Std Dev'))
    for varname, av, sd in zip(varnames, xavg, xstd):
        print('{:24s} {:8.4f} {:8.4f}'.format(varname, av, sd))

    # Generate fixed-point mapping
    aeq = a
    a = np.linspace(0, 2, 100)
    g = np.array([A(k, ptarg) for k in a])

    # Graph rational expectations equilibrium
    plt.figure(figsize=[6, 6])
    plt.plot(a, g, 'b', linewidth=4)
    plt.plot(a, a, ':', color='grey', linewidth=2)
    plt.plot([0, aeq, aeq], [aeq, aeq, 0], 'r--', linewidth=3)
    plt.plot([aeq], [aeq], 'ro', markersize=12)
    plt.gca().set_aspect(1)
    plt.xticks([0, aeq, 2])
    plt.yticks([0, aeq, 2])
    plt.gca().set_xticklabels(['0', '$a^{*}$', '2'])
    plt.gca().set_yticklabels(['0', '$a^{*}$', '2'])
    plt.xlabel('Acreage Planted')
    plt.ylabel('Rational Acreage Planted')
    plt.text(0.05, 0, '45${}^o$', color='grey')
    plt.text(1.85, aeq - 0.15,'$g(a)$', color='blue')
    plt.show()

    # Compute rational expectations equilibrium as a function of the target price
    nplot = 50
    ptarg = np.linspace(0, 2, nplot)
    a = 1
    Ep = np.empty(nplot)
    Ef = np.empty(nplot)
    Er = np.empty(nplot)
    Eg = np.empty(nplot)
    Sp = np.empty(nplot)
    Sf = np.empty(nplot)
    Sr = np.empty(nplot)
    Sg = np.empty(nplot)

    for ip in range(nplot):
        for it in range(50):
            aold = a
            a = A(a, ptarg[ip])
            if np.linalg.norm((a - aold) < 1.e-10):
                break

        q = a * y                       # quantity produced
        p = 1.5 - 0.5 * a * y           # market price
        f = np.maximum(p, ptarg[ip])    # farm price
        r = f * q                       # farm revenue
        g = (f - p) * q                 # government expenditures

        xavg, xstd = discmoments(w, np.vstack((p, f, r, g)))
        Ep[ip], Ef[ip], Er[ip], Eg[ip] = tuple(xavg)
        Sp[ip], Sf[ip], Sr[ip], Sg[ip] = tuple(xstd)


    zeroline = lambda Y: plt.plot([ptarg[0], ptarg[-1]], [Y[0], Y[0]],
                                  ':', color='gray')

    # Graph expected prices vs target price
    plt.figure(figsize=[8, 6])
    plt.subplot(1, 2, 1)
    zeroline(Ep)
    plt.plot(ptarg, Ep, linewidth=4, label='Market Price')
    plt.plot(ptarg, Ef, linewidth=4, label='Farm Price')
    plt.title('Expected price')
    plt.xlabel('Target price')
    plt.ylabel('Expectation')
    plt.legend()
    plt.xticks([0, 1, 2])
    plt.yticks([0.5, 1, 1.5, 2])
    plt.ylim(0.5, 2.0)

    # Graph expected prices vs target price
    plt.subplot(1, 2, 2)
    zeroline(Sf)
    plt.plot(ptarg, Sp, linewidth=4, label='Market Price')
    plt.plot(ptarg, Sf, linewidth=4, label='Farm Price')
    plt.title('Price variabilities')
    plt.xlabel('Target price')
    plt.ylabel('Standard deviation')
    plt.legend()
    plt.xticks([0, 1, 2])
    plt.yticks([0, 0.1, 0.2])
    #plt.ylim(0.5, 2.0)
    plt.show()


    # Graph expected farm revenue vs target price
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 3, 1)
    zeroline(Er)
    plt.plot(ptarg, Er, linewidth=4)
    plt.title('Expected revenue')
    plt.xlabel('Target price')
    plt.ylabel('Expectation')
    plt.xticks([0, 1, 2])
    plt.yticks([1, 2, 3])
    plt.ylim(0.8, 3.0)

    # Graph standard deviation of farm revenue vs target price
    plt.subplot(1, 3, 2)
    zeroline(Sr)
    plt.plot(ptarg, Sr, linewidth=4)
    plt.title('Farm Revenue Variability')
    plt.xlabel('Target price')
    plt.ylabel('Standard deviation')
    plt.xticks([0, 1, 2])
    plt.yticks([0, 0.2, 0.4])

    # Graph expected government expenditures vs target price
    plt.subplot(1, 3, 3)
    zeroline(Eg)
    plt.plot(ptarg, Eg, linewidth=4)
    plt.title('Expected Government Expenditures')
    plt.xlabel('Target price')
    plt.ylabel('Expectation')
    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.ylim(-0.05, 2.0)
    plt.show()


if __name__ == '__main__':
    import sys
    args = sys.argv
    print(args)
    if len(args) == 1:
        print('\n\nRunning demintro01')
        demintro01()
        print('\n\nRunning demintro02')
        demintro02()
    elif args[1] == '1':
        print('\n\nRunning demintro01')
        demintro01()
    elif args[1] == '2':
        print('\n\nRunning demintro02')
        demintro02()
    else:
        print('Unknown argument {}'.format(args[1]))