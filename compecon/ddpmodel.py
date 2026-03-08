__author__ = 'Randall'
import numpy as np
import pandas as pd
import scipy as sp
from compecon.tools import tic, toc, Options_Container, markov
from scipy.sparse import csc_matrix, diags, tril, identity
import warnings
import matplotlib.pyplot as plt

class DDPoptions(Options_Container):
    """ Container for numerical options to solve a DPmodel

    Attributes:
        algorithm   algorithm for solver
        tol         convergence tolerance parameter
        maxit       maximum number of iterations
        print       print iteration summary when solving the model
        vterm       terminal value
    """
    description = "Solver options for a DPmodel"

    def __init__(self, algorithm='newton', tol=np.sqrt(np.spacing(1)),
                 maxit=200, vterm=None, show=False):
        self.algorithm = algorithm
        self.tol = tol
        self.maxit = maxit
        self.vterm = vterm
        self.show = show

    def print_header(self, method, horizon):
        horizon = 'infinite' if np.isinf(horizon) else 'finite'
        if self.show:
            print('Solving %s-horizon discrete model by %s method' % (horizon, method))
            print('{:4s} {:12s} {:8s}'.format('iter', 'change', 'time'))
            print('-' * 30)

    def print_current_iteration(self, it, change, t0):
        """ Prints summary of current iteration in solve method

        Args:
          it: iteration number (scalar)
          change: distance between two iterations
          tic: time when iterations started

        Returns:
          prints output to screen
        """
        if self.show:
            print('{:4d}  {:12.1e}  {:8.4f}'.format(it, change, toc(t0)))

    def print_last_iteration(self, t0, change):
        """ Prints summary of last iteration in solve method

        Args:
          tic: time when iterations started
          change: distance between last two iterations

        Returns:
          prints output to screen
        """
        if self.show:
            if change >= self.tol:
                print('Failure to converge in DPmodel.solve()')
            print('Elapsed Time = {:7.2f} Seconds'.format(toc(t0)))


class DDPdims(Options_Container):
    """ Container for the dimensions of a DDPmodel object

    Attributes:
        n   number of discrete states
        m   number of discrete actions
    """
    description = "Dimensions of a DPmodel object"

    def __init__(self, m, n):
        self.m = m
        self.n = n


class DDPmodel(object):
    def __init__(self, reward, dynamics,
                 discount, horizon=np.inf,
                 **kwargs):
        """

        :param reward: m X n array, m discrete actions vs n discrete states
        :param dynamics:
        :param discount: float, discount factor (scalar)
        :param horizon: int, planning horizon
        :param kwargs: options for solving the problem
        """

        assert reward.ndim == 2, 'reward must be a 2-dimensional array'

        self.reward = reward
        self.discount = discount
        self.dims = DDPdims(*reward.shape)

        self.horizon = horizon
        self.opts = DDPoptions(**kwargs)

        n = self.dims.n

        self._infinite_horizon = np.isinf(horizon)

        if self._infinite_horizon:
            self.value = np.zeros(n)
            self.policy = np.zeros(n, dtype=int)
            self.transition = np.zeros((n, n))
        else:
            T = horizon
            self.opts.algorithm = 'backwards'
            self.value = np.zeros((T + 1, n))
            self.policy = np.zeros((T, n), dtype=int)
            self.transition = np.zeros((T + 1, n, n))

        txt = 'dynamics must have 2 (deterministic) or 3 (stochastic) dimensions'
        assert dynamics.ndim in [2, 3], txt
        self._is_deterministic = dynamics.ndim == 2

        self.P = dynamics.astype(int) if self._is_deterministic else dynamics

    def __valmax(self, v):
        if self._is_deterministic:
            Pv = v[self.P]
        else:
            Pv = (self.P @ v).squeeze()

        v = self.reward + self.discount * Pv
        x = np.argmax(v, 0)
        return v[x, np.arange(self.dims.n)], x

    def __valpol(self, x):
        n = self.dims.n
        nn = np.arange(n, dtype=int)
        if self._is_deterministic:
            snext = self.P[x, nn]
            pstar = csc_matrix((np.ones(n), (nn, snext)), (n, n)).toarray()
        else:
            pstar = self.P[x, nn]

        fstar = self.reward[x, nn]
        return pstar, fstar

    def solve(self, **kwargs):
        self.opts[kwargs.keys()] = kwargs.values()
        vt = self.opts.vterm
        if vt is not None:
            assert vt.size == self.dims.n, 'vterm must have {:d} elements (one per state'.format(self.dims.n)
            if self._infinite_horizon:
                self.value = vt
            else:
                self.value[-1] = vt

        algorithm = self.opts.algorithm
        if algorithm == 'newton':
            self.__solve_by_Newton_method()
        elif algorithm == 'funcit':
            self.__solve_by_function_iteration()
        elif algorithm == 'backwards':
            self.__solve_backwards()
        else:
            raise ValueError('Unknown algorithm')
        return self

    def __solve_backwards(self):
        self.opts.print_header('backward recursion', self.horizon)
        for t in range(self.horizon - 1, -1, -1):
            self.value[t], self.policy[t] = self.__valmax(self.value[t + 1])
            self.transition[t] = self.__valpol(self.policy[t])[0]  # only first output required

    def __solve_by_Newton_method(self):
        t0 = tic()
        n = self.dims.n
        delta = self.discount
        I_n = np.identity(n)

        self.opts.print_header("Newton's", self.horizon)
        for it in range(self.opts.maxit):
            vold = self.value.copy()
            tmp, self.policy = self.__valmax(vold)
            self.transition, fstar = self.__valpol(self.policy)
            self.value = np.linalg.solve(I_n - delta * self.transition, fstar)
            change = np.linalg.norm(self.value - vold)
            self.opts.print_current_iteration(it, change, t0)
            if change < self.opts.tol:
                break

        self.opts.print_last_iteration(t0, change)

    def __solve_by_function_iteration(self):
        t0 = tic()
        self.opts.print_header('function iteration', self.horizon)
        for it in range(self.opts.maxit):
            vold = self.value.copy()
            self.value, self.policy = self.__valmax(self.value)
            change = np.linalg.norm(self.value - vold)
            self.opts.print_current_iteration(it, change, t0)
            if change < self.opts.tol:
                break
        self.transition = self.__valpol(self.policy)[0]
        self.opts.print_last_iteration(t0, change)

    def solution(self):
        #TODO: Return model solution as pandas dataframe?
        pass

    def simulate(self, s0, nper):

        s0 = np.atleast_1d(s0)
        n = self.dims.n
        nrep = s0.size

        nper = nper if self.horizon > nper else self.horizon
        spath = np.zeros((nper + 1, nrep), dtype=int)
        spath[0] = s0

        if self._infinite_horizon:
            cp = self.transition.cumsum(1)
            for t in range(nper):
                r = np.random.rand(nrep, 1)
                spath[t + 1] = (r > cp[spath[t]]).sum(1)
        else:
            for t in range(nper):
                r = np.random.rand(nrep)
                cp = self.transition[t].cumsum(1)
                spath[t + 1] = (r > cp[spath[t]]).sum(1)

        xpath = self.policy[spath]
        return spath.squeeze(), xpath.squeeze()

    def markov(self):
        return markov(self.transition)


    def __repr__(self):
        txt = 'A ' + ('deterministic' if self._is_deterministic else 'stochastic')
        txt += ' discrete state, discrete action, dynamic model.\n'
        txt += 'There are {:} possible actions over {:} possible states'.format(self.dims.m, self.dims.n)
        return txt