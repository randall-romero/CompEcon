__author__ = 'Randall'
import numpy as np
import scipy as sp
from compecon.tools import tic, toc, Options_Container
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
                 maxit=200, vterm=None, print=False):
        self.algorithm = algorithm
        self.tol = tol
        self.maxit = maxit
        self.vterm = vterm
        self.print = print

    def print_header(self, method, horizon):
        horizon = 'infinite' if np.isinf(horizon) else 'finite'
        if self.print:
            print('Solving %s-horizon model collocation equation by %s method' % (horizon, method))
            print('{:4s} {:12s} {8s}'.format('iter', 'change', 'time'))
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
        if self.print:
            print('{:4i}  {:12.1e}  {:8.4f}'.format(it, change, toc(t0)))

    def print_last_iteration(self, t0, change):
        """ Prints summary of last iteration in solve method

        Args:
          tic: time when iterations started
          change: distance between last two iterations

        Returns:
          prints output to screen
        """
        if self.print:
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

    def __init__(self, n, m):
        self.n = n
        self.m = m


class DDPmodel(object):
    def __init__(self, reward, dynamics,
                 discount, horizon=np.inf,
                 S = None, X = None,
                 **kwargs):

        assert reward.ndim == 2, 'reward must be a 2-dimensional array'

        self.reward = reward
        self.discount = discount
        self.dims = DDPdims(*reward.shape)

        self.horizon = horizon
        self.opts = DDPoptions(**kwargs)

        n, m = self.dims['n', 'm']

        self._infinite_horizon = np.isinf(horizon)

        if self._infinite_horizon:
            self.value = np.zeros(n)
            self.policy = np.zeros((n), dtype=int)
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

        self.P = dynamics.astype(int) if self._is_deterministic else np.swapaxes(dynamics, 0, 1)
        self.S = np.arange(n) if S is None else S
        self.X = np.arange(m) if X is None else X

    def __valmax(self, v):
        if self._is_deterministic:
            Pv = v[self.P]
        else:
            Pv = np.dot(self.P, v).squeeze()

        v = self.reward + self.discount * Pv
        x = np.argmax(v, 1)
        return v[np.arange(self.dims.n), x], x

    def __valpol(self, x):
        n = self.dims.n
        nn = np.arange(n, dtype=int)
        if self._is_deterministic:
            snext = self.P[nn, x]
            pstar = csc_matrix((np.ones(n), (nn, snext)), (n, n)).toarray()
        else:
            pstar = self.P[nn, x]

        fstar = self.reward[nn, x]
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
        if algorithm is 'newton':
            self.__solve_by_Newton_method()
        elif algorithm is 'funcit':
            self.__solve_by_function_iteration()
        elif algorithm is 'backwards':
            self.__solve_backwards()
        else:
            raise ValueError('Unknown algorithm')

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

    def plot_value(self):
        # todo finish this method
        plt.figure()
        plt.plot(self.S, self.value)
        plt.title('Optimal Value Function')
        plt.xlabel('Age of Machine')
        plt.ylabel('Value')


    def markov(self):
        p = self.transition

        # Error Checking to ensure P is a valid stochastic matrix
        assert p.ndim == 2, 'Transition matrix does not have two dimensions'
        n, n2 = p.shape
        assert n == n2, 'Transition matrix is not square'
        assert np.all(p >= 0), 'Transition matrix contains negative elements'
        assert np.all(np.abs(p.sum(1) - 1) < 1e-12), 'Rows of transition matrix do not sum to 1'

        spones = lambda A: (A != 0).astype(int)

        # Determine accessibility from i to j
        f = np.empty_like(p)
        for j in range(n):
            dr = 1
            r = spones(p[:,j])  # a vector of ones where p(i,j)~=0
            while np.any(dr):
                dr = r
                r = spones(p.dot(r) + r)
                dr = r - dr
                f[:, j] = r

        # Determine membership in recurrence classes
        ind = np.zeros_like(p)
        numrec = -1  # number of recurrence classes
        for i in range(n):
            if np.all(ind[i] == 0):
                j = f[i]  # states accessible from i
                if np.all((f[:, i].T * j) == j):  # are all accessible states communicating states?
                    j = np.where(j)[0]           # members in class with state i
                    k = j.size                  # number of members
                    if k:
                        numrec += 1
                        ind[j, numrec] = 1

        ind = ind[:, :numrec + 1]        # ind(i,j)=1 if state i is in class j

        # Determine recurrence class invariant probabilities
        q = np.zeros((n, numrec + 1))
        for j in range(1 + numrec):
            k = np.where(ind[:, j])[0]
            nk = k.size
            k0, k1 = np.ix_(k, k)
            A = np.asarray(np.r_[np.ones((1, nk)), identity(nk) - p[k0, k1].T])
            B = np.asarray(np.r_[np.ones((1, 1)), np.zeros((nk, 1))])
            q[k, j] = np.linalg.lstsq(A, B)[0].flatten()

        return q  # todo: Mario's code has a second output argument

