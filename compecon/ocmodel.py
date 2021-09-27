


import numpy as np
import pandas as pd

from .ode import ODE
from .tools import Options_Container, gridmake
import time



class OCoptions(Options_Container):
    description = "Solver options for a OCmodel"

    def __init__(self, tol=np.sqrt(np.spacing(1)), maxit=80, show=True, nr=10):
        self.tol = tol
        self.maxit = maxit
        self.show = show
        self.nr = nr

    def print_header(self):
        if self.show:
            print('Solving optimal control model')
            print('{:4s} {:12s} {:8s}'.format('iter', 'change', 'time'))
            print('-' * 30)

    def print_current_iteration(self, it, change, tic):
        if self.show:
            print(f'{it:4d}  {change:12.1e}  {time.time() - tic:8.4f}')

    def print_last_iteration(self, tic, change):
        if self.show:
            if change >= self.tol:
                print('Failure to converge in OCmodel.solve()')
            print(f'Elapsed Time = {time.time() - tic:7.2f} Seconds')



class OCmodel(object):

    def __init__(self, basis, control, reward, transition, rho=0.0, params=[]):

        assert callable(control), 'control must be a function'
        assert callable(reward), 'reward must be a function'
        assert callable(transition), 'transition must be a function'

        self.__x = lambda s, Vs: control(s, Vs, *params)
        self.__f = lambda s, x: reward(s, x, *params)
        self.__g = lambda s, x: transition(s, x, *params)

        #  Value and policy functions
        self.Value = basis.duplicate()
        self.Policy = basis.duplicate()

        # Time parameters
        # self.time = OCtime(discount, horizon)
        self.rho = rho

        # Labels for model variables
        # self.labels = OClabels(basis.opts.labels, x, i, j)

        # Default numerical solution parameters and parameters for model functions
        self.options = OCoptions()
        self.params = params

        ''' <<<<<<<<<<<<<<<<<<<             END OF CONSTRUCTOR        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    def solve(self, **kwargs):
        tic = time.time()
        # Set user options to defaults, if not set by user with OPTSET (see above)

        self.options[kwargs.keys()] = kwargs.values()

        self.options.print_header()
        ds = self.Value.d  # dimension of state variable s
        s = self.Value.nodes  # collocation nodes
        order = np.eye(ds, dtype=int)

        # Derive interpolation matrices
        Phi1 = self.Value.Phi(order=np.eye(ds, dtype=int), dropdim=False)

        c = self.Value.c
        # Policy iteration
        for it in range(self.options.maxit):

            cold = c.copy()
            Vs = self.Value(s, order, dropdim=False)

            if Vs.shape[1] == 1:
                Vs = Vs[:, 0]
            else:
                print('ERROR CON LAS DIMENSIONES DE Vs')

            x = self.__x(s, Vs)
            f = self.__f(s, x)
            g = self.__g(s, x)

            B = self.rho * self.Value.Phi()

            for _is in range(ds):
                B -= np.diag(g[_is]) @ Phi1[_is]

            c = np.linalg.solve(B, f.T).T
            self.Value.c = c
            if np.isnan(c).any() or np.isinf(c).any():
                print('NaNs or Infs encountered')
                return

            change = np.abs(c - cold).max()
            self.options.print_current_iteration(it, change, tic)

            if change < self.options.tol:
                break

        self.options.print_last_iteration(tic, change)
        self.Policy.y = x

        return self.solution()

    def solution(self, nr=10, resid=True):
        """
        Computes solution over a refined grid

        """
        ds = self.Value.d
        labels = self.Value.opts._labels
        order = np.eye(ds, dtype=int)

        a = self.Value.a
        b = self.Value.b
        n = self.Value.n
        sr = np.atleast_2d(gridmake(*[np.linspace(a[i], b[i], nr * n[i]) for i in range(self.Value.d)]))

        ''' MAKE DATABASE'''
        # ADD CONTINUOUS STATE VARIABLE
        DATA = pd.DataFrame(sr.T, columns=labels)

        # SET INDEX FOR DATA
        if ds == 1:
            slab = DATA[labels[0]]
            DATA.index = slab

        # ADD VALUE FUNCTION
        DATA['value'] = self.Value(sr)

        Vs = self.Value(sr, order, dropdim=False)
        if Vs.shape[1] == 1:
            Vs = Vs[:, 0]
        else:
            print('ERROR CON LAS DIMENSIONES DE Vs')

        # ADD CONTROL
        xr = self.__x(sr, Vs)
        DATA['control'] = xr.T

        # ADD RESIDUAL IF REQUESTED
        if resid:
            f = self.__f(sr, xr)
            g = self.__g(sr, xr)
            DATA['resid'] = self.rho * DATA['value'] - (f + (Vs * g).sum(axis=0)).flatten()

        return DATA

    def simulate(self, sinit, T, N=1000):

        # ****** 1: Preparation***********************************************************
        ds = self.Value.d

        # Determine number of replications nrep and periods nper to be simulated.
        sinit = np.atleast_2d(sinit).astype(float)
        ds2, nrep = sinit.shape
        assert ds == ds2, 'initial continous state must have {} rows'.format(ds)

        # ***** *2: Simulate the model ***************************************************
        problem = ODE(lambda s: self.__g(s, self.Policy(s)), T, sinit)
        problem.rk4(N, self.Value.opts._labels)

        DATA = problem.x
        DATA['control'] = self.Policy(DATA.values)
        return DATA

