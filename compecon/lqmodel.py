from compecon.tools import Options_Container, qzordered
import numpy as np
import pandas as pd
from compecon.tools import jacobian, hessian, gridmake, indices


__author__ = 'Randall'


class LQlabels(Options_Container):
    """ Container for labels of the LQmodel variables

    Attributes:
        s  labels for continuous state variables
        x  labels for continuous action variables
    """
    description = "Labels for LQmodel variables"

    def __init__(self, s, x):
        self.s = list(s)
        self.x = list(x)

    @property
    def snext(self):
        return [k + '_next' for k in self.s]

    @property
    def p(self):
        return ['value_' + k for k in self.s]




class LQmodel(object):
    """
        A Linear-Quadratic dynamic programming model class

        Solves discrete time linear quadratic control model using Ricatti equation methods

    Uses QZ decomposition to solve the Ricatti equation of a deterministic  stationary
    infinite-horizon linear-quadratic dynamic optimization model

    max_x f0 + fs*s +  fx*x + 0.5*s'*fss*s + s'*fsx*x +0.5*x'*fxx*x
    s.t. s' = g0 + gs*s + gx*x

    The optimal policy function is
       x(s) = xstar + X*(s-sstar)

    The shadow price function is
       p(s) = pstar + P*(s-sstar)

    The value function is
       V(s) = vstar + pstar*(s-sstar) + 0.5*(s-sstar)'*P*(s-sstar)

    The controlled state process is
      snext = sstar + G*(s-sstar)


     """
    # TODO: write the docstring

    def __init__(self, f0,fs,fx,fss,fsx,fxx,g0,gs,gx,delta, slabels=None, xlabels=None):
        """
        Args:
            f0: 1.1   objective function parameter
            fs: 1.ds  objective function parameter
            fx: 1.dx  objective function parameter
            fss: ds.ds objective function parameter
            fsx: ds.dx objective function parameter
            fxx: dx.dx objective function parameter
            g0: ds.1  state transition function parameter
            gs: ds.ds state transition function parameter
            gx: ds.dx state transition function parameter
            delta: discount factor
        """

        fs, fx, fss, fsx, fxx, g0, gs, gx = np.atleast_2d(fs,fx,fss,fsx,fxx,g0,gs,gx)

        # Determine dimensions of state and action variables
        ds = fs.size
        dx = fx.size

        fs.shape = 1, ds
        fx.shape = 1, dx

        # Check conformability
        assert fss.shape == (ds, ds), f'error in LQmodel: fss must be a {ds} by {ds} matrix'
        assert fsx.shape == (ds, dx), f'error in LQmodel: fsx must be a {ds} by {dx} matrix'
        assert fxx.shape == (dx, dx), f'error in LQmodel: fxx must be a {dx} by {dx} matrix'
        assert g0.shape == (ds, 1), f'error in LQmodel: g0 must be a {ds} by 1 matrix'
        assert gs.shape == (ds, ds), f'error in LQmodel: gs must be a {ds} by {ds} matrix'
        assert gx.shape == (ds, dx), f'error in LQmodel: gx must be a {ds} by {dx} matrix'

        self.f0 = f0
        self.fs = fs
        self.fx = fx
        self.fss = fss
        self.fsx = fsx
        self.fxx = fxx
        self.g0 = g0
        self.gs = gs
        self.gx = gx
        self.delta = delta
        self.dims ={'ds': ds, 'dx': dx}
        self.X = np.nan
        self.P = np.nan
        self.G = np.nan

        '''MAKE THE LABELS'''
        if slabels is None:
            slabels = ['s'] if ds == 1 else [f's{i}' for i in range(ds)]

        if xlabels is None:
            xlabels = ['x'] if dx == 1 else [f'x{i}' for i in range(dx)]

        self.labels = LQlabels(slabels, xlabels)

        '''SOLVE THE MODEL'''
        self.solve()
        ''' <<<<<<<<<<<<<<<<<<<             END OF CONSTRUCTOR        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


    def __repr__(self):
        pass

        txt = 'A continuous state, ' + ('continuous' if self.dims.dx > 0 else 'discrete') + ' action dynamic model.\n'
        txt = txt.upper()
        txt += '\n\t* Continuous states:\n'
        n, a, b = self.Value.n, self.Value.a, self.Value.b
        for k in range(self.Value.d):
            txt += "\t\t{:<2d}:  {:<s} --> {:d} nodes in [{:.2f}, {:.2f}]\n".format(k, self.labels.s[k], n[k], a[k], b[k])
        if self.dims.dx > 0:
            txt += '\n\t* Continuous actions\n'
            for v, vlab in enumerate(self.labels.x):
                txt += '\t\t{:<2d}:  {:s}\n'.format(v, vlab)
        if self.dims.ni > 1:
            txt += '\n\t* Discrete states\n'
            for v, vlab in enumerate(self.labels.i):
                txt += '\t\t{:<2d}:  {:s}\n'.format(v, vlab)
        if self.dims.nj > 1:
            txt += '\n\t* Discrete choices:\n'
            for v, vlab in enumerate(self.labels.j):
                txt += '\t\t{:<2d}:  {:s}\n'.format(v, vlab)

        return txt


    @property
    def steady_state(self):
        return self.steady['s'], self.steady['x'], self.steady['p'], self.steady['v']


    def Value(self, ss):
        '''

        Args:
            ss: state evaluation points, ds.ns

        Returns:
            value function at ss

        Note:
            The value function is
            V(s) = vstar + pstar' * (s-sstar) + 0.5*(s-sstar)'*P*(s-sstar)
        '''
        sstar, xstar, pstar, vstar = self.steady_state
        ss0 = ss-sstar # ds.ns
        Pss0 = self.P @ ss0
        ss0Pss0 = [k.dot(h) for k, h in zip(ss0.T, Pss0.T)]

        return vstar + pstar.T @ ss0 + 0.5 * np.array(ss0Pss0)

    def Policy(self, ss):
        '''

        Args:
            ss: state evaluation points

        Returns:
            policy function at ss

        Notes:
            The optimal policy function is
            x(s) = xstar + X * (s - sstar)

        '''
        sstar, xstar, pstar, vstar = self.steady_state
        return xstar + self.X @ (ss - sstar)

    def Shadow(self, ss):
        '''

        Args:
            ss: state evaluation points

        Returns:
            shadow price function at ss

        Notes:
            The shadow price function is
            p(s) = pstar + P * (s - sstar)

        '''
        sstar, xstar, pstar, vstar = self.steady_state
        return pstar + self.P @ (ss - sstar)

    def Next(self, ss):
        '''

        Args:
            ss: state evaluation points

        Returns:
            controlled state process at ss

        Notes:
            The controlled state process is
            snext(s) = sstar + G * (s - sstar)

        '''
        sstar, xstar, pstar, vstar = self.steady_state
        return sstar + self.G @ (ss - sstar)

    def solution(self, ss):
        """
        Computes solution over a refined grid

        s:  -- array  >> compute solution over provided values
        """

        ds, dx = self.dims['ds'], self.dims['dx']

        '''GET THE DATA'''
        ss = np.atleast_2d(ss)
        assert ss.shape[0] == ds, 'provided s grid must have {} rows'.format(ds)
        xr = self.Policy(ss)
        vr = self.Value(ss)
        pr = self.Shadow(ss)
        snext = self.Next(ss)

        ''' MAKE DATABASE'''
        DATA = pd.DataFrame(np.r_[ss, xr, vr, pr, snext].T,
                            columns=self.labels.s + self.labels.x + ['value'] + self.labels.p + self.labels.snext)

        '''SET INDEX FOR DATA'''
        if ds == 1:
            DATA.index = DATA[self.labels.s[0]]

        return DATA



    def simulate(self, nper, sinit, seed=None):

        # Simulate the model
        #
        #   S = self.simulate(nper, sinit, iinit)
        #
        # nper = number of periods to simulate (scalar)
        # sinit = initial continuos state (nrep x ds), where nrep is number of repetitions

        #
        # S = simulation results (table), with variables:
        #    r, repetion number
        #    t, time period
        #    s, continuous state
        #    x, optimal continuous action

        # ****** 1: Preparation***********************************************************
        #TODO: ADD THE STOCHASTIC COMPONENT

        ds, dx, = self.dims['ds'], self.dims['dx']

        if seed:
            np.random.seed(seed)

        # Determine number of replications nrep and periods nper to be simulated.
        # nper cannot exceed time.horizon.
        sinit = np.atleast_2d(sinit).astype(float)
        ds2, nrep = sinit.shape
        assert ds==ds2, 'initial continous state must have {} rows'.format(ds)


        ### Allocate memory to output arrays
        ssim = np.empty((nper+1, ds, nrep))
        xsim = np.empty((nper, dx, nrep))

        ### Set initial states
        ssim[0] = sinit

        # ***** *2: Simulate the model ***************************************************
        for t in range(nper):
            xsim[t] = self.Policy(ssim[t])
            ssim[t+1] = self.Next(ssim[t])

        ### Trim the last observation
        ssim = ssim[:nper]

        # ****** 3: Make a table with the simulated data *********************************

        '''MAKE THE LABELS'''
        slabels = ['s'] if ds == 1 else [f's{i}' for i in range(ds)]
        xlabels = ['x'] if dx == 1 else [f'x{i}' for i in range(dx)]

        '''MAKE DATA MATRICES'''
        sdata = ssim.swapaxes(0, 1).reshape((ds, -1))
        xdata = xsim.swapaxes(0, 1).reshape((dx, -1))

        ### Add variables rsim and tsim to identify the repetition number and the time
        # period of each observation
        tsim, rsim = gridmake(np.arange(nper), np.arange(nrep))

        # Make the table.
        DATA = pd.DataFrame()
        DATA['time'] = tsim
        if nrep > 1:
            DATA['_rep'] = rsim

        for k, slab in enumerate(slabels):
            DATA[slab] = sdata[k]

        for k, xlab in enumerate(xlabels):
            DATA[xlab] = xdata[k]

        return DATA


    def solve(self):
        # Unpak data
        ds = self.dims['ds']
        dx = self.dims['dx']
        delta = self.delta

        f0, fx, fxx = self.f0, self.fx, self.fxx
        g0, gx = self.g0, self.gx

        fs, fsx, fss, gs = self.fs, self.fsx, self.fss, self.gs

        ''' Solve Riccati equation using QZ decomposition '''
        dx2ds = dx + 2 * ds
        A = np.zeros((dx2ds, dx2ds))
        A[:ds, :ds] = np.identity(ds)
        A[ds:-ds, -ds:] = -delta * gx.T
        A[-ds:, -ds:] = delta * gs.T

        B = np.zeros_like(A)
        B[:ds, :-ds] = np.c_[gs, gx]
        B[ds: -ds, :-ds] = np.c_[fsx.T, fxx]
        B[-ds:] = np.c_[-fss, -fsx, np.identity(ds)]

        S, T, Q, Z = qzordered(A, B)
        C = np.real(np.linalg.solve(Z[:ds, :ds].T, Z[ds:, :ds].T)).T
        X = C[:dx]
        P = C[dx:, :]
        G = gs + gx @ X

        self.X = X
        self.P = P
        self.G = G

        ''' Compute steady-state state, action, and shadow price'''
        t0 = np.r_[np.c_[fsx.T, fxx, delta * gx.T],
                  np.c_[fss, fsx, delta*gs.T - np.eye(ds)],
                  np.c_[gs - np.eye(ds), gx, np.zeros((ds, ds))]]
        t1 = np.r_[-fx.T, -fs.T, -g0]
        t = np.linalg.solve(t0, t1)
        sstar, xstar, pstar = np.split(t, [ds, ds + dx])
        vstar = (f0 + fs @ sstar + fx @ xstar + 0.5 * sstar.T @ fss @ sstar +
                 sstar.T @ fsx @ xstar + 0.5 * xstar.T @ fxx @ xstar) / (1 - delta)

        self.steady = {'s':sstar, 'x': xstar, 'p': pstar, 'v':vstar}
