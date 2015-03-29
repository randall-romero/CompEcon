import copy
from compecon import Basis, InterpolatorArray
import numpy as np
from scipy import linalg

__author__ = 'Randall'

class DPmodel(object):
    """
        A Dynamic Programming Model class

        A DPmodel object has the following attributes:

        -- Time dimension:
        * horizon:  time horizon (infinite)
        * discount: discount factor (required)

        -- Dimension of state and action spaces:
        * ds: number of continuous state variables (1)
        * dx: number of continuous action variables (1)
        * ni: number of discrete states (1)
        * nj: number of discrete actions (1)

        -- Stochastic components: Markov transition and continuous iid shocks:
        * e:  ne.de discretized continuous state transition shocks (0)
        * w:  ne.1 continuous state shock probabilities (1)
        * q:  ni.ni.nj discrete state transition probabilities (empty)
        * h:  nj.ni deterministic discrete state transitions (empty)
        * X:  nx.dx discretized continuous actions (empty)

        -- Value and policy functions: InterpolatorArray objects:
        * Value:    ni-array for the value function
        * Policy:   ni-array for the policy function
        * Value_j:  nj.ni-array for the value function at each discrete action
        * Policy_j: nj.ni-array for the policy function at each discrete action
        * DiscreteAction: ni.ns-array(integer), discrete actions at each state node

        -- Numerical solution:
        * algorithm:  algorithm for solver
        * tol:        convergence tolerance parameter
        * ncpmethod:  method for complementarity problem
        * maxit:      maximum number of iterations
        * maxitncp:   maximun number of iterations for ncpmethod
        * knownFunctions:  nj.ni-array(boolean), True if policy and value functions are known
        * D_reward_provided: True if derivatives of reward function are provided
        * D_transition_provided: True if derivatives of transition function are provided

        -- Output details and other parameters:
        * nr:      number of refined nodes
        * output:  print output per iterations
        * nc:      number of continuous nodes
        * ns:      number of continuous nodes on output ##TODO
        * xnames:  1.dx cell, cell of names for continuous actions
        * discretized: True if continuous action is discretized or not present.
     """
    # todo: Review the dimensions of attributes in above docstring

    def __init__(self, basis, ni, nj, dx):
        # -- Time dimension
        #self.horizon = T   I am going to assume it is infinite horizon, solve the finite horizon later
        self.discount = None

        #  -- Dimension of state and action spaces
        self.ds = basis.d
        self.ns = basis.N  # nc in matlab
        self.dx = dx
        self.ni = ni
        self.nj = nj

        #  -- Stochastic components
        self.e = 0
        self.w = 1
        self.q = None #[]
        self.h = None #[]
        self.X = None #np.zeros([1,0])

        #  -- Value and policy functions
        ns = self.ns
        self.Value = InterpolatorArray(basis, [ni])
        self.Value.y = np.zeros([ni, ns])

        self.Policy = InterpolatorArray(basis, [ni, dx])
        self.Policy.y = np.zeros([ni, dx, ns])

        self.Value_j = InterpolatorArray(basis, [ni, nj])
        self.Value_j.y = np.zeros([ni, nj, ns])

        self.Policy_j = InterpolatorArray(basis, [ni, nj, dx])
        self.Policy_j.y = np.zeros([ni, nj, dx, ns])

        self.DiscreteAction = np.zeros([ni, self.ns], int)

        #  -- Numerical solution:
        self.algorithm = 'newton'
        self.tol = np.sqrt(np.spacing(1))
        self.ncpmethod = 'minmax'
        self.maxit = 500
        self.maxitncp = 50
        self.knownFunctions = np.zeros([ni, nj], bool)
        self.D_reward_provided = True
        self.D_transition_provided = True

        #  -- Output details and other parameters:
        self.nr = 10
        self.output = True
        self.nc = None #[]
        self.xnames = None #[]
        self.discretized = False

    def solve(self):
        # Step necessary for discretized models
        if self.dx == 0 or self.X is not None:
            self.discretized = True

        if self.dx > 1 and self.discretized:
            if self.X.shape[0] != self.dx:
                raise ValueError('If model is discretized, field "X" must have {} columns'.format(self.dx))


    def getDerivative(self, func, s, x, *args, **kwargs):
        dx, nx = x.shape

        if func == 'reward':
            F = lambda X: self.reward(s, X, *args, **kwargs)
            df = 1
            f = F(x)
            #assert(f.ndim == 1, 'reward must return a numpy vector')
        elif func == 'transition':
            F = lambda X: self.transition(s, X, *args, **kwargs)
            df = self.ds
            f = F(x)
            #assert(f.shape[0] == df, 'transition must have {}rows'.format(df))
        else:
            raise ValueError('unknown function')

        ''' Compute Jacobian'''
        tol = np.spacing(1) ** (1/3)

        h = tol * np.maximum(abs(x), 1)
        x_minus_h = x - h
        x_plus_h = x + h
        deltaX = x_plus_h - x_minus_h
        fx = np.zeros([dx, df, nx])

        for k in range(dx):
            xx = x.copy()
            xx[k] = x_plus_h[k]
            fplus = F(xx)

            xx[k] = x_minus_h[k]
            fminus = F(xx)

            #fx[k] = (fplus - fminus) / np.tile(deltaX[k],(df, 1))
            fx[k] = (fplus - fminus) / deltaX[k]

        ''' Compute Hessian'''
        tol = np.spacing(1) ** (1/4)

        h = tol * np.maximum(abs(x), 1)
        x_minus_h = x - h
        x_plus_h = x + h
        deltaX = h  #repmat(h, 1, 1, dx)

        #deltaXX = deltaX .* permute(deltaX,[1,3,2])

        fxx = np.zeros([dx, dx, df, nx])
        for k in range(dx):
            for h in range(dx):
                xx = x.copy()
                if h == k:
                    xx[k] = x_plus_h[k]
                    fplus = F(xx)

                    xx[k] = x_minus_h[k]
                    fminus = F(xx)
                    fxx[k, k] = (fplus - 2 * f + fminus) / (deltaX[k] ** 2)
                else:
                    xx[k] = x_plus_h[k]
                    xx[h] = x_plus_h[h]
                    fpp = F(xx)

                    xx[h] = x_minus_h[h]
                    fpm = F(xx)

                    xx[k] = x_minus_h[k]
                    fmm = F(xx)

                    xx[h] = x_plus_h[h]
                    fmp = F(xx)

                    fxx[k, h] = (fpp + fmm - fpm - fmp) / (4 * deltaX[k] * deltaX[h])

        fxx = (fxx + fxx.swapaxes(0, 1)) / 2

        if func == 'reward':  # reduce the second dimension
            fx = fx[:, 0]
            fxx = fxx[:, :, 0]

        return f, fx, fxx

    def updateValue(self):
        value = self.Value_j.y
        self.DiscreteAction = jmax = np.array([np.argmax(Vi, 0) for Vi in value])
        for i in range(self.ni):
            self.Value[i] = value[i][jmax[i], range(self.ns)]

    def updatePolicy(self):
        policy = self.Policy_j.y.swapaxes(1, 2)
        jmax = self.DiscreteAction
        for i in range(self.ni):
            for k in range(self.dx):
                self.Policy[i, k] = policy[i, k][jmax[i], range(self.ns)]


# TODO: design this class:
"""
    * Should finite-infinite horizon models be split into two subclasses?
    * Should discretized models be handled by a subclass?
    * Should vmax operate directly on Value_j and Policy_j? how to deal with residu


als?
"""