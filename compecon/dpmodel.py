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

    def __init__(self, basis, ni=1, nj=1):
        # -- Time dimension
        #self.horizon = T   I am going to assume it is infinite horizon, solve the finite horizon later
        self.discount = None

        #  -- Dimension of state and action spaces
        self.ds = 1
        self.dx = 1
        self.ni = ni
        self.nj = nj

        #  -- Stochastic components
        self.e = 0
        self.w = 1
        self.q = None #[]
        self.h = None #[]
        self.X = None #np.zeros([1,0])

        #  -- Value and policy functions
        self.Value = InterpolatorArray(basis, [ni])
        self.Policy = InterpolatorArray(basis, [ni])
        self.Value_j = InterpolatorArray(basis, [nj, ni])
        self.Policy_j = InterpolatorArray(basis, [nj, ni])
        self.DiscreteAction = []

        #  -- Numerical solution:
        self.algorithm = 'newton'
        self.tol = np.sqrt(np.spacing(1))
        self.ncpmethod = 'minmax'
        self.maxit = 500
        self.maxitncp = 50
        self.knownFunctions = np.array([nj, ni], bool)
        self.D_reward_provided = True
        self.D_transition_provided = True

        #  -- Output details and other parameters:
        self.nr = 10
        self.output = True
        self.nc = None #[]
        self.ns = None #[]
        self.xnames = None #[]
        self.discretized = False

# TODO: design this class:
"""
    * Should finite-infinite horizon models be split into two subclasses?
    * Should discretized models be handled by a subclass?
    * Should vmax operate directly on Value_j and Policy_j? how to deal with residuals?
"""