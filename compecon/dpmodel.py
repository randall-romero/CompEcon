import copy
from compecon import Basis, Interpolator_array
import numpy as np
from scipy import linalg

__author__ = 'Randall'

class DPmodel(object):
    ## Model structure
    # The economic agent's (consumer, firm, country) problem is characterized by the following parameters (default values in paranthesis):
    
    def __init__(self, basis, ni, nj):
        ###
        # *Time dimensions*
        #
        # * horizon:  time horizon (infinite)
        # * discount: discount factor (required)
        self.horizon = np.inf
        self.discount = None
        
        ###
        # *Dimension of state and action spaces*
        #
        # * ds: dimension of continuous state (1)
        # * dx: dimension of continuous action (1)
        # * ni: number of discrete states (1)
        # * nj: number of discrete actions (1)
        self.ds = 1
        self.dx = 1
        self.ni = ni
        self.nj = nj
        
        ###
        # *Stochastic componests*: Markov transition and continuous iid shocks
        #
        # * e:  ne.de discretized continuous state transition shocks (0)
        # * w:  ne.1 continuous state shock probabilities (1)
        # * q:  ni.ni.nj discrete state transition probabilities (empty)
        # * h:  nj.ni deterministic discrete state transitions (empty)
        # * X:  nx.dx discretized continuous actions (empty)
        self.e = 0
        self.w = 1
        self.q = []
        self.h = []
        self.X = zeros(1,0)
        
        ###
        # *Value and policy functions*: interpolator objects
        #
        # * Value:    interpolator object to approximate the value function
        # * Policy:   interpolator object to approximate the policy function
        # * Value_j:  ni.nj interpolator object to approximate value function for each discrete action
        # * Policy_j: ni.nj interpolator objecto
        # * DiscreteAction: ns.ni matrix for discrete actions
        self.Value = Interpolator_array(basis, [ni])
        self.Policy = Interpolator_array(basis, [ni])
        self.Value_j = Interpolator_array(basis, [nj, ni])
        self.Policy_j = Interpolator_array(basis, [nj, ni])
        self.DiscreteAction = []
        
        ###
        # *Numerical solution*:
        #
        # * algorithm:  algorithm for solver
        # * tol:        convergence tolerance parameter
        # * ncpmethod:  method for complementarity problem
        # * maxit:      maximum number of iterations
        # * maxitncp:   maximun number of iterations for ncpmethod
        # * knownFunctions:  ni.nj bolean array, True if discrete policy and value functions are known
        self.algorithm = 'newton'
        self.tol       = sqrt(eps)
        self.ncpmethod = 'minmax'
        self.maxit     = 500
        self.maxitncp  = 50
        self.knownFunctions
        
        ###
        # *Output details and other parameters*
        #
        # * nr:      number of refined nodes
        # * output:  print output per iterations
        # * nc:      number of continuous nodes
        # * ns:      number of continuous nodes on output ##TODO
        # * xnames:  1.dx cell, cell of names for continuous actions
        # * discretized: True if continuous action is discretized or not present.
        self.nr        = 10
        self.output    = True
        self.nc        = []
        self.ns        = []
        self.xnames = {}
        self.discretized = False
        self.D_reward_provided = True
        self.D_transition_provided = True
 
