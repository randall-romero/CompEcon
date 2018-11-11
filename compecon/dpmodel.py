import time

from compecon.tools import Options_Container, qzordered
from compecon.nonlinear import MCP
from compecon.lcpstep import lcpstep
from compecon.lqmodel import LQmodel
import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse import block_diag, kron, issparse, identity
from scipy.sparse.linalg import spsolve
from compecon.tools import jacobian, hessian, gridmake, indices
from inspect import getargspec
#from .lcpstep import lcpstep  # todo: is it worth to add lcpstep?
import warnings


#fixme In docstrings, indicate that discrete model should not include x in definitions

__author__ = 'Randall'


class DPtime(Options_Container):
    """ Container for the time parameters of a DPmodel object

    Attributes:
        discount  scalar, discount factor in the interval (0,1)
        horizon   scalar, agents horizon
    """
    def __init__(self, discount=0.0, horizon=np.inf):
        self.discount = discount
        self.horizon = horizon


class DPrandom(Options_Container):
    """ Container for the random components of a DPmodel object

    Attributes:
        e   continuous state transition shocks.
        w   continuous state shock probabilities.
        q   discrete state transition probabilities.
    """
    description = 'Random components of a DPmodel object'

    def __init__(self, ni, nj, e=None, w=None, q=None, h=None):
        if ni == 1:
            qq = np.ones((nj, 1, 1))
        else:
            txt = 'If the model has 2 or more discrete states, a state transition must be provided: ' + \
                  "either deterministic (option 'h') or stochastic ('q'). "
            assert (q is None) ^ (h is None), txt

            qq = np.zeros((nj, ni, ni))
            if q is None:
                h = np.atleast_2d(h)
                for i in range(ni):
                    for j in range(nj):
                        qq[j, i, h[j, i]] = 1
            else:
                q = np.array(q)
                if q.ndim == 2:  # assume it is the same regardless of choice
                    qq[:] = q
                else:
                    qq = q
                ss = qq.shape
                assert ss[1] == ss[2], 'A Markov transition matrix must be square (last two dimensions of q must be equal)'
                # assert np.allclose(qq.sum(-1), 1.0), 'The rows of a Markov transition must add up to 1.0'
                #fixme set tolerance level, seems to be too strict

        self.e = np.zeros((2, 1)) if e is None else np.atleast_2d(e)
        self.w = np.ones((1)) if w is None else np.atleast_1d(w)
        self.q = qq


class DPdims(Options_Container):
    """ Container for the dimensions of a DPmodel object

    Attributes:
        ds  dimension of continuous state s
        dx  dimension of continuous action x
        ni  number of discrete states i
        nj  number of discrete actions j
        ns  number of continuous state nodes output
        nx  number of discretized continuous actions, if applicable
        nc  number of collocation polynomials = of coefficients to be determined
        dr  number of restrictions on x (other than bounds on individual x variables)
    """
    description = "Dimensions of a DPmodel object"

    def __init__(self, ds=1, ns=None,
                 dx=1, nx=None,
                 ni=1, nj=1, ne=1,
                 nc=None):
        self.ds = ds
        self.ns = ns
        self.dx = dx
        self.nx = nx
        self.ni = ni
        self.nj = nj
        self.ne = ne
        self.nc = nc
        self.dr = 0


class DPlabels(Options_Container):
    """ Container for labels of the DPmodel variables

    Attributes:
        s  labels for continuous state variables
        x  labels for continuous action variables
        i  labels for discrete states
        j  labels for discrete choices
    """
    description = "Labels for DPmodel variables"

    def __init__(self, s, x, i, j):
        self.s = s
        self.x = x
        self.i = i
        self.j = j


class DPoptions(Options_Container):
    """ Container for numerical options to solve a DPmodel

    Attributes:
        algorithm             algorithm for solver
        tol                   convergence tolerance parameter
        ncpmethod             method for complementarity problem
        maxit                 maximum number of iterations
        maxitncp              maximunm number of iterations for ncpmethod
        discretized           true if continuous action is discretized or not present.
        X                     dx.nx discretized continuous actions
        D_reward_provided     true if Jacobian and Hessian of reward are provided
        D_transition_provided true if Jacobian and Hessian of transition are provided
        knownFunctions        ni.nj boolean array, true if discrete policy and value functions are known
        print                whether to print output
    """
    description = "Solver options for a DPmodel"

    def __init__(self, algorithm='newton', tol=np.sqrt(np.spacing(1)), ncpmethod='minmax',
                 maxit=80, maxitncp=50, discretized=False, X=None,
                 knownFunctions=None, show=True, print=None):
        self.algorithm = algorithm
        self.tol = tol
        self.ncpmethod = ncpmethod
        self.maxit = maxit
        self.maxitncp = maxitncp
        self.show = show
        self.discretized = discretized
        self.X = X
        self.knownFunctions = knownFunctions
        if print is not None:
            warnings.warn("Keyword 'print=' is deprecated. Use 'show=' instead.")


    def print_header(self, method, horizon):
        horizon = 'infinite' if np.isinf(horizon) else 'finite'
        if self.show:
            print('Solving %s-horizon model collocation equation by %s method' % (horizon, method))
            print('{:4s} {:12s} {:8s}'.format('iter', 'change', 'time'))
            print('-' * 30)

    def print_current_iteration(self, it, change, tic):
        """ Prints summary of current iteration in solve method

        Args:
          it: iteration number (scalar)
          change: distance between two iterations
          tic: time when iterations started

        Returns:
          prints output to screen
        """
        if self.show:
            print('{:4d}  {:12.1e}  {:8.4f}'.format(it, change, time.time() - tic))

    def print_last_iteration(self, tic, change):
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
            print('Elapsed Time = {:7.2f} Seconds'.format(time.time() - tic))


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

        -- Value and policy functions: Interpolator objects:
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

    def __init__(self, basis, reward, transition, bounds=None, restrictions=None,
                 i=('State 0',), j=('Choice 0', ), x=(),
                 discount=0.0, horizon=np.inf,
                 e=None, w=None, q=None, h=None, params=None):

        assert callable(reward), 'reward must be a function'
        assert callable(transition), 'transition must be a function'
        if bounds is None:
            assert len(x) == 0, 'Bounds for continuous action are missing!'
        else:
            assert callable(bounds), 'bounds must be a function'

        self.__b = bounds
        self.__f = reward
        self.__g = transition
        self.__h = restrictions

        assert isinstance(i, (list, tuple)), 'i must be a tuple of strings (names of discrete states)'
        assert isinstance(j, (list, tuple)), 'j must be a tuple of strings (names of discrete choices)'
        assert isinstance(x, (list, tuple)), 'x must be a tuple of strings (names of continuous actions)'

        i, j, x = tuple(i), tuple(j), tuple(x)
        ni, nj, dx = len(i), len(j), len(x)

        assert (nj > 1) or (dx > 0), 'Model did not specified any policy variable! Set j or x (or both).'

        #  Value and policy functions
        if np.isinf(horizon):
            self.Value = basis.duplicate(l=[i])
            self.Value_j = basis.duplicate(l=[i, j])
            self.Policy = basis.duplicate(l=[i, x])
            self.Policy_j = basis.duplicate(l=[i, j, x])
            self.DiscreteAction = np.zeros([ni, basis.N], int)
        else:
            t0 = np.arange(horizon)
            t1 = np.arange(horizon + 1)
            self.Value = basis.duplicate(l=[t1, i])
            self.Value_j = basis.duplicate(l=[t1, i, j])
            self.Policy = basis.duplicate(l=[t0, i, x])
            self.Policy_j = basis.duplicate(l=[t0, i, j, x])
            self.DiscreteAction = np.zeros([horizon, ni, basis.N], int)

        # Time parameters
        self.time = DPtime(discount, horizon)

        # Labels for model variables
        self.labels = DPlabels(basis.opts.labels, x, i, j)

        # Stochastic specification
        self.random = DPrandom(ni, nj, e, w, q, h)


        # Default numerical solution parameters and parameters for model functions
        self.options = DPoptions()
        self.params = params

        # Model dimensions
        self.dims = DPdims(basis.d,  # number of continuous state variables
                           basis.N,  # number of continuous state nodes
                           dx,  # number of continuous policy variables
                           0,  # number of discretized policy values
                           ni,  # number of discrete states
                           nj,  # number of discrete choices
                           self.random.w.size,  # number of discretized continuous shocks
                           basis.M)  # number of collocation coefficients

        ''' <<<<<<<<<<<<<<<<<<<             END OF CONSTRUCTOR        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    def __repr__(self):
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

    def bounds(self, s, i, j):  # --> (lowerBound, UpperBound)
        """ Returns upper-  and lower-bounds for the continuous action variable.

        Depends only on state variables and discrete action
        """
        ns = s.shape[-1]
        dx, ds = self.dims['dx', 'ds']
        lb, ub = self.__b(s, i, j) #, *self.__par['bounds']) only if params defined
        lb.shape = dx, ns
        ub.shape = dx, ns
        return lb, ub

    def restrictions(self, s, x, i, j, derivative=False):  # --> (h, hx, hxx)
        """ Returns the restriction functions and their first- and second-derivatives.

        Depends only on current variables

        """
        ns = s.shape[-1]
        dx = self.dims.dx

        hh = self.__h(s, x, i, j)


        if isinstance(hh, tuple):
            # assert len(ff) == 3, 'reward must return 1 or 3 arrays'  # commented-out for speed
            h, hx, hxx = hh[0].reshape(-1, ns), hh[1].reshape(-1, dx, ns), hh[2].reshape(-1, dx, dx, ns)
        else:
            h, hx, hxx = hh.reshape(-1, ns), None, None

        return (h, hx, hxx) if derivative else h


    def reward(self, s, x, i, j, derivative=False):  # --> (f, fx, fxx)
        """ Returns the reward function (e.g. utility, profits) and its first- and second-derivatives.

        Depends only on current variables

        """
        ns = s.shape[-1]
        dx = self.dims.dx

        ff = self.__f(s, x, i, j)
        if isinstance(ff, tuple):
            # assert len(ff) == 3, 'reward must return 1 or 3 arrays'  # commented-out for speed
            f, fx, fxx = ff[0].reshape(1, ns), ff[1].reshape(dx, ns), ff[2].reshape(dx, dx, ns)
        else:
            f, fx, fxx = ff.reshape(1, ns), None, None

        return (f, fx, fxx) if derivative else f


    def transition(self, s, x, i, j, in_, e, derivative=False):  # --> (g, gx, gxx)
        """ Returns the next-period continuous state and its first- and second-derivatives.

         Depends on current (s,x,i,j) and future (in,e) variables
        """
        ns = s.shape[-1]
        dx, ds = self.dims['dx', 'ds']

        gg = self.__g(s, x, i, j, in_, e)
        if isinstance(gg, tuple):
            # assert len(ff) == 3, 'reward must return 1 or 3 arrays'  # commented-out for speed
            g, gx, gxx = gg[0].reshape(ds, ns), gg[1].reshape(dx, ds, ns), gg[2].reshape(dx, dx, ds, ns)
        else:
            g, gx, gxx = gg.reshape(ds, ns), None, None

        return (g, gx, gxx) if derivative else g


    def solve(self, v=None, x=None, nr=10, **kwargs):
        """ Solves the model

        Args:
          solver: a DPsolver object (optional)

        Returns:

        """

        # Update solution options using kwargs
        self.options[kwargs.keys()] = kwargs.values()

        t = slice(None) if np.isinf(self.time.horizon) else -1  # if finite horizon, v is taken as last period

        if v is not None:
            self.Value[t] = v[t]

        if x is not None:
            self.Policy_j[t] = x[t]


        ''' 1: PREPARATIONS*********************** '''
        ni, nj, dx = self.dims['ni', 'nj', 'dx']

        if self.options.knownFunctions is None:
            self.options.knownFunctions = np.zeros([ni, nj], bool)

        # Step necessary for discretized models
        if dx == 0:
            self.options.discretized = True
        elif self.options.X is not None:
            self.options.X = np.atleast_2d(self.options.X)
            self.dims.nx = self.options.X.shape[-1]  # number of discretized policy values
            self.options.discretized = True
            if self.options.X.shape[0] != dx:
                raise ValueError('If model is discretized, field "X" must have {} rows'.format(dx))

        # s0 = self.Value.nodes
        # x0 = self.Policy.y[..., 0, :, :]
        # self.options.D_reward_provided = isinstance(self.__f(s0, x0, 0, 0), tuple)
        # self.options.D_transition_provided = isinstance(self.__g(s0, x0, 0, 0, 0, 0), tuple)


        ''' 2: SOLVE THE MODEL******************** '''
        if np.isfinite(self.time.horizon):
            self.__solve_backwards()
        elif self.options.algorithm == 'funcit':
            self.__solve_by_function_iteration()
        elif self.options.algorithm == 'newton':
            self.__solve_by_Newton_method()
        else:
            raise ValueError('Unknown solution algorithm')

        self.update_policy()

        if nr is not None:
            return self.solution(nr)

    def solution(self, nr=10, resid=True):
        """
        Computes solution over a refined grid

        nr:  scalar or np.array
            -- scalar >> compute a grid from basis
            -- array  >> compute solution over provided values
        resid: compute residuals if True

        """
        # TODO:  Make finite horizon case

        ni, nj, ds, dx = self.dims['ni', 'nj','ds', 'dx']


        scalar_input = np.isscalar(nr) and isinstance(nr, int)

        if scalar_input:
            a = self.Value.a
            b = self.Value.b
            n = self.Value.n
            sr = np.atleast_2d(gridmake(*[np.linspace(a[i], b[i], nr * n[i]) for i in range(self.Value.d)]))
        else:
            sr = np.atleast_2d(nr)
            assert sr.shape[0] == self.dims.ds, 'provided s grid must have {} rows'.format(self.dims.ds)

        ''' MAKE DATABASE'''
        # ADD CONTINUOUS STATE VARIABLE
        DATA = pd.DataFrame(np.tile(sr, ni).T, columns=self.labels.s)

        # ADD DISCRETE STATE VARIABLE
        if ni > 1:
            ivals= np.repeat(np.arange(ni), sr.shape[1])
            icat = pd.Categorical.from_codes(ivals, self.labels.i)
            DATA['i'] = ivals

        # SET INDEX FOR DATA
        if ds == 1:
            slab = DATA[self.labels.s[0]]
            if ni > 1:
                DATA.index = pd.MultiIndex.from_arrays([icat, slab])
            else:
                DATA.index = slab
        elif ni > 1:
            DATA.index = icat

        # COMPUTE OPTIMAL POLICY AND VALUE
        xr = self.Policy_j(sr, dropdim=False)  # [0] because there is only 1 order
        vr = self.vmax(sr, xr, self.Value)
        v_LHS = self.Value(sr, dropdim=False) # LHS of Bellman equation: V(s)

        # ADD VALUE FUNCTION
        DATA['value'] = v_LHS.flatten()

        # ADD RESIDUAL IF REQUESTED
        if resid:
            v_RHS = np.max(vr, -2)  # RHS of Bellman equation: max[f(s,x) + dV(s')]
            DATA['resid'] = (v_LHS - v_RHS).flatten()

        # ADD VALUE FUNCTION PER DISCRETE ACTION
        if nj > 1:
            tempj = np.argmax(vr,-2).flatten()
            DATA['j*'] = pd.Categorical.from_codes(tempj,self.labels.j)  # optimal discrete choice

            for j, jlabel in enumerate(self.labels.j):
                DATA['value[' + jlabel + ']'] = vr[:, j].flatten()

        # ADD CONTINUOUS ACTION
        if dx:
            for ix, xlabel in enumerate(self.labels.x):
                DATA[xlabel] = self.Policy(sr, dropdim=False)[:,ix].flatten()

                # ADD CONTINUOUS ACTION PER DISCRETE ACTION
                if nj > 1:
                    for j, jlabel in enumerate(self.labels.j):
                        DATA[xlabel + '[' + jlabel + ']'] = xr[:, j, ix].flatten()


        return DATA


    def simulate(self, nper, sinit, iinit=0, seed=None):

        # Simulate the model
        #
        #   S = self.simulate(nper, sinit, iinit)
        #
        # nper = number of periods to simulate (scalar)
        # sinit = initial continuos state (nrep x ds), where nrep is number of repetitions
        # iinit = initial discrete state (scalar)
        #
        # S = simulation results (table), with variables:
        #    r, repetion number
        #    t, time period
        #    i, discrete state
        #    j, optimal discrete action
        #    s, continuous state
        #    x, optimal continuous action

        # ****** 1: Preparation***********************************************************
        ds, dx, ni, nj, ne = self.dims['ds', 'dx', 'ni', 'nj', 'ne']

        assert iinit < ni, 'Invalid initial discrete state: must be an integer between 0 and {}'.format(ni - 1)
        if seed:
            np.random.seed(seed)

        # Determine number of replications nrep and periods nper to be simulated.
        # nper cannot exceed time.horizon.
        sinit = np.atleast_2d(sinit).astype(float)
        ds2, nrep = sinit.shape
        assert ds==ds2, 'initial continous state must have {} rows'.format(ds)
        nper = min(nper, self.time.horizon) + 1

        ### Allocate memory to output arrays
        ssim = np.empty((nper + 1, ds, nrep))
        xsim = np.empty((nper, dx, nrep))
        isim = np.empty((nper + 1, nrep), dtype=int)
        jsim = np.empty((nper, nrep), dtype=int)

        ### Set initial states
        ss = sinit
        ii = np.full(nrep, iinit)
        ssim[0] = ss
        isim[0] = ii

        # ***** *2: Simulate the model ***************************************************
        #
        # For each period
        for ip in range(nper):
            # # *TODO* Write the finite horizon case
            # #{
            # if T<inf
            #     for i=1:ni
            #         for j=1:nj
            #             cx[:,:,i,j] = Phi\x[:,:,i,j,ip]
            #             cv[:,i,j]   = Phi\v[:,i,j,ip]
            #         end
            #     end
            # end
            #}

            ### Allocate memory for current-period policy/value functions, all repetitions
            xx = np.empty_like(xsim[0])
            jj = np.empty_like(jsim[0])
            vv = np.empty((nj, nrep))

            ### For the current discrete state ii, compute the conditional value functions use this to determine the
            # optimal discrete policy jmax
            for i in range(ni):
                ir = ii == i
                if not np.any(ir):
                    continue
                vv[:, ir] = self.Value_j[i](ss[:, ir], dropdim=False)
            jmax = np.argmax(vv, 0)

            ### For current discrete state/policy,  interpolate the optimal continuous policy, keeping it within bounds
            #  record discrete policy
            for i in range(ni):
                for j in range(nj):
                    ir = (ii == i) & (jmax == j)
                    if not np.any(ir):
                        continue
                    if dx > 0:
                        xx[:, ir] = self.Policy_j[i, j, :](ss[:, ir])
                        xl, xu = self.bounds(ss[:, ir], i, j)
                        xx[:, ir] = np.minimum(np.maximum(xx[:, ir], xl), xu)

                    jj[ir] = j

            ### Compute the new continuous shock (necessary for the transition function)
            if self.random.w.size == 1:
                ee = np.tile(self.random.e, nrep)     #self.random.e[ones(nrep,1),:]  #fixme make sure dimensions are ok
            else:
                ee = self.random.e[:, np.random.choice(ne, nrep, p=self.random.w)] #fixme make sure dimensions are ok



            ### Compute the new state: use the Markov transition matrix self.random.q to update
            # the discrete state then update the continuous state
            iiold = ii[:]
            ii = np.zeros(nrep)
            for i in range(ni):
                for j in range(nj):
                    ir = (iiold == i) & (jj == j)
                    if not np.any(ir):
                        continue
                    if ni > 1:
                        ii[ir] =  np.random.choice(ni, ir.sum(), p=self.random.q[j, i, :])
                        # ii[ir] =  discrand(ir.sum(), self.random.q[j, i, :])
                    ss[:, ir] = self.transition(ss[:, ir], xx[:, ir], i, j, ii[ir], ee[:, ir])

            ### Save the current-period simulation
            ssim[ip + 1] = ss
            isim[ip + 1] = ii
            xsim[ip] = xx
            jsim[ip] = jj


        ### Trim the last observation if model has infinite horizon
        if np.isinf(self.time.horizon):
            ssim = ssim[:nper]
            isim = isim[:nper]

        # ****** 3: Make a table with the simulated data *********************************
        #
        # # Permute (transpose) the dimensions of the data, so first index refer to time.
        # ssim = permute(ssim,[2,1,3])
        # xsim = permute(xsim,[2,1,3])
        # isim = isim'
        # jsim = jsim'

        ### Add variables rsim and tsim to identify the repetition number and the time
        # period of each observation
        tsim, rsim = gridmake(np.arange(nper), np.arange(nrep))

        ### Make the table.
        DATA = pd.DataFrame()
        DATA['time'] = tsim
        if nrep > 1:
            DATA['_rep'] = rsim

        if ni > 1:
            DATA['i'] = pd.Categorical.from_codes(isim.flatten(),self.labels.i)

        sdata = ssim.swapaxes(0, 1).reshape((ds, -1))
        for k, slab in enumerate(self.labels.s):
            DATA[slab] = sdata[k]

        if nj > 1:
            DATA['j*'] = pd.Categorical.from_codes(jsim.flatten(), self.labels.j)

        if dx > 0:
            xdata = xsim.swapaxes(0, 1).reshape((dx, -1))
            for k, xlab in enumerate(self.labels.x):
                DATA[xlab] = xdata[k]

        return DATA


    def lqapprox(self, s0, x0):
        """
        Solves discrete time continuous state/action dynamic programming model using a linear quadratic approximation
        Args:
            s0: steady-state state
            x0:  steady-state action

        Returns:
            A LQmodel object
        """


        assert (self.dims.ni * self.dims.nj < 2), 'Linear-Quadratic not implemented for models with discrete state or choice'
        s0, x0 = np.atleast_1d(s0, x0)

        assert s0.size == self.dims.ds, 's0 must have %d values' % self.dims.ds
        assert x0.size == self.dims.dx, 'x0 must have %d values' % self.dims.dx

        s0, x0 = s0.astype(float), x0.astype(float)
        s0.shape = -1, 1
        x0.shape = -1, 1

        delta = self.time.discount

        # Fix shock at mean
        estar = self.random.w @ self.random.e.T
        estar.shape = -1, 1

        # Get derivatives
        f0, fx, fxx = self.reward(s0, x0, None, None, True)
        g0, gx, gxx = self.transition(s0, x0, None, None, None, estar, derivative=True)

        fs = jacobian(lambda y: self.reward(y.reshape((-1, 1)), x0, None, None), s0)
        fxs = jacobian(lambda y: self.reward(y.reshape((-1, 1)), x0, None, None, True)[1], s0)
        fss = hessian(lambda y: self.reward(y.reshape((-1, 1)), x0, None, None), s0)
        gs = jacobian(lambda y: self.transition(y.reshape((-1, 1)), x0, None, None, None, estar), s0)

        # Reshape to ensure conformability
        ds, dx = self.dims['ds', 'dx']

        f0.shape = 1, 1
        s0.shape = ds, 1
        x0.shape = dx, 1
        fs.shape = 1, ds
        fx.shape = 1, dx
        fss.shape = ds, ds
        fxs.shape = dx, ds
        fxx.shape = dx, dx
        g0.shape = ds, 1
        gx.shape = ds, dx
        gs.shape = ds, ds
        fsx = fxs.T

        f0 += - fs @ s0 - fx @ x0 + 0.5 * s0.T @ fss @ s0 + s0.T @ fsx @ x0 + 0.5 * x0.T @ fxx @ x0
        fs += - s0.T @ fss - x0.T @ fxs
        fx += - s0.T @ fsx - x0.T @ fxx
        g0 += - gs @ s0 - gx @ x0

        return LQmodel(f0, fs, fx, fss, fsx, fxx, g0, gs, gx, delta,self.labels.s, self.labels.x)


    def __solve_backwards(self):
        """
        Solve collocation equations for finite horizon model by backward recursion
        """
        T = self.time.horizon
        s = self.Value.nodes

        tic = time.time()
        self.options.print_header('backward recursion', T)
        for t in reversed(range(T)):
            self.options.print_current_iteration(t, 0, tic)
            self.Value_j[t] = self.vmax(s,
                                        self.Policy_j.y[t],
                                        self.Value[t + 1])
            self.make_discrete_choice(t)

        self.options.print_last_iteration(tic, 0)
        return None

    def __solve_by_function_iteration(self):
        """
            Solves infinite-horizon model collocation equation by function iteration. Solution is found when the
            collocation coefficients of the value function converge to a fixed point (within |self.tol| tolerance).
         """
        tic = time.time()
        s = self.Value.nodes
        self.options.print_header('function iteration', self.time.horizon)
        for it in range(self.options.maxit):
            cold = self.Value.c.copy()
            self.Value_j[:] = self.vmax(s, self.Policy_j.y, self.Value)
            self.make_discrete_choice()
            change = np.linalg.norm((self.Value.c - cold).flatten(), np.Inf)
            self.options.print_current_iteration(it, change, tic)
            if change < self.options.tol:
                break
            if np.isnan(change):
                raise ValueError('nan found on function iteration')
        self.options.print_last_iteration(tic, change)

    def __solve_by_Newton_method(self):
        tic = time.time()
        s = self.Value_j.nodes
        x = self.Policy_j.y
        ni = self.dims.ni

        if issparse(self.Value_j._Phi):
            Phik = kron(identity(ni), self.Value_j._Phi, format='csr').toarray()
            # Solve = spsolve
        else:
            Phik = np.kron(np.eye(self.dims.ni), self.Value._Phi)
            # Solve = np.linalg.solve

        # todo: fix the dimensions and check that Phik is transposed?

        def SOLVE(A,b):
            return np.linalg.solve(A, b) if (self.dims.ns == self.dims.nc) else np.linalg.lstsq(A, b)[0]


        self.options.print_header("Newton's", self.time.horizon)
        for it in range(self.options.maxit):
            cold = self.Value.c.copy().flatten()
            # print('\ncold', cold)
            self.Value_j[:], vc = self.vmax(s, x, self.Value, True)
            self.make_discrete_choice()
            step = - SOLVE(Phik - vc, Phik @ cold - self.Value.y.flatten())
            c = cold + step
            change = np.linalg.norm(step, np.Inf)
            self.Value.c = c.reshape(self.Value.c.shape)
            self.options.print_current_iteration(it, change, tic)
            if np.isnan(change):
                raise ValueError('nan found on Newton iteration')
            if change < self.options.tol:
                break
        self.options.print_last_iteration(tic, change)




    def vmax(self, s, x, Value, dVc=False):  # [v,x,vc]
        # Unpack model structure
        ni, nj = self.dims['ni', 'nj']
        ns = s.shape[-1]
        ms = Value.M  # number of polynomials
        v = np.empty([ni, nj, ns])

        if self.dims.dx == 0:  # Discrete model
            # hh = slice(None)
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.__Bellman_rhs_discrete(Value, None, s, i, j)
        elif self.options.discretized:
            # hh = slice(None)
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_discretized(Value, s, x[i, j], i, j)
        elif self.__h is None:
            # hh = 0
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_continuous(Value, s, x[i, j], i, j)
        else:
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_continuous_restricted(Value, s, x[i, j], i, j)


        if not dVc:
            return v

        # Computes derivative with respect to Value function interpolation coefficients
        e, w, q = self.random['e', 'w', 'q']


        if ni * nj > 1:
            vc = np.zeros((ns, ni, ms, ni))
            jmax = np.argmax(v, 1)

            for i in range(ni):
                for j in range(nj):
                    is_ = jmax[i] == j
                    if not np.any(is_):
                        continue

                    for k in range(w.size):
                        ee = np.tile(e[:, [k]], ns)  # indexing with [k] instead of k retains shape of vector!
                        for in_ in range(ni):
                            if q[j, i, in_] > 0:
                                snext = self.transition(s[:, is_], x[i, j, :, is_], i , j, in_, ee[:, is_])  #fixme need to know number of output arguments!!!
                                prob = w[k] * q[j, i, in_,]
                                vc[is_, i, :, in_] += prob * Value.Phi(snext).toarray().reshape((is_.sum(), ms), order='F')   #fixme I can't find the proper way to index this

            vc = vc.reshape((ns*ni,ms*ni),order='F')
        else:
            vc = np.zeros((ns, ms))
            for k in range(w.size):
                ee = np.tile(e[:, [k]], ns)
                snext = self.transition(s, x[0, 0], 0, 0, 0, ee) #fixme need to know number of output arguments!!!
                vc += w[k] * Value.Phi(snext)

        vc *= self.time.discount
        return v, vc







    # vmax_discretized
    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, when the continuous policy has been discretized.

    def vmax_discretized(self, Value, s, xij, i, j):

        nx = self.dims.nx
        ns = s.shape[-1]
        dx = self.dims.dx
        X = self.options.X

        vv = np.full((nx, ns), -np.inf)

        xl, xu = self.bounds(s, i, j)
        xl = xl.T
        xu = xu.T

        for h, x0 in enumerate(X.T):
            is_= np.all((xl <= x0) & (x0 <= xu), 1)
            if np.any(is_):
                xx = np.repeat(x0, ns, 0)
                vv[h, is_] = self.__Bellman_rhs_discrete(Value, xx, s, i, j)

        xmax = np.argmax(vv, 0)

        vxs = [a[0] for a in np.indices(vv.shape)]  # the [0] reduces one dimension
        vxs[0] = xmax
        vij = vv[vxs]

        xxs = [a[0] for a in np.indices(X.T.shape)]
        xxs[0] = xmax
        xij[:] = X.T[xxs]

        return vij




    # vmax_continuous
    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, by solving the linear complementarity problem.
    def vmax_continuous_MCP(self, Value, s, xij, i, j):

        ns = s.shape[-1]
        dx = self.dims.dx

        def kkt(xvec):
            """ Karush-Kuhn Tucker conditions

            The first-order conditions are given by the derivative of Bellman equation. The problem needs to be
            solved ns times (number of nodes), each time with dx unknowns (policy variables). This routine
            expresses all this FOCs as a single vector, making a block-diagonal matrix with the respective Hessian matrices
            (= jacobian of FOCs). The resulting output is suitable to be solved by the MCP class.

            """
            xij = xvec.reshape((dx, ns))
            EV, EVx, EVxx = self.__Bellman_rhs(Value, s, xij, i, j)

            # and let the first index indicate node
            Vx = np.swapaxes(EVx, 0, -1)
            Vxx = np.swapaxes(EVxx, 0, -1)
            return Vx.flatten(), block_diag(Vxx, 'csc')#.toarray()  #todo not so sure I want a full array, but a lot of trouble with sparse

        xl, xu = self.bounds(s, i, j)

        xlv, xuv, xijv = map(lambda z: z.flatten('F'), (xl, xu, xij))  # vectorize

        FOCs = MCP(kkt, xlv, xuv, xijv)
        xij[:] = FOCs.zero(show=False, transform=self.options.ncpmethod).reshape((ns, dx)).T

        return self.__Bellman_rhs(Value, s, xij, i, j)[0][0]

    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, by solving the linear complementarity problem.
    def vmax_continuous(self, Value, s, xij, i, j):
        xl, xu = self.bounds(s, i, j)

        for it in range(self.options.maxitncp):
            vv, vx, vxx = self.__Bellman_rhs(Value, s, xij, i, j)

            # Compute Newton step, update continuous action, check convergence
            vx, delx = lcpstep(self.options.ncpmethod, xij, xl, xu, vx, vxx)
            xij[:] += delx
            if np.linalg.norm(vx.flatten(), np.Inf) < self.options.tol:
                break

        return self.__Bellman_rhs(Value, s, xij, i, j)[0][0]


    def vmax_continuous_restricted(self, Value, s, xij, i, j):

        ns = s.shape[-1]
        dx = self.dims.dx
        xl, xu = self.bounds(s, i, j)

        dr = self.restrictions(s, xij, i, j).shape[0]
        lij = np.ones((dr, ns))
        xl = np.vstack((xl, np.zeros((dr, ns))))
        xu = np.vstack((xu, np.tile(np.inf, (dr, ns))))
        ZEROS = np.zeros((dr, dr, ns))


        for it in range(self.options.maxitncp):
            vv, vx, vxx = self.__Bellman_rhs(Value, s, xij, i, j)

            # Adjust derivative to add the Lagrange multiplier term
            h, hx, hxx = self.restrictions(s, xij, i, j, True)

            for ir in range(dr):
                vx += lij[ir] * hx[ir]
                vxx += lij[ir] * hxx[ir]

            vx = np.vstack((vx, -h))

            vxx = np.vstack((np.hstack((vxx, np.swapaxes(hx, 0,1))),
                             np.hstack((-hx, ZEROS))))


            #  Compute Newton step, update continuous action, check convergence
            vx, delxl = lcpstep(self.options.ncpmethod,
                               np.vstack((xij, lij)),
                               xl, xu, vx, vxx)
            delx, dell = np.split(delxl, [dx])
            xij[:] += delx
            lij[:] += dell

            print('it = ', it, '\tchange = ', np.linalg.norm(vx.flatten(), np.Inf))
            if np.linalg.norm(vx.flatten(), np.Inf) < self.options.tol:
                break


        return self.__Bellman_rhs(Value, s, xij, i, j)[0][0]





    def __Bellman_rhs_discrete(self, Value, xij, s, i, j):
        ni, nj = self.dims['ni', 'nj']
        ns = s.shape[-1]
        e, w, q = self.random['e', 'w', 'q']
        vv = self.reward(s, xij, i, j)

        for k in range(w.size):
            # Compute states next period
            ee = np.tile(e[:, k: k + 1], ns)
            for in_ in range(ni):
                if q[j, i, in_] == 0:
                    continue
                snext = np.real(self.transition(s, xij, i, j, in_, ee))
                prob_delta = self.time.discount * w[k] * q[j, i, in_]
                vv += prob_delta * Value[in_](snext)
        return vv



    def __Bellman_rhs(self, Value, s, xij, i, j):
        ni, nj = self.dims['ni', 'nj']  # dimensions
        ns = s.shape[-1]
        e, w, q = self.random['e', 'w', 'q']                # randomness

        vv, vx, vxx = self.reward(s, xij, i, j, True)

        for k in range(w.size):
            # Compute states next period and derivatives
            ee = np.tile(e[:, [k]], ns)
            for in_ in range(ni):
                if q[j, i, in_] == 0:
                    continue

                snext, snx, snxx = self.transition(s, xij, i, j, in_, ee, derivative=True)
                snext = np.real(snext)
                prob_delta = self.time.discount * w[k] * q[j, i, in_]

                vn, vns, vnss = Value[in_](snext, order='all', dropdim=False)  # evaluates function, jacobian, and hessian if order='all'

                # Drop the unnecessary dimension
                vns = vns[:, 0]
                vnss = vnss[:, :, 0]


                vv += prob_delta * vn
                vx += prob_delta * np.einsum('k...,jk...->j...', vns, snx)
                vxx += prob_delta * np.einsum('hi...,ij...,kj...->hk...', snx, vnss, snx) +  \
                      np.einsum('k...,ijk...->ij...', vns, snxx)

        return vv, vx, vxx


    def make_discrete_choice(self, t=None):
        # notice : Value_j.y  dims are: 0=state, 1=action, 2=node

        if self.dims.nj == 1:
            if t is None:
                self.Value[:] = self.Value_j.y[:, 0]
            else:
                self.Value[t] = self.Value_j.y[t]
            return

        if t is None:
            self.DiscreteAction = np.argmax(self.Value_j.y, 1)
            ijs = [a[:, 0] for a in np.indices(self.Value_j.y.shape)]
            ijs[1] = self.DiscreteAction
            self.Value[:] = self.Value_j.y[ijs]


        else:
            self.DiscreteAction[t] = np.argmax(self.Value_j.y[t], 1)
            ijs = [a[:, 0] for a in np.indices(self.Value_j.y[t].shape)]
            ijs[1] = self.DiscreteAction[t]
            self.Value[t] = self.Value_j.y[t][ijs]

    def update_policy(self):
        if self.dims.nj == 1:
            self.Policy[:] = self.Policy_j.y[:, 0]
        else:
            ijxs = [a[:, 0] for a in np.indices(self.Policy_j.y.shape)]
            ijxs[-3] = self.DiscreteAction[:, np.newaxis, :]
            self.Policy[:] = self.Policy_j.y[ijxs]

    def check_derivatives(self):
        ni, nj, ds, ns = self.dims['ni', 'nj', 'ds', 'ns']
        nij = ni * nj
        idx = pd.MultiIndex.from_product([np.arange(ni), np.arange(nj)], names=['i', 'j'])
        F = pd.DataFrame({
            'fx': np.zeros(nij) + np.nan,
            'fxx': np.zeros(nij) + np.nan},
            index=idx)

        idy = pd.MultiIndex.from_product([['gx', 'gxx'], np.arange(ds)], names=['func', 's'])
        G = pd.DataFrame(np.zeros((nij,2 * ds)), index=idx, columns=idy)

        s = self.Value.nodes
        e = np.tile(self.random.e[:, 0], ns)

        for k, (i, j) in enumerate(indices(ni,nj).T):
            xl, xu = self.bounds(s, i , j)
            xlinf = np.isinf(xl)
            xuinf = np.isinf(xu)
            xm = (xl + xu) / 2
            xm[xlinf] = xu[xlinf]
            xm[xuinf] = xl[xuinf]
            xm[xuinf & xlinf] = 0.0

            f, fx, fxx = self.reward(s,xm,i,j, derivative=True)
            fxa = jacobian(lambda z: self.reward(s, z, i, j), xm)
            fxxa = hessian(lambda z: self.reward(s, z, i, j), xm)

            #FIXME: HESSIAN SEEMS TO BE THROWING WRONG DIMENSIONS!!

            F.values[k, 0] = np.linalg.norm(fx - fxa, np.inf)
            F.values[k, 1] = np.linalg.norm(fxx - fxxa, np.inf)




    def __get_derivatives(self):
        pass

# TODO: design this class:
"""
    * Should finite-infinite horizon models be split into two subclasses?
    * Should discretized models be handled by a subclass?
    * Should vmax operate directly on Value_j and Policy_j? how to deal with residuals?
"""


