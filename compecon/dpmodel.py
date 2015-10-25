import time

from compecon.tools import Options_Container, qzordered
from .nonlinear import MCP
import numpy as np
from numpy.linalg import multi_dot as dot
from scipy.sparse import block_diag
from .tools import jacobian, hessian, gridmake, ix
#from .lcpstep import lcpstep  # todo: is it worth to add lcpstep?


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
                for i in range(ni):
                    for j in range(nj):
                        qq[j, i, h[j, i]] = 1
            else:
                q = np.array(q)
                if q.ndim == 2:  # assume it is the same regardless of choice
                    qq[:] = q
                ss = qq.shape
                assert ss[1] == ss[2], 'A Markov transition matrix must be square (last two dimensions of q must be equal)'
                assert np.allclose(qq.sum(-1), 1), 'The rows of a Markov transition must add up to 1.0'

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
    """
    description = "Dimensions of a DPmodel object"

    def __init__(self, ds=1, ns=None,
                 dx=1, nx=None,
                 ni=1, nj=1,
                 nc=None):
        self.ds = ds
        self.ns = ns
        self.dx = dx
        self.nx = nx
        self.ni = ni
        self.nj = nj
        self.nc = nc


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
                 D_reward_provided=True, D_transition_provided=True, knownFunctions=None, print=True):
        self.algorithm = algorithm
        self.tol = tol
        self.ncpmethod = ncpmethod
        self.maxit = maxit
        self.maxitncp = maxitncp
        self.print = print
        self.discretized = discretized
        self.X = X
        self.D_reward_provided = D_reward_provided
        self.D_transition_provided = D_transition_provided
        self.knownFunctions = knownFunctions

    def print_header(self, method, horizon):
        horizon = 'infinite' if np.isinf(horizon) else 'finite'
        if self.print:
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
        if self.print:
            print('{:4d}  {:12.1e}  {:8.4f}'.format(it, change, time.time() - tic))

    def print_last_iteration(self, tic, change):
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

    def __init__(self, basis, i=('State 0',), j=('Choice 0', ), x=(),
                 discount=0.0, horizon=np.inf,
                 e=None, w=None, q=None, h=None):

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


        # Default numerical solution parameters
        self.options = DPoptions()

        # Model dimensions
        self.dims = DPdims(basis.d,  # number of continuous state variables
                           basis.N,  # number of continuous state nodes
                           dx,  # number of continuous policy variables
                           0,  # number of discretized policy values
                           ni,  # number of discrete states
                           nj,  # number of discrete choices
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


    ''' Dummy auxilliary functions
                To be overwritten by actual model '''

    def bounds(self, s, i, j):  # --> (lowerBound, UpperBound)
        """ Returns upper-  and lower-bounds for the continuous action variable.

        Depends only on state variables and discrete action
        """
        ns = s.shape[-1]
        dx, ds = self.dims['dx', 'ds']
        ub = np.zeros([dx, ns])
        lb = np.zeros([dx, ns])
        return lb, ub

    def reward(self, s, x, i, j):  # --> (f, fx, fxx)
        """ Returns the reward function (e.g. utility, profits) and its first- and second-derivatives.

        Depends only on current variables
        """
        ns = s.shape[-1]
        dx = self.dims.dx
        f = np.zeros([ns])
        fx = np.zeros([dx, ns])
        fxx = np.zeros([dx, dx, ns])
        return f, fx, fxx

    def transition(self, s, x, i, j, in_, e):  # --> (g, gx, gxx)
        """ Returns the next-period continuous state and its first- and second-derivatives.

         Depends on current (s,x,i,j) and future (in,e) variables
        """
        ns = s.shape[-1]
        dx, ds = self.dims['dx', 'ds']
        g = np.zeros([ds, ns])
        gx = np.zeros([dx, ds, ns])
        gxx = np.zeros([dx, dx, ds, ns])
        return g, gx, gxx

    def solve(self, v=None, x=None, solver=None):
        """ Solves the model

        Args:
          solver: a DPsolver object (optional)

        Returns:

        """

        t = slice(None) if np.isinf(self.time.horizon) else -1  # if finite horizon, v is taken as last period

        if v is not None:
            self.Value[t] = v[t]

        if x is not None:
            self.Policy_j[t] = x[t]



        ''' 1: PREPARATIONS*********************** '''
        ni, nj, dx = self.dims['ni', 'nj', 'dx']

        if type(solver) is DPoptions:
            self.options = solver

        if self.options.knownFunctions is None:

            self.options.knownFunctions = np.zeros([ni, nj], bool)

        # Step necessary for discretized models
        if dx == 0:
            self.options.discretized = True
        elif self.options.X is not None:
            self.dims.nx = self.options.X.shape[-1]  # number of discretized policy values
            self.options.discretized = True
            if self.options.X.shape[0] != dx:
                raise ValueError('If model is discretized, field "X" must have {} columns'.format(dx))


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
            change = np.linalg.norm((self.Value_j.c - cold).flatten(), np.Inf)
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

        self.options.print_header("Newton's", self.time.horizon)
        # todo: fix the dimensions and check that Phik is transposed?
        Phik = np.kron(np.eye(self.dims.ni), self.Value_j._Phi)
        for it in range(self.options.maxit):
            cold = self.Value.c.copy().flatten()
            # print('\ncold', cold)
            self.Value_j[:] = self.vmax(s, x, self.Value)
            vc = self.vmax_derivative(self.Value, s, x)
            self.make_discrete_choice()
            step = - np.linalg.lstsq(Phik - vc, Phik.dot(cold) - self.Value.y.flatten())[0]
            c = cold + step
            change = np.linalg.norm(step, np.Inf)
            self.Value.c = c.reshape(self.Value.c.shape)
            self.options.print_current_iteration(it, change, tic)
            if np.isnan(change):
                raise ValueError('nan found on Newton iteration')
            if change < self.options.tol:
                break
        self.options.print_last_iteration(tic, change)

    def residuals(self, nr=10):
        """
        Computes residuals over a refined grid

        If nr is scalar, compute a grid. Otherwise compute residuals over provided nr (sr)

        """
        scalar_input = np.isscalar(nr) and isinstance(nr, int)

        if scalar_input:
            a = self.Value.a
            b = self.Value.b
            n = self.Value.n
            sr = gridmake([np.linspace(a[i], b[i], nr * n[i]) for i in range(self.Value.d)])
        else:
            sr = np.atleast_2d(nr)
            assert sr.shape[0] == self.dims.ds, 'provided s grid must have {} rows'.format(self.dims.ds)

        xr = self.Policy_j(sr, dropdim=False)
        vr = self.vmax(sr, xr, self.Value)
        resid = self.Value(sr, dropdim=False) - np.max(vr, -2)

        # eliminate singleton dimensions and return
        if scalar_input:
            return np.squeeze(resid), sr, np.squeeze(vr), np.squeeze(xr)
        else:
            return np.squeeze(resid)


    def getDerivative(self, func, s, x, *args, **kwargs):
        dx, nx = x.shape

        if func == 'reward':
            def F(X):
                return self.reward(s, X, *args, **kwargs)
            df = 1
            f = F(x)
            # assert(f.ndim == 1, 'reward must return a numpy vector')
        elif func == 'transition':
            def F(X):
                return self.transition(s, X, *args, **kwargs)
            df = self.dims.ds
            f = F(x)
            # assert(f.shape[0] == df, 'transition must have {}rows'.format(df))
        else:
            raise ValueError('unknown function')

        ''' Compute Jacobian'''
        tol = np.spacing(1) ** (1 / 3)

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

            # fx[k] = (fplus - fminus) / np.tile(deltaX[k],(df, 1))
            fx[k] = (fplus - fminus) / deltaX[k]

        ''' Compute Hessian'''
        tol = np.spacing(1) ** (1 / 4)

        h = tol * np.maximum(abs(x), 1)
        x_minus_h = x - h
        x_plus_h = x + h
        deltaX = h  # repmat(h, 1, 1, dx)

        # deltaXX = deltaX .* permute(deltaX,[1,3,2])

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

    def vmax(self, s, x, Value):  # [v,x,vc]
        # Unpack model structure
        ni, nj = self.dims['ni', 'nj']
        ns = s.shape[-1]
        v = np.empty([ni, nj, ns])


        if self.options.discretized:
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_discretized(Value, s, x[i, j], i, j)
        else:
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_continuous(Value, s, x[i, j], i, j)



        # Compute derivative of Bellman equation optimand with respect to basis coefficients for Newton method

        return v
    # vmax_discretized
    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, when the continuous policy has been discretized.
    def vmax_discretized(self, Value, s, x, i, j):
        # function vij = vmax_discretized(i,j)
        #     nx = size(X,1)  # number of discretized continuous actions
        #     vv = zeros(ns,nx)
        #     if nx>1
        #         [xl,xu] = self.bounds(s,i,j)
        #
        #     for ix=1:nx
        #         vv[:,ix] = -inf
        #         xx = X[ix+zeros(ns,1),:]
        #         if nx>1
        #             is = find(all(xx>=xl,2).* all(xx<=xu,2)==1)
        #         else
        #             is = 1:ns
        #
        #         if ~isempty(is)
        #             # Initialize optimand and derivatives with reward terms
        #             vv[is,ix] = self.reward(s[is,:],xx[is,:],i,j)
        #             for k=1:length(w)
        #                 # Compute states next period and derivatives
        #                 ee = e[k+zeros(length(is),1),:]
        #                 for in=1:ni
        #                     if q[i,in,j]==0, continue,
        #                     snext = self.transition(s[is,:],xx[is,:],i,j,in,ee)
        #                     snext = real(snext)
        #                     prob = w[k]*q[i,in,j]
        #
        #                     vn = self.Value[in](snext)    # vn = funeval(c(:,in),basis,snext)
        #                     vv[is,ix] = vv[is,ix] + delta*prob*vn
        #
        #     [vij,ixmax] = max(vv,[],2)
        #     x[:,:,i,j] = X[ixmax,:]
        raise NotImplementedError

    # vmax_continuous
    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, by solving the linear complementarity problem.
    def vmax_continuous(self, Value, s, xij, i, j):
        ns = s.shape[-1]
        dx = self.dims.dx
        xl, xu = self.bounds(s, i, j)

        xlv, xuv, xijv = map(lambda z: z.flatten('F'), (xl, xu, xij))  # vectorize

        KK2 = MCP(self.kkt, xlv, xuv, xijv, Value, s, i, j)
        xij[:] = KK2.zero(print=False, transform=self.options.ncpmethod).reshape((ns, dx)).T

        return self.__Bellman_rhs(Value, s, xij, i, j)[0][0]

    def vmax_derivative(self, Value, s, x):
        ## vmax_derivative
        # Nested function in vmax:  Computes derivative with respect to Value function interpolation coefficients
        ns = self.dims.ns
        ns, ni, nj = self.dims['ns', 'ni', 'nj']
        e, w, q = self.random['e', 'w', 'q']

        if ni * nj > 1:
            vc = np.zeros((ni, ns, ni, ns))
            jmax = self.DiscreteAction
            for i_ in range(ni):
                for j_ in range(nj):
                    is_ = jmax[i_] == j_
                    if not np.any(is_):
                        continue

                    for k in range(w.size):
                        ee = np.tile(e[:, k : k+1], ns)
                        for in_ in range(ni):
                            if q[j_, i_, in_] > 0:
                                snext = self.transition(s[:, is_], x[i_, j_, :, is_], i_ , j_, in_, ee[:, is_])[0]
                                prob = w[k] * q[j_, i_, in_,]
                                vc[i_, is_, in_] += prob * np.reshape(Value.Phi(snext), (is_.size, 1, ns))

            vc.shape = (ni * ns, ni * ns)
        else:
            vc = np.zeros((ns, ns))
            for k in range(w.size):
                ee = np.tile(e[:, k: k + 1], ns)
                snext = self.transition(s, x[0, 0], 0, 0, 0, ee)[0]
                vc += w[k] * Value.Phi(snext)

        vc *= self.time.discount
        return vc


    def __Bellman_rhs(self, Value, s, xij, i, j):
        ds, dx, ni, nj = self.dims['ds', 'dx', 'ni', 'nj']  # dimensions
        ns = s.size
        e, w, q = self.random['e', 'w', 'q']                # randomness

        if self.options.D_reward_provided:
            vv, vx, vxx = self.reward(s, xij, i, j)
        else:
            vv, vx, vxx = self.getDerivative('reward', s, xij, i, j)

        vv = np.atleast_2d(vv)  # todo do I really need this?
        vx = vx[:, np.newaxis]
        vxx = vxx[:, :, np.newaxis]

        for k in range(w.size):
            # Compute states next period and derivatives
            ee = np.tile(e[:, k: k + 1], ns)
            for in_ in range(ni):
                if q[j, i, in_] == 0:
                    continue

                if self.options.D_transition_provided:
                    snext, snx, snxx = self.transition(s, xij, i, j, in_, ee)
                else:
                    snext, snx, snxx = self.getDerivative('transition', s, xij, i, j, in_, ee)

                snext = np.real(snext)
                prob_delta = self.time.discount * w[k] * q[j, i, in_]

                vn, vns, vnss = Value[in_](snext, order='all', dropdim=False)  # evaluates function, jacobian, and hessian if order='all'


                vv += prob_delta * vn
                vx += prob_delta * np.einsum('k...,jk...->j...', vns, snx)

                vxx += prob_delta * np.einsum('hi...,ij...,kj...->hk...', snx, vnss, snx) +  \
                      np.einsum('k...,ijk...->ij...', vns, snxx)

            return vv, vx, vxx

    def kkt(self, xvec, Value, s, i, j):
        """ Karush-Kuhn Tucker conditions

        The first-order conditions are given by the derivative of Bellman equation. The problem needs to be
        solved ns times (number of nodes), each time with dx unknowns (policy variables). This routine
        expresses all this FOCs as a single vector, making a block-diagonal matrix with the respective Hessian matrices
        (= jacobian of FOCs). The resulting output is suitable to be solved by the MCP class.

        """
        xij = xvec.reshape((self.dims.dx, s.shape[-1]))
        EV, EVx, EVxx = self.__Bellman_rhs(Value, s, xij, i, j)

        EVx = EVx[:, 0]
        EVxx = EVxx[:, :, 0]

        # and let the first index indicate node
        Vx = np.swapaxes(EVx, 0, -1)
        Vxx = np.swapaxes(EVxx, 0, -1)
        return Vx.flatten(), block_diag(Vxx, 'csc').toarray()  #todo not so sure I want a full array, but a lot of trouble with sparse

    def make_discrete_choice(self, t=None):
        # notice : Value_j.y  dims are: 0=state, 1=action, 2=node

        if self.dims.nj == 1:
            if t is None:
                self.Value[:] = self.Value_j.y[:]
            else:
                self.Value[t] = self.Value_j.y[t]
            return

        if t is None:
            self.DiscreteAction = np.argmax(self.Value_j.y, 1)
            ijs = ix(self.Value_j.y)
            ijs[-2] = self.DiscreteAction
            self.Value[:] = self.Value_j.y[ijs]
            # self.Value[:] = np.max(self.Value_j.y, 1)
        else:
            self.DiscreteAction[t] = np.argmax(self.Value_j.y[t], 1)
            ijs = ix(self.Value_j.y[t])
            ijs[-2] = self.DiscreteAction[t]
            self.Value[t] = self.Value_j.y[t][ijs]
            # self.Value[t] = np.max(self.Value_j.y[t], 1)

    def update_policy(self):
        if self.dims.nj == 1:
            self.Policy[:] = self.Policy_j.y[:]
        else:
            ijxs = ix(self.Policy_j.y)
            ijxs[-3] = self.DiscreteAction[:, np.newaxis, :]
            self.Policy[:] = self.Policy_j.y[ijxs]



    def lqapprox(self, s0, x0):

        assert (self.dims.ni * self.dims.nj < 2), 'Linear-Quadratic not implemented for models with discrete state or choice'
        s0, x0 = np.atleast_1d(s0, x0)
        delta = self.time.discount

        # Fix shock at mean
        estar = np.inner(self.random.w, self.random.e)

        # Get derivatives
        f0, fx, fxx = self.reward(s0, x0, None, None)
        g0, gx, gxx = self.transition(s0, x0, None, None, None, estar)

        fs = jacobian(lambda s: self.reward(s, x0, None, None)[0], s0)
        fxs = jacobian(lambda s: self.reward(s, x0, None, None)[1], s0)
        fss = hessian(lambda s: self.reward(s, x0, None, None)[0], s0)
        gs = jacobian(lambda s: self.transition(s, x0, None, None, None, estar)[0], s0)

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

        f0 += - fs.dot(s0) - fx.dot(x0) + 0.5 * dot((s0.T, fss, s0)) + dot((s0.T, fsx, x0)) + 0.5 * dot((x0.T, fxx, x0))
        fs += - s0.T.dot(fss) - x0.T.dot(fxs)
        fx += - s0.T.dot(fsx) - x0.T.dot(fxx)
        g0 += - gs.dot(s0) - gx.dot(x0)

        # Solve Riccati equation using QZ decomposition
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
        C = np.real(Z[ds:, :ds] / Z[:ds, :ds])
        X = C[:dx]
        P = C[dx:, :]

        # Compute steady-state state, action, and shadow price
        t0 = np.r_[np.c_[fsx.T, fxx, delta * gx.T],
                  np.c_[fss, fsx, delta*gs.T - np.eye(ds)],
                  np.c_[gs - np.eye(ds), gx, np.zeros((ds, ds))]]
        t1 = np.r_[-fx.T, -fs.T, -g0]
        t = np.linalg.solve(t0, t1)
        sstar, xstar, pstar = np.split(t, [ds, ds + dx])
        vstar = (f0 + fs.dot(sstar) + fx.dot(xstar) + 0.5 * dot((sstar.T, fss, sstar)) +
                 dot((sstar.T, fsx, xstar)) + 0.5 * dot((xstar.T, fxx, xstar))) / (1 - delta)

        # Compute lq-approximation optimal policy and shadow price functions at state nodes
        s = self.Value.nodes.T
        sstar = sstar.T
        xstar = xstar.T
        pstar = pstar.T
        s -= sstar   # hopefully broadcasting works here  (np.ones(ns,1),:)  #todo make sure!!
        xlq = xstar + s.dot(X.T)  #(np.ones(1,ns),:)
        plq = pstar + s.dot(P.T)   #(np.ones(1,ns),:)
        vlq = vstar + s.dot(pstar.T) + 0.5 * np.sum(s * s.dot(P.T), axis=1,keepdims=True)

        self.Value[:] = vlq.T[:]
        self.Value_j[:]= vlq.T[:]
        self.Policy[:] = xlq.T[:]
        self.Policy_j[:] = xlq.T[:]

        return sstar, xstar, pstar











# TODO: design this class:
"""
    * Should finite-infinite horizon models be split into two subclasses?
    * Should discretized models be handled by a subclass?
    * Should vmax operate directly on Value_j and Policy_j? how to deal with residuals?
"""


