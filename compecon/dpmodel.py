import time

from compecon.tools import Options_Container
from .nonlinear import MCP
from .lcpstep import lcpstep

# from .basis import Basis
from .interpolator import Interpolator
import numpy as np
# from scipy import linalg


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
        h   deterministic discrete state transitions.
    """
    description = 'Random components of a DPmodel object'

    def __init__(self, e=0.0, w=1.0, q=None, h=None):
        self.e = np.atleast_2d(e)
        self.w = np.atleast_1d(w)
        self.q = q
        self.h = h

    def set_transition_matrix(self, ni, nj):
        if ni == 1:
            self.h = np.zeros([nj, 1], int)

        if self.q is None:
            q_ = np.zeros([ni, ni, nj])
            if self.h is None:
                if ni == nj:
                    print('Neither q nor h specified; will default to h(j,i)=j.')
                    print('Please specify q or h if this is not correct.')
                    print(' ')
                else:
                    raise ValueError('Either q or h must be specified.')

            for i in range(ni):
                for j in range(nj):
                    q_[i, self.h[j, i], j] = 1

            self.q = q_


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

    def __init__(self, basis, dx, ni, nj):
        self.s = basis.opts.varnames
        self.x = ['Policy ' + str(k) for k in range(dx)]
        self.i = ['State ' + str(k) for k in range(ni)]
        self.j = ['Choice ' + str(k) for k in range(nj)]


class DPoptions(Options_Container):
    """ Container for numerical options to solve a DPmodel

    Attributes:
        algorithm             algorithm for solver
        tol                   convergence tolerance parameter
        ncpmethod             method for complementarity problem
        maxit                 maximum number of iterations
        maxitncp              maximunm number of iterations for ncpmethod
        showiters             print iteration summary when solving the model
        discretized           true if continuous action is discretized or not present.
        X                     dx.nx discretized continuous actions
        D_reward_provided     true if Jacobian and Hessian of reward are provided
        D_transition_provided true if Jacobian and Hessian of transition are provided
        knownFunctions        ni.nj boolean array, true if discrete policy and value functions are known
        output                whether to print output
    """
    description = "Solver options for a DPmodel"

    def __init__(self, algorithm='newton', tol=np.sqrt(np.spacing(1)), ncpmethod='minmax',
                 maxit=200, maxitncp=50, showiters=False, discretized=False, X=None,
                 D_reward_provided=True, D_transition_provided=True, knownFunctions=None, output=True):
        self.algorithm = algorithm
        self.tol = tol
        self.ncpmethod = ncpmethod
        self.maxit = maxit
        self.maxitncp = maxitncp
        self.showiters = showiters
        self.discretized = discretized
        self.X = X
        self.D_reward_provided = D_reward_provided
        self.D_transition_provided = D_transition_provided
        self.knownFunctions = knownFunctions
        self.output = output

    def print_header(self, method, horizon):
        horizon = 'infinite' if np.isinf(horizon) else 'finite'
        if self.output:
            print('Solving %s-horizon model collocation equation by %s method' % (horizon, method))
            if self.showiters:
                print('{:4s} {:12s} {8s}'.format('iter', 'change', 'time'))
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
        if self.output and self.showiters:
            print('{:4i}  {:12.1e}  {:8.4f}'.format(it, change, time.time() - tic))

    def print_last_iteration(self, tic, change):
        """ Prints summary of last iteration in solve method

        Args:
          tic: time when iterations started
          change: distance between last two iterations

        Returns:
          prints output to screen
        """
        if self.output:
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

    def __init__(self, basis, ni, nj, dx,
                 discount=0.0, horizon=np.inf,
                 e=0, w=1, q=None, h=None):

        #  Value and policy functions
        self.Value = Interpolator(basis, s=[ni])
        self.Policy = Interpolator(basis, s=[ni, dx])
        self.Value_j = Interpolator(basis, s=[ni, nj])
        self.Policy_j = Interpolator(basis, s=[ni, nj, dx])
        self.DiscreteAction = np.zeros([ni, basis.N], int)

        # Default numerical solution parameters
        self.options = DPoptions()

        # Stochastic specification
        self.random = DPrandom(e, w, q, h)

        # Time parameters
        self.time = DPtime(discount, horizon)

        # Labels for model variables
        self.labels = DPlabels(basis, dx, ni, nj)

        # Model dimensions
        self.dims = DPdims(basis.d,  # number of continuous state variables
                           basis.N,  # number of continuous state nodes
                           dx,  # number of continuous policy variables
                           0,  # number of discretized policy values
                           ni,  # number of discrete states
                           nj,  # number of discrete choices
                           basis.M)  # number of collocation coefficients

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

    def solve(self, solver=None):
        """ Solves the model

        Args:
          solver: a DPsolver object (optional)

        Returns:

        """

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

        # Set transition matrix
        self.random.set_transition_matrix(ni, nj)

        ''' 2: SOLVE THE MODEL******************** '''
        if np.isfinite(self.time.horizon):
            self.__solve_backwards()
            return
        elif self.options.algorithm == 'funcit':
            self.__solve_by_function_iteration()
        elif self.options.algorithm == 'newton':
            self.__solve_by_Newton_method()
        else:
            raise ValueError('Unknown solution algorithm')

        self.update_policy_function()

    def __solve_backwards(self):
        raise NotImplementedError()

    def __solve_by_function_iteration(self):
        """
            Solves infinite-horizon model collocation equation by function iteration. Solution is found when the
            collocation coefficients of the value function converge to a fixed point (within |self.tol| tolerance).
         """
        tic = time.time()
        self.options.print_header('function iteration', self.time.horizon)
        for it in range(self.options.maxit):
            cold = self.Value.c.copy()
            self.vmax(vc=False)
            self.update_value_function()
            change = np.linalg.norm((self.Value.c - cold).flatten(), np.Inf)
            self.options.print_current_iteration(it, change, tic)
            if change < self.options.tol:
                break
        self.options.print_last_iteration(tic, change)

    def __solve_by_Newton_method(self):
        tic = time.time()
        self.options.print_header("Newton's", self.time.horizon)
        # todo: fix the dimensions and check that Phik is transposed?
        Phik = np.kron(np.eye(self.dims.ni), self.Value.Phi)
        for it in range(self.options.maxit):
            cold = self.Value.c.copy().flatten()
            vc = self.vmax()
            self.update_value_function()
            step = - np.linalg.lstsq(Phik - vc, Phik * cold - self.Value.y.flatten())
            c = cold + step
            change = np.linalg.norm(step, np.Inf)
            self.Value.c = c.reshape(self.dims.nc, self.dims.ni)
            self.options.print_current_iteration(it, change, tic)
            if change < self.options.tol:
                break
        self.options.print_last_iteration(tic, change)

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

    def vmax(self, s=None, x=None, vc=True):  # [v,x,vc]
        # Unpack model structure
        delta = self.time.discount
        ds, dx, ni, nj, ns = self.dims['ds', 'dx', 'ni', 'nj', 'ns']
        e, w, q = self.random['e', 'w', 'q']

        X = self.options.X                                      # discrete

        if s is None:
            s = self.Value.nodes
            x = self.Policy_j.y
            v = self.Value_j.y
        else:
            ns = s.shape[1]  # number of continuous state nodes
            v = np.zeros((ni, nj, ns))
            if x is None:
                x = self.Policy_j(s)

        if self.options.discretized:
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_discretized(s, x, i, j)
        else:
            for i in range(ni):
                for j in range(nj):
                    v[i, j] = self.vmax_continuous(s, x, i, j)

        self.Value_j.update_c()
        self.Policy_j.update_c()

        # Compute derivative of Bellman equation optimand with respect to basis coefficients for Newton method
        if vc:
            VC = self.vmax_derivative()
            return VC
        else:
            return None

    # vmax_discretized
    # Nested function in vmax: Finds the optimal policy and value function for a given pair of discrete state
    # and discrete action, when the continuous policy has been discretized.
    def vmax_discretized(self, s, x, i, j):
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
    def vmax_continuous(self, s, x, i, j):
        ns = self.dims.ns
        ni = self.dims.ni
        xl, xu = self.bounds(s, i, j)

        KKT = MCP(self.kkt, xl, xu, s, i, j)

        for it in range(self.options.maxitncp):
            # Initialize optimand and derivatives with reward terms
            if self.options.D_reward_provided:
                vv, vx, vxx = self.reward(s, x[i, j], i, j)
            else:
                vv, vx, vxx = self.getDerivative('reward', s, x[i, j], i, j)

            for k in range(self.random.w.size):
                # Compute states next period and derivatives
                ee = np.tile(self.random.e[:, k: k + 1], ns)
                for in_ in range(ni):
                    if self.random.q[i, in_, j] == 0:
                        continue

                    if self.options.D_transition_provided:
                        snext, snx, snxx = self.transition(s, x[i, j], i, j, in_, ee)
                    else:
                        snext, snx, snxx = self.getDerivative('transition', s, x[i, j], i, j, in_, ee)

                    snext = np.real(snext)
                    prob = self.random.w[k] * self.random.q[i, in_, j]

                    vn, vns, vnss = self.Value[in_](snext, order='all')  # evaluates function, jacobian, and hessian if order='all'  #todo: implement this bahavior

                    vv += self.time.discount * prob * vn
                    vx = np.einsum('k...,jk...->j...', vns, snx)
                    vxx = np.einsum('hi...,ij...,kj...->hk...', snx, vnss, snx) +  \
                          np.einsum('k...,ijk...->ij...', vns, snxx)


            # Compute Newton step, update continuous action, check convergence
            vx, delx = lcpstep(self.options.ncpmethod, x[i, j], xl, xu, vx, vxx)
            x[i, j] += delx

            if np.linalg.norm(vx.flatten(), np.inf) < self.options.tol:  break

        return vv


    def vmax_derivative(self):
        # ## vmax_derivative
        # # Nested function in vmax:  Computes derivative with respect to Value function interpolation coefficients
        # def vmax_derivative():
        #     if ni * nj > 1:
        #         vc = np.zeros(ni, ns, ni, ns)
        #         jmax = np.argmax(v, 1)
        #         for i_ in range(ni):
        #             for j_ in range(nj):
        #                 is_ = jmax[i_] == j_
        #                 if not any(is_): continue
        #
        #                 for k in range(w.size):
        #                     ee = np.tile(e[:, k : k+1], ns)
        #                     for in_ in range(ni):
        #                         if q[i_, in_, j_] == 0: continue
        #                         snext = self.transition(s[:, is_], x[i_, j_, :, is_], i_ , j_, in_, ee[:, is_])
        #                         B = self.Value.interpolation(snext)
        #                         prob = w[k] * q[i_, in_, j_]
        #                         vc[i_, is_, in_] += delta * prob * reshape(B,length(is_),1,ns)
        #
        #
        #
        #
        #
        #         vc.shape = (ni * ns, ni * ns)
        #     else:
        #         vc = zeros(ns,ns)
        #         for k=1:length(w)
        #             ee = e[k+zeros(ns,1),:]
        #             snext = self.transition(s,x,1,1,1,ee)
        #             vc = vc + delta*w[k]*self.Value.Interpolation(snext)
        #
        #     return vc
        raise NotImplementedError

    def expectedValue(self, s, x, i, j, jac):
        delta = self.time.discount
        ds, dx, ni, nj, ns = self.dims['ds', 'dx', 'ni', 'nj', 'ns']  # dimensions
        e, w, q = self.random['e', 'w', 'q']                # randomness

        if self.options.D_reward_provided:
            vv, vx, vxx = self.reward(s, x[i, j], i, j)
        else:
            vv, vx, vxx = self.getDerivative('reward', s, x[i, j], i, j)

        for k in range(w.size):
            # Compute states next period and derivatives
            ee = np.tile(e[:, k : k+1], ns)
            for in_ in range(ni):
                if q[i, in_, j] == 0:
                    continue

                if self.options.D_transition_provided:
                    snext, snx, snxx = self.transition(s, x[i,j], i, j, in_, ee)
                else:
                    snext, snx, snxx = self.getDerivative('transition', s, x[i, j], i, j, in_, ee)

                snext = np.real(snext)
                prob = w[k] * q[i, in_, j]

                vn, vns, vnss = self.Value[in_](snext, order='all')  # evaluates function, jacobian, and hessian if order='all'  #todo: implement this bahavior


                vv += delta * prob * vn
                vx = np.einsum('k...,jk...->j...', vns, snx)

                if jac:
                    vxx = np.einsum('hi...,ij...,kj...->hk...', snx, vnss, snx) +  \
                          np.einsum('k...,ijk...->ij...', vns, snxx)

            return (vv, vx, vxx) if jac else (vv, vx)

    def kkt(self, xvec, jac, s, i, j):
        x = xvec.reshape((self.dims.dx, self.dims.ns))
        out = self.expectedValue(s, x, i, j, jac)
        return out[1:] if jac else out[1]






    def update_value_function(self):
        value = self.Value_j.y
        self.DiscreteAction = jmax = np.array([np.argmax(Vi, 0) for Vi in value])
        for i in range(self.dims.ni):
            self.Value.y[i] = value[i][jmax[i], range(self.dims.ns)]

    def update_policy_function(self):
        policy = self.Policy_j.y.swapaxes(1, 2)
        jmax = self.DiscreteAction
        for i in range(self.dims.ni):
            for k in range(self.dims.dx):
                self.Policy.y[i, k] = policy[i, k][jmax[i], range(self.dims.ns)]

    def VNEXT(self):
        dx, ds, ns = self.dims['dx', 'ds', 'ns']
        SNX = np.ones((dx, ds, ns))
        SNXX = np.ones((dx, dx, ds, ns))

        VN =  np.random.random_integers(1, 9, ns) / 9
        VNS = np.random.random_integers(1, 9, (ds, ns))/9
        VNSS = np.random.random_integers(1, 9, (ds, ds, ns))/9

        """vnx  =  vns * snx """
        VNX = np.einsum('k...,jk...->j...', VNS, SNX)

        """vnxx   =  snx' * vnss * snx + Sum_{k}[ vns(k) * snxx(:,:,k)]         """
        VNXX = np.einsum('hi...,ij...,kj...->hk...', SNX, VNSS, SNX) + np.einsum('k...,ijk...->ij...', VNS, SNXX)

        return VN, VNX, VNXX



# TODO: design this class:
"""
    * Should finite-infinite horizon models be split into two subclasses?
    * Should discretized models be handled by a subclass?
    * Should vmax operate directly on Value_j and Policy_j? how to deal with residuals?
"""


