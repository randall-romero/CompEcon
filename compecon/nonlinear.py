import numpy as np
import pandas as pd
from numpy.linalg import solve
from scipy.sparse import issparse, identity
from scipy.sparse.linalg import spsolve
from .tools import jacobian
from compecon.tools import Options_Container
import warnings

SQEPS = np.sqrt(np.spacing(1))


def fischer(u, v, du=None, dv=None, plus=True):
    """ Computes Fischer's function

    phi±(u, v) = u + v ± sqrt(u^2 + v^2)

    In turn, it is assumed that u and v are functions of x. If the Jacobian of Fischer's
    function wrt x is required, then partial derivatives du and dv are required.

    Fischer's function is useful to transform a complementarity problem into
    a nonlinear root-finding problem.

    Args:
        u:    first term
        v:    second term
        du:   Jacobian of u wrt x (optional)
        dv:   Jacobian of v wrt x (optional)
        plus: if True (default), compute  u + v + sqrt(u^2+v^2), else u + v - sqrt(u^2+v^2)

    Returns:
        phi:                  if either du or dv is None
        phi, d(phi)/dx        if both du and dv are provided

    References:
        Miranda and Fackler 2002 Applied Computational Economics and Finance, pp. 49-50
    """
    s = 1 if plus else -1
    sq = np.sqrt(u * u + v * v)
    f = u + v + s * sq

    if du is None or dv is None:
        return f

    nx = du.shape[1]
    nx1 = [nx, 1]
    U, V, SQ = [np.tile(w, nx1).T for w in (u, v, sq)]

    J = du + dv + s * (U * du + V * dv) / SQ
    return f, J


class NLPoptions(Options_Container):
    """ A container for options to find a zero for a NLP or MCP

    Attributes: default in brackets
        method:     either ['newton'] or 'broyden'
        maxit:      maximum number of iterations [100]
        maxsteps:   maximum number of backsteps [10]
        tol:        convergence tolerance [sqrt(machine eps)]
        initb:      an initial inverse Jacobian aprroximation matrix [None]
        initi:      if True, use the identity matrix to initialize Jacobian,
                    if [False], a numerical Jacobian will be used
        transform:  either ['ssmooth'] or 'minmax', required for MC problems
        show:       print to screen if [True], quiet if False
        all_x:      whether to output the full solution sequence too [False]
        squeeze:    if problem has ony one dimension, return solution as scalar [True]
    """
    description = 'Options for solving a NLP'

    def __init__(self, method='newton', maxit=100, maxsteps=10, tol=SQEPS,
                 show=False, initb=None, initi=False, transform='ssmooth', all_x=False, squeeze=True):
        self.method = method
        self.maxit = maxit
        self.maxsteps = maxsteps
        self.tol = tol
        self.initb = initb
        self.initi = initi
        self.transform = transform
        self.show = show
        self.all_x = all_x
        self.squeeze = squeeze

    def print_header(self):
        if self.show:
            print("Solving nonlinear equations by {}'s method".format(self.method.capitalize()))
            print('{:4}  {:4}  {:6}'.format('it', 'bstep', 'change'))
            print('-' * 20)

    def print_current_iteration(self, it, backstep, fnormnew):
        if self.show:
            print('{:4}  {:4}  {:6.2e}'.format(it, backstep, fnormnew))

    def print_last_iteration(self, it, x):
        if (it + 1) == self.maxit:
            warnings.warn('Failure to converge after {} iteration in {} method.'.format(self.maxit, self.method.capitalize()))
        else:
            print('Solution is = ', x)


class NLP(Options_Container):
    """
            x0:         initial value for x (solution guess)
    """
    def __init__(self, f, x0=None, **kwargs):
        self.x0 = np.asarray(x0)
        self.x = None # last solution found
        self._x_list = list() # last sequence of solutions
        self.opts = NLPoptions(**kwargs)
        self.it = -1  # iterations needed in last solution (without backsteps)

        if callable(f):
                self._f = f  #lambda x: f(x, *args)
                self.user_provides_jacobian = False
        else:
            raise ValueError('First argument to NLP must be a function')


    @property
    def x0(self):
        return np.copy(self._x0)

    @x0.setter
    def x0(self, value):
        self._x0 = np.atleast_1d(value)

    @property
    def x_sequence(self):
        df = pd.DataFrame(self._x_list)
        df.index.name = 'iteration'
        df.columns = [f'x_{n}' for n in df.columns]
        return df

    @property
    def fx(self):
        return self.f(self.x)[0]

    @property
    def fnorm(self):
        return np.max(np.abs(self.fx))

    def return_solution(self, x, it):
        y = x.copy()
        if self.opts.squeeze and x.size == 1:
            y = y[0]

        self.x, self.it = y, it
        return self.x

    def check_whether_there_is_a_jacobian(self, x0):
        resultado = self._f(self.x0)
        self.user_provides_jacobian = (type(resultado) is tuple) and (len(resultado) == 2)

    def f(self, x):
        fx = self._f(x)
        if self.user_provides_jacobian:
            return np.asarray(fx[0]).flatten(), np.asarray(fx[1])
        else:
            return np.asarray(fx).flatten(), None

    def _get_initial_value(self, x):
        if x is not None:
            self.x0 = np.asarray(x)

        if self.x0 is None or self.x0[0] is None:
            raise ValueError('Initial value is required to zero a NLP, none provided!')

        if type(self) in (MCP, LCP):
            # This step is important, as it sets the proper transformation (minmax, ssmooth) in case the original
            # problem has complementarity conditions
            if self.opts.show:
                print('Using the %s transformation' % self.opts.transform.upper())
            self.transform_problem()

        self._x_list = [self.x0]
        return self.x0

    def _keep_within_limits(self, x):
        return x

    def newton(self, x0=None, **kwargs):
        # Check if Jacobian is available, switch to Broyden's method if not.
        x = self._get_initial_value(x0)
        self.check_whether_there_is_a_jacobian(x0)

        if not self.user_provides_jacobian:
            print("Newton's method requires Jacobian function, but none is provided.\n",
                  "Using Broyden's method instead")
            return self.broyden(x0, **kwargs)

        # Update solution options using kwargs
        self.opts[kwargs.keys()] = kwargs.values()
        self.opts.method = 'newton'

        # Unpack options and initial value
        maxit, maxsteps, tol, all_x = self.opts['maxit', 'maxsteps', 'tol', 'all_x']


        # Iterate to find solution
        self.opts.print_header()
        for it in range(maxit):
            fx, J = self.f(x)
            if not issparse(J):  #with sparse matrices doesn't work nice
                J = np.atleast_2d(J)
            fnorm = np.max(np.abs(fx))
            if fnorm < tol:
                return self.return_solution(x, it)

            solve_func = spsolve if issparse(J) else solve

            dx = -np.real(solve_func(J, fx))


            # this part comes from my ncpsolve4 matlab file
            if np.any(np.isnan(dx)):
                notYet = np.array([np.linalg.norm(J[k]) > SQEPS for k in range(dx.size)])
                dx = -np.real(np.linalg.lstsq(J[notYet], fx[notYet])[0])


            fnormold = np.inf

            for backstep in range(maxsteps):
                fxnew = self.f(self._keep_within_limits(x + dx))[0]  # only function evaluation, not Jacobian
                fnormnew = np.max(np.abs(fxnew))
                if fnormnew < fnorm:
                    break
                if fnormold < fnormnew:
                    dx *= 2
                    break
                fnormold = fnormnew
                dx /= 2
                # ---> end of back-step
            x = self._keep_within_limits(x + dx)
            if all_x:
                self._x_list.append(x.copy())

            if maxsteps > 0:
                self.opts.print_current_iteration(it, backstep, fnormnew)
            else:
                self.opts.print_current_iteration(it, 0, fnorm)

        self.opts.print_last_iteration(it, x)
        return self.return_solution(x, it)

    def broyden(self, x0=None, **kwargs):
        # Check if Jacobian is available
        x = self._get_initial_value(x0)
        self.check_whether_there_is_a_jacobian(x)

        # Update solution options using kwargs

        self.opts[kwargs.keys()] = kwargs.values()
        self.opts.method = 'broyden'

        # Unpack options and initial values
        maxit, maxsteps, tol, all_x = self.opts['maxit', 'maxsteps', 'tol', 'all_x']
        maxsteps = max(1, maxsteps)
        fx, _ = self.f(x)

        Jinv = self.reset_inverse_jacobian(x)

        fnorm = np.max(np.abs(fx))

        # Iterate to find solution
        self.opts.print_header()
        for it in range(maxit):
            if fnorm < tol:
                return self.return_solution(x, it)
            dx = - Jinv.dot(fx)
            fnormold = np.inf

            for backstep in range(maxsteps):
                fxnew, _ = self.f(x + dx)
                fnormnew = np.max(np.abs(fxnew))
                if fnormnew < fnorm:
                    break
                if fnormold < fnormnew:
                    fxnew = fxold
                    dx *= 2
                    break

                fnormold = fnormnew
                dx /= 2
                fxold = fxnew.copy()
                #---> end of back-step

            x += dx
            if all_x:
                self._x_list.append(x.copy())

            if fnormnew > fnorm:
                Jinv = self.reset_inverse_jacobian(x)
            else:
                self.update_inverse_jacobian(fxnew - fx, dx, Jinv)

            fx = fxnew
            fnorm = fnormnew
            self.opts.print_current_iteration(it, backstep, fnormnew)

        self.opts.print_last_iteration(it, x)
        return self.return_solution(x, it)

    def funcit(self, x0=None, **kwargs):
        x = self._get_initial_value(x0)
        self.check_whether_there_is_a_jacobian(x)
        f_original = self.f
        self.f = lambda z: (z - f_original(z)[0], None)
        x = self.fixpoint(x0, **kwargs)
        self.f = f_original
        return x

    def fixpoint(self, x0=None, **kwargs):
        x = self._get_initial_value(x0)
        self.check_whether_there_is_a_jacobian(x)
        # Update solution options using kwargs
        self.opts[kwargs.keys()] = kwargs.values()
        maxit, tol, old_method, all_x = self.opts['maxit', 'tol', 'method', 'all_x']
        self.opts.method = 'fixpoint'
        self.opts.print_header()
        self.opts.method = old_method



        for it in range(maxit):
            xold = x
            x, _ = self.f(x)
            if all_x:
                self._x_list.append(x.copy())
            step = np.linalg.norm(x - xold)
            self.opts.print_current_iteration(it, 0, step)
            if step < tol:
                return self.return_solution(x, it)

        self.opts.print_last_iteration(it, x)
        warnings.warn('Failure to converge in fixpoint')

    def bisect(self, a, b, **kwargs):
        x = self._get_initial_value(a)
        self.check_whether_there_is_a_jacobian(x)
        # Update solution options using kwargs
        self.opts[kwargs.keys()] = kwargs.values()
        tol = self.opts.tol

        if a > b:
            a, b = b, a

        #faj, _ = self.f(a)
        f = lambda xx: self.f(xx)[0]

        fa, fb = f(a), f(b)

        sa = np.sign(fa)
        sb = np.sign(fb)

        if sa == sb:
            raise ValueError('In bisect: function has same sign at interval endpoints.')

        # Initializations
        dx = (b - a) / 2
        tol *= dx
        x = a + dx
        dx *= sb

        # Iteration loop
        it=0
        while abs(dx) > tol:
            fx = f(x)
            self.opts.print_current_iteration(x, 0, abs(fx))
            dx *= 0.5
            x -= np.sign(fx) * dx
            it +=1

        self.x = x
        return self.return_solution(x, it)

    def reset_inverse_jacobian(self, x):
        if self.opts.initb is None:
            if self.opts.initi:
                fjacinv = - np.identity(x.size)
            else:
                #fjac = self.f(x)[1]
                fjacinv = np.linalg.inv(jacobian(lambda z:self.f(z)[0], x))
                #fjacinv = np.linalg.pinv(np.atleast_2d(fjac))
        else:
            fjacinv = self.opts.initb

        return fjacinv

    def update_inverse_jacobian(self, df, dx, fjacinv):
        """ Rule to update the inverse of Broyden's approximation of the Jacobian

        Arguments:
            df:  change in f value = f(x_t) - f(x_{t-1})
            dx:  change in x = x_t - x_{t-1}
            fjacinv: current Broyden inverse Jacobian = J^{-1}(x_{t-1})

        Returns:
            None  The inverse Jacobian is updated in place, so fjacinv becomes J^{-1}(x_t)

        References:
            Miranda and Fackler, p. 38
        """
        temp = fjacinv.dot(df)
        fjacinv += np.outer(dx - temp, np.dot(dx, fjacinv)) / np.dot(dx, temp)

    def check_jacobian(self, x=None, signif=8):
        if x is None:
            x = self.x0

        if not self.user_provides_jacobian:
            print('Jacobian was not provided by user!')
            return None

        func = lambda z: self.f(z)[0]

        f, provided_jacobian = self.f(x)
        numerical_jacobian = jacobian(func, x)
        gap = np.abs(provided_jacobian - numerical_jacobian)
        idx = np.unravel_index(gap.argmax(), gap.shape)

        log10gap = np.ceil(np.log10(gap))
        trouble = -log10gap < signif

        n_funcs, n_vars = gap.shape

        if np.any(trouble):
            print("In entries marked by '#' not all of the first {:d} decimal digits are the same in the numerical ")
            print('and the user-provided derivatives.\n'.format(signif))
            tmp = lambda a: '#' if a else '.'
            header_frmt = '\t      ' + '{:5s}' * n_vars
            headers = ['x' + str(k) for k in range(n_vars)]
            print(header_frmt.format(*headers))

            row_frmt = '\tf_{:<4d} ' + '{:5s}' * n_vars
            for k in range(n_funcs):
                symbols = [tmp(x) for x in trouble[k]]
                print(row_frmt.format(k, *symbols))

        else:
            print("All numerical derivatives differ from")
            print('the user-provided ones by less than {:d} decimal digits.\n'.format(signif))

        print('The maximum error is {:.2e}, for row {:d} and column {:d}.\n'.format(gap[idx], *idx))

        return None

    def zero(self, x0=None, **kwargs):
        x = self._get_initial_value(x0)
        self.check_whether_there_is_a_jacobian(x)
        self.opts[kwargs.keys()] = kwargs.values()
        if self.user_provides_jacobian and self.opts.method == 'newton':
            return self.newton(x, **kwargs)
        else:
            return self.broyden(x, **kwargs)





class MCP(NLP):

    def __init__(self, f, a, b, x0=None, **kwargs):
        a, b = np.atleast_1d(a, b)
        a, b = a.astype(float), b.astype(float)
        if x0 is None:
            x0 = (a + b) / 2

        super().__init__(f, x0, **kwargs)
        self.islinear = False
        self.a, self.b = a, b
        self.hasLowerBound = np.isfinite(a)
        self.hasUpperBound = np.isfinite(b)
        self._original = self._f

    def _ssmooth(self, x):
        x = np.atleast_1d(x)
        L, U = self.hasLowerBound, self.hasUpperBound
        da, db = self.a - x, self.b - x

        fx = self._original(x)
        if type(fx) is tuple:
            fx, J = fx

        if self.opts.method == 'newton':  # return the Jacobian
            I = -np.identity(x.size)
            if not issparse(J):
                J = np.atleast_2d(J)

            if np.any(L):  # apply the Fischer + transform
                fx[L], J[:, L] = fischer(fx[L], da[L], J[:, L], I[:, L])
            if np.any(U):  # apply the Fischer - transform
                fx[U], J[:, U] = fischer(fx[U], db[U], J[:, U], I[:, U], False)
            return fx, J
        else:
            if np.any(L):  # apply the Fischer + transform
                fx[L] = fischer(fx[L], da[L])
            if np.any(U):  # apply the Fischer - transform
                fx[U] = fischer(fx[U], db[U], plus=False)
            return fx, None

    def _minmax(self, x):
        a, b = self.a, self.b
        x = np.atleast_1d(x)
        da, db = a - x, b - x

        fx = self._original(x)
        if type(fx) is tuple:
            fx, J = fx
        else:
            J = jacobian(self._original, x)

        fhat = np.fmin(np.fmax(fx, da), db)
        if True:  #self.opts.method is 'newton': # compute the Jacobian  #fixme not sure this is ok
            if issparse(J):
                Jhat = -identity(x.size, format='csc')
            else:
                Jhat = -np.identity(x.size)
                J = np.atleast_2d(J)

            i = (fx > da) & (fx < db)
            if np.any(i):
                Jhat[i] = J[i]
            return fhat, Jhat
        else:
            return fhat, None

    def ssmooth(self, x):
        dx = self.hasLowerBound.size
        x = np.atleast_2d(x) if dx > 1 else np.atleast_1d(x)
        return np.array([self._ssmooth(xi)[0] for xi in np.atleast_1d(x)])

    def _keep_within_limits(self, x):
        return np.minimum(self.b, np.maximum(self.a, x))


    def minmax(self, x):
        dx = self.hasLowerBound.size
        x = np.atleast_2d(x) if dx > 1 else np.atleast_1d(x)
        return np.array([self._minmax(xi)[0] for xi in np.atleast_1d(x)])

    def original(self,x):
        fJ = self._original(x)
        if type(fJ) is tuple:
            return self._original(x)[0]
        else:
            return self._original(x)

    def transform_problem(self):
        # Choose proper transformation
        if self.opts.transform == 'ssmooth':
            self.f = self._ssmooth
        else:
            self.f = self._minmax

    @property
    def a_is_binding(self):
        return np.isclose(self.a, self.x)

    @property
    def b_is_binding(self):
        return np.isclose(self.b, self.x)




class LCP(MCP):
    def __init__(self, M, q, a, b, x0=None, **kwargs):
        M = np.atleast_2d(M)
        q = np.atleast_1d(q)
        if x0 is None:
            x0 = q
        super().__init__(lambda z: (M.dot(z) + q, M), a, b, x0, **kwargs)
        self.islinear = True
        self.q = q

    def lemke(self, x0=None):
        pass


# class COP(MCP):
#     """ Constrained Optimization Problem"""
#     def __init__(self, f, g=None, h=None):
#         """
#
#         :param f: objective function -> max_x f(x)
#         :param g: inequality constrains, g(x) >= 0
#         :param h: equality constrains, h(x) = 0
#         """
#         super().__init__(f,a,b,*args,**kwargs)




