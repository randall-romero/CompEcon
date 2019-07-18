__author__ = 'Randall'

import numpy as np
from numpy.linalg import solve
from .tools import jacobian, hessian
from compecon.tools import Options_Container
from scipy.stats import norm as Normal_Distribution
import warnings


#
# SearchMeth = optget('qnewton','SearchMeth',3);
# StepMeth   = optget('qnewton','StepMeth',3);
# maxit      = optget('qnewton','maxit',250);
# maxstep    = optget('qnewton','maxstep',50);
# tol        = optget('qnewton','tol',sqrt(eps));
# eps0       = optget('qnewton','eps0',1);
# eps1       = optget('qnewton','eps1',1e-12);
# ShowIters  = optget('qnewton','ShowIters',0);

SQEPS = np.sqrt(np.spacing(1))


class OPoptions(Options_Container):
    """ A container for options to solve a UOP
    """
    description = 'Options for solving a Unconstraint Optimization Problem'

    def __init__(self, SearchMeth='bfgs', StepMeth='bt', maxit=250, maxsteps=50, tol=SQEPS,
                 show=False, eps0=1.0, eps1=1.e-12,all_x=False, print=None):
        self.SearchMeth = SearchMeth
        self.StepMeth = StepMeth
        self.maxit = maxit
        self.maxsteps = maxsteps
        self.tol = tol
        self.eps0 = eps0
        self.eps1 = eps1
        self.show = show
        self.all_x = all_x
        if print is not None:
            warnings.warn("Keyword 'print=' is deprecated. Use 'show=' instead")
            self.show = print

    def print_header(self):
        if self.show:
            print("Solving nonlinear equations by {}'s method".format(self.SearchMeth.capitalize()))
            print('{:4}  {:4}  {:6}'.format('it', 'bstep', 'change'))
            print('-' * 20)

    def print_current_iteration(self, it, backstep, fnormnew):
        if self.show:
            print('{:4}  {:4}  {:6.2e}'.format(it, backstep, fnormnew))

    def print_last_iteration(self, it):
        if (it + 1) == self.maxit:
            warnings.warn('Failure to converge after {} iteration in {} method.'.format(self.maxit, self.SearchMeth.capitalize()))


class OP(Options_Container):
    """
            x0:         initial value for x (solution guess)
    """

    def __init__(self, f, x0=None, *args, A=None, **kwargs):

        if callable(f):
                self.f = lambda x: f(x, *args)
        else:
            raise ValueError('First argument to NLP must be a function')

        self.n = 0  # number of x variables
        self.x0 = x0
        self.x = None # last solution found

        self._x_list = list() # last sequence of solutions
        self.opts = OPoptions(**kwargs)
        # self.step_methods = {'none': self._step_none,
        #                      'bhhh': self._step_bhhh,
        #                      'bt': self._step_bt,
        #                      'golden': self._step_golden
        #                      }
        # self.search_methods = {'steepest': self._search_steepest,
        #                        'bfgs': self._search_bfgs,
        #                        'dfp': self._search_dfp
        #                        }

        self.A = A  # inverse hessian
        self.reset = False
        self.errcode = True

    @property
    def x0(self):
        return np.copy(self._x0)

    @x0.setter
    def x0(self, value):
        self._x0 = np.atleast_1d(value)
        self.n = self._x0.size

    @property
    def x_sequence(self):
        return np.array(self._x_list).T

    @property
    def fx(self):
        return self.f(self.x)


    @property
    def fnorm(self):
        return np.max(np.abs(self.fx))

    def _get_initial_value(self, x):
        if x is not None:
            self.x0 = x

        if self.x0 is None or self.x0[0] is None:
            raise ValueError('Initial value is required to solve a OP, none provided!')

        self._x_list = [self.x0]
        return self.x0

    def _update_methods(self):
        self.step = self.step_methods[self.opts.StepMeth]
        self.search = self.search_methods[self.opts.SearchMeth]

    def golden(self, a, b, **kwargs):
        # Update solution options using kwargs
        self.opts[kwargs.keys()] = kwargs.values()
        self.opts.method = 'golden'
        φ = (np.sqrt(5) - 1) / 2  # golden ratio

        if a > b:
            a, b = b, a

        Δ = (b - a) * (1 - φ)
        x = a + Δ
        y = b - Δ
        fx, fy = self.f(x), self.f(y)

        Δ *= φ
        while Δ > self.opts.tol:
            Δ *= φ
            if fy < fx: # y is new upper bound
                x, y = x - Δ, x
                fx, fy = self.f(x), fx
            else:  # x is new lower bound
                x, y = y, y + Δ
                fx, fy = fy, self.f(y)

        return x if fx > fy else y

    def qnewton(self, x0=None, A=None, **kwargs):

        # Update solution options using kwargs
        self.opts[kwargs.keys()] = kwargs.values()
        self.opts.method = 'qnewton'
        x = self._get_initial_value(x0)

        # Unpack options
        self._update_methods()
        maxit, maxsteps, tol, all_x = self.opts['maxit', 'maxsteps', 'tol', 'all_x']
        eps0, eps1 = self.opts['eps0', 'eps1']

        eps = np.spacing(1)
        f0 = self.f(x)
        g0 = self.jacobian(x)

        if np.linalg.norm(g0) < eps:
            self.x = x
            return x

        if A is None:
            self.reset_inverse_hessian(f0)
        else:
            self.A = A

        for it in range(maxit):
            d = -np.dot(self.A, g0)  # search direction
            if (np.inner(d, g0) / (np.inner(d, d))) < eps1:  # must go uphill
                self.reset_inverse_hessian(f0) # otherwise use
                d = g0 / np.maximum(np.abs(f0), 1)  # steepest ascent

            s, f = self.step(self, x, f0, g0, d)
            if f <= f0:
                if self.reset:
                    warnings.warn('Iterations stuck in qnewton')
                    self.x = x
                    return x
                else:
                    self.reset_inverse_hessian(f0)
                    d = g0.T / np.maximum(abs(f0), 1)  # steepest ascent
                    s, f = self.step(self, x, f0, g0, d)
                    if self.errcode:
                        warnings.warn('Cannot find suitable step in qnewton')
                        self.x = x
                        return x
            d *= s
            x = x + d
            if all_x:
                self._x_list.append(x.copy())

            if np.any(np.isnan(x) | np.isinf(x)):
                raise ValueError('NaNs or Infs encountered')
            f = self.f(x)
            g = self.jacobian(x)
            self.opts.print_current_iteration(it, 0, np.linalg.norm(d))  # FIXME Mario's report more fields

            # Test convergence using Marquardt's criteria and gradient test
            if ((f - f0) / (abs(f) + eps0) < tol and
                    np.all(np.abs(d) / (np.abs(x) + eps0) < tol)) or\
                    np.all(np.abs(g) < eps):
                self.x = x
                return x

            # Update inverse Hessian
            u = g - g0  # change in Jacobian
            ud = np.inner(u, d)
            if np.all(np.abs(ud) < eps):
                self.reset_inverse_hessian(f)
            else:
                self.search(self, f, u, d)

            # Update iteration
            f0 = f
            g0 = g

        self.opts.print_last_iteration(it)
        warnings.warn('Maximum iterations exceeded in qnewton')

    def _step_none(self, x0, f0, g0, d):
        f = self.f(x0 + d)
        if f < f0:
            s = 1
            self.errcode = False
            return s, f
        else:
            return self._step_golden(x0, f0, d)

    def _step_bhhh(self, x0, f0, g0, d):
        # Intializations
        delta = 0.0001
        dg = -np.inner(d, g0)  # directional derivative
        tol1 = dg * delta
        tol0 = dg * (1 - delta)
        s, ds = 1, 1
        self.errcode = False

        # Bracket the cone
        for it in range(self.opts.maxit):
            x = x0 + s * d
            fs = self.f(x)
            temp = (f0 - fs) / s
            if temp < tol0:
                ds *= 2
                s += ds
            else:
                break

        if (tol0 <= temp) and (temp <=tol1):
            return s, fs

        ds /= 2
        s -= ds
        it0 = it + 1

        # Then use bisection to get inside it
        for it in range(it0, self.opts.maxit):
            ds /= 2
            x = x0 + s * d
            fs = self.f(x)
            temp = (f0 - fs) / s
            if temp > tol1:
                s -= ds
            elif temp < tol0:
                s += ds
            else:
                return s, fs

        # If it has not returned yet, call _step_golden!
        return self._step_golden(x0, f0, d)

    def _step_bt(self, x0, f0, g0, d):
        delta = 1e-4 # Defines cone of convergence; must be on (0,1/2)
        ub = 0.5     # Upper bound on acceptable reduction in s.
        lb = 0.1     # Lower bound on acceptable reduction in s.
        errcode = 0
        dg = -np.inner(d, g0)  # directional derivative
        tol1 = delta * dg
        tol0 = (1 - delta) * dg

        # full step
        s = 1
        fs = self.f(x0+d)
        if -fs + f0 <= tol1:
            return s, fs

        # quadratic approximation
        s2, fs2 = s, fs
        s = -0.5 * dg / (-fs + f0 - dg)
        s = max(s, lb)
        fs = self.f(x0 + s * d)
        temp = (-fs + f0) / s
        if (tol0 <= temp) and (temp <= tol1):
            return s, fs

        # cubic approximation
        for it in range(3, self.opts.maxit):
            temp = (s - s2) * np.array([s * s, s2 * s2])
            temp = np.array([- fs + f0 - dg * s, -fs2 + f0 - dg * s2]) / temp
            a = temp[0] - temp[1]
            b = s * temp[1] - s2 * temp[0]
            s2 = s
            fs2 = fs
            if np.all(a == 0):  # quadratic fits exactly
                s = -0.5 * dg / b
            else:
                disc = b * b - 3 * a * dg
                if np.all(disc < 0):
                    errcode = 2
                    return s, fs  # complex root
                s = (np.sqrt(disc) - b) / (3 * a)

            s = np.maximum(np.minimum(s, ub * s2), lb * s2)  # ensures acceptable step size
            fs = self.f(x0 + s * d)
            temp = (-fs + f0) / s
            if np.all(tol0 <= temp) and np.all(temp <= tol1):
                return s, fs

        # If it has not returned yet, call _step_golden instead
        return self._step_golden(x0, f0, d)

    def _step_golden(self, x0, f0, d):
        alpha1 = (3 - np.sqrt(5)) / 2
        alpha2 = (np.sqrt(5) - 1) / 2
        tol = 1.e-4
        tol *= alpha1*alpha2
        s = 1
        self.errcode = True
        self.iter = 0
        s0 = 0
        it = 0

        # Find a bracketing interval
        fs = self.f(x0 + d)
        if f0 >= fs:
            lenght = alpha1
        else:
            for it in range(self.opts.maxit):
                s *= 2
                fl = fs
                fs = self.f(x0 + s*d)
                if fs <=fl:
                    lenght = alpha1 * (s - s0)
                    break
                else:
                    s0 /= 2

            if (it + 1) >= self.opts.maxit:
                s /= 2
                fs = fl
                return s, fs

        xl = x0 + (s + lenght) * d
        xs = x0 + (s - lenght) * d

        s -= lenght
        lenght *= alpha2  # lenght now measures relative distance between xl and xs

        fs = self.f(xs)
        fl = self.f(xl)

        # Golden search to find minimum
        while it < self.opts.maxit:
            it += 1
            if fs < fl:
                s -= lenght
                lenght *= alpha2
                xs = xl
                xl -= lenght * d
                fs = fl
                fl = self.f(xl)
            else:
                lenght *= alpha2
                s += lenght
                xl = xs
                xs += lenght * d
                fl = fs
                fs = self.f(xs)

            if lenght < tol:
                self.errcode = False
                break
        if fl > fs:
            fs = fl
            s -= lenght
        return s, fs

    def _search_bfgs(self, ff=None, u=None, d=None):
            ud = np.inner(u, d)
            w = d - self.A.dot(u)
            wd = np.outer(w, d)
            self.A += ((wd + wd.T) - (np.inner(u, w) * np.outer(d, d)) / ud) / ud
            self.reset = False

    def _search_dfp(self, ff=None, u=None, d=None):
            ud = np.inner(u, d)
            v = self.A.dot(u)
            self.A += np.outer(d, d) / ud - np.outer(v, v) / np.inner(u, v)
            self.reset = False

    def _search_steepest(self, ff, u=None, d=None):
        self.reset_inverse_hessian(ff)

    def reset_inverse_hessian(self, ff):
        self.A = -np.identity(self.n) / np.maximum(abs(ff), 1)
        self.reset = True

    def jacobian(self, x):
        return jacobian(self.f, x)[0]  # indexing to eliminate a dimension (must always be a R-function)

    def hessian(self, x):
        return np.atleast_2d(hessian(self.f, x).squeeze())

    step_methods = {'none': _step_none,
                             'bhhh': _step_bhhh,
                             'bt': _step_bt,
                             'golden': _step_golden
                             }
    search_methods = {'steepest': _search_steepest,
                               'bfgs': _search_bfgs,
                               'dfp': _search_dfp
                               }

class MLE(OP):
    def estimate(self):
        self.beta = self.qnewton()
        self.Sigma = np.linalg.inv(-self.hessian(self.beta))

    def ci(self, signif=0.05):
        t = Normal_Distribution.ppf(1 - signif/2)
        bvar = np.diag(self.Sigma)
        tbst = t * np.sqrt(bvar)
        return np.array([self.beta - tbst, self.beta + tbst]).T