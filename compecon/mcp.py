import numpy as np
from numpy.linalg import solve
from .tools import jacobian
from warnings import warn

SQEPS = np.sqrt(np.spacing(1))


def fischer(u, v, du=None, dv=None, plus=True):
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


class NLP(object):
    def __init__(self, F, *args,
                 method='newton',
                 maxit=100, maxsteps=10,
                 tol=SQEPS, showiters=False):

        self.opts = {'method': method, 'maxit': maxit, 'maxsteps': maxsteps, 'tol': tol, 'showiters': showiters,
                     'initb': None, 'initi': False}

        self.printiterations = showiters

        if callable(F):
                self.f = lambda x, jac=False: F(x, jac, *args)
        else:
            raise ValueError('First argument to NLP must be function')

    def printHeader(self, method):
        if self.printiterations:
            print('Solving nonlinear equations by {} method'.format(method))
            print('{:4}  {:4}  {:6}'.format('it', 'bstep', 'change'))
            print('-' * 20)

    def printIteration(self, it, backstep, fnormnew):
        if self.printiterations:
            print('{:4}  {:4}  {:6.2e}'.format(it, backstep, fnormnew))

    def printBottom(self, it, maxit, method):
        if (it + 1) == maxit:
            print('Failure to converge after {} iteration in {} method.'.format(maxit, method))

    def unpackOptions(self, **kwargs):
        OPTS = self.opts.copy()
        for k, v in kwargs.items():
            if k in OPTS.keys():
                OPTS[k] = v
            else:
                print('Unknown option {}. Valid options are'.format(k), OPTS.keys())
        self.printiterations = OPTS['showiters']
        return OPTS

    def newton(self, x, **kwargs):

        self.printHeader("Newton's")
        opts = self.unpackOptions(**kwargs)
        maxit, maxsteps, tol = (opts[key] for key in ['maxit', 'maxsteps', 'tol'])

        for it in range(maxit):
            fx, J = self.f(x, True)
            fnorm = np.max(np.abs(fx))
            if fnorm < tol:
                return x, fx

            dx = - np.real(solve(J, fx))
            fnormold = np.inf

            for backstep in range(maxsteps):
                fxnew = self.f(x + dx, False)
                fnormnew = np.max(np.abs(fxnew))
                if fnormnew < fnorm:
                    break
                if fnormold < fnormnew:
                    dx *= 2
                    break
                fnormold = fnormnew
                dx /= 2
                # ---> end of back-step

            x += dx
            self.printIteration(it, backstep, fnormnew)

        self.printBottom(it, maxit, "Newton's")
        return x, fx


    def broyden(self, x, **kwargs):

        self.printHeader("Broyden's")
        opts = self.unpackOptions(**kwargs)
        maxit, maxsteps, tol = (opts[key] for key in ['maxit', 'maxsteps', 'tol'])
        Jinv = self.resetJacobian(x)

        for it in range(maxit):
            fx = self.f(x, False)
            fnorm = np.max(np.abs(fx))
            if fnorm < tol:
                return x, fx

            dx = - Jinv.dot(fx)
            fnormold = np.inf

            for backstep in range(maxsteps):
                fxnew = self.f(x + dx, False)
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

            if fnormnew > fnorm:
                Jinv = self.resetJacobian(x)
            else:
                self.updateJacobian(fxnew - fx, dx, Jinv)

            self.printIteration(it, backstep, fnormnew)

        self.printBottom(it, maxit,"Broyden's")
        return x, fx


    def resetJacobian(self, x):
        if self.opts['initb'] is None:
            if self.opts['initi']:
                fjacinv = - np.identity(x.size)
            else:
                fjacinv = np.linalg.pinv(jacobian(self.f, x, False))
        else:
            fjacinv = self.opts['initb']

        return fjacinv

    def updateJacobian(self, df, dx, fjacinv):
        temp = fjacinv.dot(df)
        fjacinv += np.outer(dx - temp, np.dot(dx, fjacinv)) / np.dot(dx, temp)

    def solve(self, x, **kwargs):
        if self.f(x, True)[0] is None:
            return self.broyden(self, x, **kwargs)
        else:
            return self.newton(self, x, **kwargs)


class MCP(NLP):
    def __init__(self, F, a, b, *args,
                 transform='ssmooth', **kwargs):

        if callable(F):
            NLP.__init__(self, F, *args, **kwargs)
            self.islinear = False
        elif isinstance(F, np.ndarray):
            if len(args) == 1:
                q = args[0]
                NLP.__init__(self, lambda x, jac: (F.dot(x) + q, F) if jac else F.dot(x) + q)
                self.islinear = True
                self.q = q
            else:
                raise ValueError('If problem is linear, exactly one argument (vector q) must be provided after the bounds')
        else:
            raise ValueError('First argument to MCP must be either a function (nonlinear problem) or a numpy array (linear problem')

        a, b = np.atleast_1d(a, b)
        self.a, self.b = a.astype(float), b.astype(float)
        self.ainf = np.isinf(a)
        self.binf = np.isinf(b)
        self.original = self.f
        self.opts['transform'] = transform

        self.transforms = {'ssmooth': self.ssmooth,
                           'minmax': self.minmax}

    def ssmooth(self, x, jac=False):

        x = np.atleast_1d(x)
        ainf, binf = self.ainf, self.binf
        AINF, BINF = ainf.all(), binf.all()
        Ainf, Binf = ainf.any(), binf.any()


        da, db = self.a - x, self.b - x

        if jac:
            fx, J = self.original(x, True)
            I = -np.identity(x.size)
            J = np.atleast_2d(J)

            # apply the Fischer + transform
            if AINF:
                fplus, Jplus = fx, J
            else:
                fplus, Jplus = fischer(fx, da, J, I)
                if Ainf:
                    fplus[ainf], Jplus[:, ainf] = fx[ainf], J[:, ainf]

            # apply the Fischer - transform
            if BINF:
                fhat, Jhat = fplus, Jplus
            else:
                fhat, Jhat = fischer(fplus, db, Jplus, I, False)
                if Binf:
                    fhat[binf], Jhat[:, binf] = fplus[binf], Jplus[:, binf]
            return fhat, Jhat
        else:
            fx = self.original(x, False)
            # apply the Fischer + transform
            if AINF:
                fplus = fx
            else:
                fplus = fischer(fx, da)
                if Ainf:
                    fplus[ainf] = fx[ainf]

            # apply the Fischer - transform
            if BINF:
                fhat = fplus
            else:
                fhat = fischer(fplus, db, plus=False)
                if Binf:
                    fhat[binf] = fplus[binf]
            return fhat

    def minmax(self, x, jac=False):

        a, b = self.a, self.b
        x = np.atleast_1d(x)
        da, db = a - x, b - x

        if jac:
            fx, J = self.original(x, True)
            fhat = np.minimum(np.maximum(fx, da), db)
            J = np.atleast_2d(J)
            Jhat = -np.identity(x.size)
            i = (fx > da) & (fx < db)
            Jhat[i] = J[i]
            return fhat, Jhat
        else:
            fx = self.original(x, False)
            return np.minimum(np.maximum(fx, da), db)

    def transformProblem(self, x0, transform):

        # Choose proper transformation
        validoption = lambda option: option in ['ssmooth', 'minmax']
        transform = transform if validoption(transform) else self.opts['transform']
        self.f = self.transforms[transform]

        # Set an initial value
        a, b = self.a, self.b
        if x0 is None:
            x0 = self.q if self.islinear else (a + b) / 2
        else:
            x0 = np.atleast_1d(x0).astype(np.float64)
        return x0

    def newton(self, x0=None, transform=None, **kwargs):
        x = self.transformProblem(x0, transform)
        return NLP.newton(self, x, **kwargs)

    def broyden(self, x0=None, transform=None, **kwargs):
        x = self.transformProblem(x0, transform)
        return NLP.broyden(self, x, **kwargs)

    def solve(self, x0=None, transform=None, **kwargs):
        if self.original(x0, True)[1] is None: # true if jacobian is missing
            return self.broyden(x0, transform, **kwargs)
        else:
            return self.newton(x0, transform, **kwargs)