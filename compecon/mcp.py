import numpy as np
from numpy.linalg import solve
from warnings import warn

SQEPS = np.sqrt(np.spacing(1))



class MCP(object):
    def __init__(self, F, a, b, *args,
                 transform='ssmooth',
                 maxit=100, maxsteps=10,
                 tol=SQEPS, showiters=False):

        if callable(F):
            self.f = lambda x: F(x, *args)
            self.islinear = False
        elif isinstance(F, np.ndarray):
            if len(args) == 1:
                self.q = q = args[0]
            else:
                raise ValueError('If problem is linear, exactly one argument (vector q) must be provided after the bounds')

            self.f = lambda x: (F.dot(x) + q, F)
            self.islinear = True

        a, b = np.atleast_1d(a, b)
        self.a, self.b = a.astype(float), b.astype(float)

        self.opts = {'transform': transform, 'maxit': maxit, 'maxsteps': maxsteps, 'tol': tol, 'showiters': showiters}


    def ssmooth(self, x, jac=False):

        x = np.atleast_1d(x)
        a, b, x = np.broadcast_arrays(self.a, self.b, x)

        ainf, binf = map(np.isinf, (a, b))


        fx, J = self.f(x)

        if jac:
            I = -np.identity(x.size)
            J = np.atleast_2d(J)
            fplus, Jplus = fischer(fx, a - x, J, I)
            fplus[ainf], Jplus[ainf] = fx[ainf], J[ainf]

            fhat, Jhat = fischer(fplus, b - x, Jplus, I, False)
            fhat[binf], Jhat[binf] = fplus[binf], Jplus[binf]
            return fhat, Jhat
        else:
            fplus = fischer(fx, a - x)   # apply fischer plus
            fplus[ainf] = fx[ainf]

            fhat = fischer(fplus, b - x, plus=False)
            fhat[binf] = fplus[binf]
            return fhat

    def minmax(self, x, jac=False):

        a, b = self.a, self.b
        x = np.atleast_1d(x)
        da, db = a - x, b - x

        fx, J = self.f(x)
        fhat = np.minimum(np.maximum(fx, da), db)

        if jac:
            J = np.atleast_2d(J)
            Jhat = -np.identity(x.size)
            i = (fx > da) & (fx < db)
            Jhat[i] = J[i]
            return fhat, Jhat
        else:
            return fhat


    def solve(self, x0=None, **kwargs):

        OPTS = self.opts.copy()
        for k, v in kwargs.items():
            if k in OPTS.keys():
                OPTS[k] = v
            else:
                print('Unknown option {}. Valid options are'.format(k), OPTS.keys())

        a, b = self.a, self.b
        Transform = self.ssmooth if OPTS['transform'].lower() == 'ssmooth' else self.minmax

        if x0 is None:
            x = self.q if self.islinear else (a + b) / 2
        else:
            x = np.atleast_1d(x0).astype(np.float64)

        if OPTS['showiters']:
            print('{:4}  {:4}  {:6}'.format('it', 'bstep', 'change'))

        for it in range(OPTS['maxit']):
            fx, J = Transform(x, True)
            fnorm = np.max(np.abs(fx))
            if fnorm < OPTS['tol']:
                break

            dx = - solve(J, fx)
            fnormold = np.inf

            for backstep in range(OPTS['maxsteps']):
                xnew = x + dx
                fxnew = Transform(xnew)
                fnormnew = np.max(np.abs(fxnew))
                if fnormnew < fnorm:
                    break
                if fnormold < fnormnew:
                    dx *= 2
                    break
                fnormold = fnormnew
                dx /= 2

            x += dx
            if OPTS['showiters']:
                print('{:4}  {:4}  {:6.2e}'.format(it, backstep, fnormnew))


        if (it + 1) == OPTS['maxit']:
            print('Failure to converge in MCP.solve after {} iterations, using the {} transformation'.format(it+1,OPTS['transform']))

        x = np.minimum(np.maximum(a, np.real(x)), b)
        return x, fx


def fischer(u, v, du=None, dv=None, plus=True):
    s = 1 if plus else -1
    sq = np.sqrt(u * u + v * v)
    f = u + v + s * sq

    if du is None or dv is None:
        return f

    nx = f.size
    nx1 = [f.size, 1]
    U, V, SQ = map(np.tile, (u, v, sq), (nx1, nx1, nx1))
    J = du + dv + s * (U * du + V * dv) / SQ
    return f, J


# def ssmooth(x, a, b, fx, J=None):
#     """
#     :param x: an evaluation point (vector)
#     :param a: lower bound
#     :param b: upper bound
#     :param fx: function values at x
#     :param J: Jacobian of f at x (optional
#     :return: fhat = vector, Jhat = matrix (if J provided)
#     """
#
#     x, a, b, fx = np.broadcast_arrays(*np.atleast_1d(x, a, b, fx))
#     ainf, binf = map(np.isinf, (a, b))
#     I = -np.identity(x.size)
#
#     if J is None:
#         fplus = fischer(fx, a - x)   # apply fischer plus
#         fplus[ainf] = fx[ainf]
#
#         fhat = fischer(fplus, b - x, plus=False)
#         fhat[binf] = fplus[binf]
#         return fhat
#     else:
#         J = np.atleast_2d(J)
#         fplus, Jplus = fischer(fx, a - x, J, I)
#         fplus[ainf], Jplus[ainf] = fx[ainf], J[ainf]
#
#         fhat, Jhat = fischer(fplus, b - x, Jplus, I, False)
#         fhat[binf], Jhat[binf] = fplus[binf], Jplus[binf]
#         return fhat, Jhat
#
#
# def minmax(x, a, b, fx, J = None):
#     """
#     :param x: an evaluation point (vector)
#     :param a: lower bound
#     :param b: upper bound
#     :param fx: function values at x
#     :param J: Jacobian of f at x (optional
#     :return: fhat = vector, Jhat = matrix (if J provided)
#     """
#
#     x, a, b, fx = np.broadcast_arrays(*np.atleast_1d(x, a, b, fx))
#     da, db = a - x, b - x
#     fhat = np.minimum(np.maximum(fx, da), db)
#
#     if J is None:
#         return fhat
#     else:
#         J = np.atleast_2d(J)
#
#     Jhat = -np.identity(x.size)
#     i = (fx > da) & (fx < db)
#     Jhat[i] = J[i]
#     return fhat, Jhat


def fixpoint(g, x, *args, maxit=100, tol=SQEPS, showiters=False):
    if showiters:
        print('{:^6}  {:^9}\n{}'.format('iter', 'dx', '-' * 18))

    for it in range(maxit):
        xnew = g(x, *args)
        dx = np.linalg.norm(xnew - x)
        if showiters:
            print('{:^6}   {:^9.3e}'.format(it, dx))

        if dx < tol:
            return xnew
        else:
            x = xnew.copy()

    print('Failure to converge in fixpoint, dx = {:.3e}'.format(dx))
    return xnew


# def ncpsolve(f, a, b, x=None, *args,
#              kind='ssmooth', method='newton',
#              maxit=100, maxsteps = 10,
#              tol=SQEPS, showiters=False):
#
#     Transform = ssmooth if kind.lower() == 'ssmooth' else minmax
#
#     if x is None:
#         x = (a + b)/2
#     a, b, x = map(np.atleast_1d, (a, b, x))
#
#     if showiters:
#         print('{:4}  {:4}  {:6}'.format('it', 'bstep', 'change'))
#
#     if method.lower() == 'newton':
#         for it in range(maxit):
#             fx, J = Transform(x, a, b, *f(x, *args))
#             fnorm = np.max(np.abs(fx))
#             if fnorm < tol:
#                 break
#
#             dx = - solve(J, fx)
#             fnormold = np.inf
#
#             for backstep in range(maxsteps):
#                 xnew = x + dx
#                 fxnew = Transform(xnew, a, b, f(xnew, *args)[0])
#                 fnormnew = np.max(np.abs(fxnew))
#                 if fnormnew < fnorm:
#                     break
#                 if fnormold < fnormnew:
#                     dx *= 2
#                     break
#                 fnormold = fnormnew
#                 dx /= 2
#
#             x = x + dx
#             if showiters:
#                 print('{:4}  {:4}  {:6.2e}'.format(it, backstep, fnormnew))
#
#
#         if (it + 1) == maxit:
#             print('Failure to converge in ncpsolve')
#
#         x = np.minimum(np.maximum(a, np.real(x)), b)
#         return x, fx
#
#     else:
#         if isinstance(f(x, *args), tuple):
#             xstar = fixpoint(lambda y: y + Transform(y, a, b, f(y, *args)[0]), x)
#         else:
#             xstar = fixpoint(lambda y: y + Transform(y, a, b, f(y, *args)), x)
#         return xstar, xstar - x
#
#
#
# def lcpsolve(M, q, a, b, x=None,
#              kind='ssmooth', method='newton',
#              maxit=100, maxsteps = 20,
#              tol=SQEPS, showiters=False):
#
#     Transform = ssmooth if kind.lower() == 'ssmooth' else minmax
#     if x is None:
#         x = q
#
#     if method.lower() == 'newton':
#         for it in range(maxit):
#             z = np.dot(M, x) + q
#             fx, J = Transform(x, a, b, z, M)
#             fnorm = np.max(np.abs(fx))
#             if fnorm < tol:
#                 break
#
#             dx = - solve(J, fx)
#             fnormold = np.inf
#
#             for backstep in range(maxsteps):
#                 xnew = x + dx
#                 znew = np.dot(M, xnew) + q
#                 fxnew = Transform(xnew, a, b, znew)
#                 fnormnew = np.max(np.abs(fxnew))
#                 if fnormnew < fnorm:
#                     break
#                 if fnormold < fnormnew:
#                     dx *= 2
#                     break
#                 fnormold = fnormnew
#                 dx /= 2
#
#             x = x + dx
#             if showiters:
#                 print('{:4}  {:4}  {:6.2f}'.format(it, backstep, fnormnew))
#
#
#         if (it + 1) == maxit:
#             warn('Failure to converge in ncpsolve')
#
#         x = np.minimum(np.maximum(a, np.real(x)), b)
#         return x, np.dot(M, x) + q
#
#     else:
#         raise NotImplementedError
#         #return fixpoint(lambda y: Transform(y, a, b, M), x)