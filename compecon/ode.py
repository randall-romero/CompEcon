# This module has tools to solve ordinary differential equations

# The ODE class defined in this module combines the following function from original MATLAB version
#    * odecol
#    * oderk4
#    * odespx
#    * odephase

# TODO: Add a OCmodel class, that combines docsolve and socsolve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compecon import BasisChebyshev, BasisSpline, NLP, jacobian


class ODE:
    def __init__(self, f, T, bv, *params):
        self.f = lambda x: f(x, *params)
        self.T = T
        self.bv = bv
        self._d = len(self.bv)
        self.fsol = None
        self.xspx = None

    def solve_collocation(self, *, n=100, bt=None, bx=None, btype='cheb', c=None, nf=10):
        if bt is None:
            bt = np.zeros_like(self.bv)
        if bx is None:
            bx = np.arange(len(self.bv))

        basis = BasisChebyshev if btype.lower() == 'cheb' else BasisSpline

        # compute collocation nodes
        t = basis(n - 1, 0, T).nodes

        # Approximation structure
        labels = [f'$x_{j}$' for j in range(self._d)]
        self.fsol = basis(n, 0, T, l=labels, labels=['Time'])  # falta inicializar los coeficientes

        # residual function for nonlinear problem formulation
        def residual(c):
            # reshape coefficient vector
            self.fsol.c = c.reshape(self._d, n)

            # compute residuals at nodal times
            x = self.fsol(t)
            dx = self.fsol(t, 1)
            r = dx - self.f(x)

            # compute residuals at boundaries
            x = self.fsol(bt)
            b = np.array([x[j, bx[j]] - self.bv[j] for j in range(self._d)])
            b = np.atleast_2d(b).T
            return np.c_[r, b].flatten()

        # Solve the nonlinear system
        self.fsol.c = NLP(residual).broyden(x0=self.fsol.c.flatten()).reshape(self._d, n)

        # Compute solution at plotting nodes
        if nf > 0:
            m = int(nf) * n
            t = np.linspace(0, T, m)
        else:
            t = t.flatten()

        x = self.fsol(t)

        # Compute residual
        dx = self.fsol(t, 1)
        resid = dx - self.f(x)

        self.x = pd.DataFrame(x.T, index=t, columns=labels)
        self.resid = pd.DataFrame(resid.T, index=t, columns=labels)

    def rk4(self, n=1000, xnames=None):
        t = np.linspace(0, self.T, n)
        x0 = np.asarray(self.bv)

        x = np.zeros((n, self._d), float)

        x[0] = x0
        h = np.r_[0, t[1:] - t[:-1]]
        for i, hh in enumerate(h[1:], start=1):
            f1 = self.f(x0) * (hh / 2)
            f2 = self.f(x0 + f1) * hh
            f3 = self.f(x0 + f2 / 2) * hh
            f4 = self.f(x0 + f3) * (hh / 2)
            x0 = x0 + (f1 + f2 + f3 + f4) / 3
            x[i] = x0

        self.x = pd.DataFrame(x, index=t, columns=xnames)
        # self.resid = pd.DataFrame(resid.T, index=t, columns=labels)

    def spx(self, x0=None, T=None, n=1000):

        f = self.f
        if x0 is None:
            x0 = self.bv

        if T is None:
            T = self.T

        J = jacobian(f, x0)
        D, V = np.linalg.eig(J)
        i = D < 0
        j = D > 0

        # if np.iscomplex(D).any() or i.any():
        #    print('x is not saddle point or stable steady-state')   #todo review, make warning
        #    x = []
        #    return x

        i = np.argmin(D)
        delx = 0.0001 * V[:, i]  # eigenvector asociated with minimum eigenvalue
        t = np.linspace(0, -T, n)
        h = t[1:] - t[:-1]

        # xsp
        xsp = np.zeros((n, 2))
        x = x0 + delx

        xsp[0] = x
        for i, hh in enumerate(h, start=1):
            f1 = f(x) * (hh / 2)
            f2 = f(x + f1) * hh
            f3 = f(x + f2 / 2) * hh
            f4 = f(x + f3) * (hh / 2)
            x += (f1 + f2 + f3 + f4) / 3
            xsp[i] = x

        # xsp(i+1:n,:) = []; parece innecesario
        xsp = np.real(xsp)

        # xsn
        xsn = np.zeros((n, 2))
        x = x0 - delx

        xsp[0] = x
        for i, hh in enumerate(h, start=1):
            f1 = f(x) * (hh / 2)
            f2 = f(x + f1) * hh
            f3 = f(x + f2 / 2) * hh
            f4 = f(x + f3) * (hh / 2)
            x += (f1 + f2 + f3 + f4) / 3
            xsn[i] = x

        # xsn(i+1:n,:) = []; parece innecesario
        xsn = np.real(xsn)

        x = np.r_[xsn[::-1], np.atleast_2d(x0), xsp]
        j = np.isfinite(x).all(axis=1)

        self.xspx = x[j].T

    def phase(self, x1lim, x2lim, *, x=None, xstst=None, xnulls=None, ax=None, **ax_kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        ax.set(xlim=x1lim, ylim=x2lim, **ax_kwargs)

        # Plot Velocity Field
        x1 = np.linspace(*x1lim, 15)
        x2 = np.linspace(*x2lim, 15)

        xg = gridmake(x1, x2)
        v = np.real(self.f(xg))
        ax.quiver(*xg, *v, color='0.7')  # , ls=1,ms=2, )

        # Plot Separatrix
        if self.xspx is not None:
            ax.plot(*self.xspx, color='0.2', ls='--', lw=2, label='Separatrix')

        # Plot Nullclines
        if xnulls is not None:
            xnulls.plot(ax=ax, lw=2)

        # Plot State Path
        if x is None:
            x = self.x.values.T

        if np.ndim(x) < 3:
            x = np.rollaxis(np.atleast_3d(x), 2)  # ejemplo, variable, tiempo
        n = x.shape[-1]
        m = max(1, np.floor(n / 100))

        for k in range(x.shape[0]):
            ax.plot(x[k, 0, 0], x[k, 1, 0], marker='o', ms=10, color='C2')
            # if nargin<12:
            #    for j=m+1:n
            #        if mod(j-1,m)==0:
            #            plot(x(j-m:j,1,k),x(j-m:j,2,k),'r','LineWidth',2)
            #            getframe;
            #            if x(j-m,1,k)<x1lim(1)||x(j-m,1,k)>x1lim(2)||x(j-m,2,k)<x2lim(1)||x(j-m,2,k)>x2lim(2):
            #                break
            # else:

            ax.plot(*x[k], color='C2', lw=2)

        # Plot Steady State
        if xstst is not None:
            for xss in np.atleast_2d(xstst):
                ax.plot(*xss, color='C2', marker='*', ms=12)

        ax.legend(loc='lower center', ncol=3)
        return ax