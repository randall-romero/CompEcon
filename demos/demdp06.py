import numpy as np
from numpy import log
from compecon import Basis, DPmodel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

class growth_model(DPmodel):
    def __init__(self, basis):
        DPmodel.__init__(self, basis,
                         ni=1,
                         nj=1,
                         dx=1,
                         discount=0.9)
        self.beta  = 0.7
        s = basis.nodes
        vtrue = self.vstar + self.b * log(s / self.sstar)
        ktrue = self.time.discount * s

        self.Value_j.y += vtrue
        self.Policy_j.y += ktrue
        self.update_value_function()

    @property
    def sstar(self):  # steady-state wealth
        beta = self.beta
        d = self.time.discount
        return((beta * d) ** (beta/(1-beta)))

    @property
    def kstar(self):  # steady-state capital investment
        beta = self.beta
        d = self.time.discount
        return(beta * d * self.sstar)

    @property
    def vstar(self):  # steady-state value
        return(log(self.sstar-self.kstar)/(1-self.time.discount))

    @property
    def pstar(self):  # steady-state shadow price
        return(1 / (self.sstar * (1 - self.beta * self.time.discout)))

    @property
    def b(self):
        return(1 / (1 - self.time.discount * self.beta))

    def bounds(self,s,i,j):
        n = len(s)
        lowerBound = np.zeros([n])
        upperBound = s
        return(lowerBound, upperBound)

    def reward(self,s,k,i,j):
        sk = s - k
        f = log(sk)
        Df = - sk ** -1
        D2f = - sk ** -2
        return(f, Df, D2f)

    def transition(self,s,k,i,j, inext,e):
        beta = self.beta
        g = k ** beta
        Dg = beta * k ** (beta - 1)
        D2g = (beta - 1) * beta * k ** (beta - 2)
        return(g, Dg, D2g)

n     = 25                             # number of collocation nodes
smin  = 0.2                            # minimum wealth
smax  = 1.0                            # maximum wealth
BASIS = Basis(n,smin,smax)             # basis functions

M = growth_model(BASIS)

M.solve()