
# coding: utf-8

# ### DEMAPP07 

# # Solve Cournot oligopoly model via collocation

# To illustrate implementation of the collocation method for implicit function problems, consider the example of Cournot oligopoly. In the standard microeconomic model of the firm, the firm maximizes profit by equating marginal revenue to marginal cost (MC). An oligopolistic firm, recognizing that its actions affect price, takes the marginal revenue to be $p + q\frac{dp}{dq}$, where $p$ is price, $q$ is quantity produced, and $\frac{dp}{dq}$ is the marginal impact of output on market price. The Cournot assumption is that the firm acts as if any change in its output will be unmatched by its competitors. This implies that
# 
# \begin{equation}
#     \frac{dp}{dq} = \frac{1}{D'(p)}
# \end{equation}
# 
# where $D(p)$ is the market demand curve.
# 
# Suppose we wish to derive the effective supply function for the firm, which specifies
# the quantity $q = S(p)$ it will supply at any price. The firm's effective supply function is
# characterized by the functional equation
# 
# \begin{equation}
#     p + \frac{S(p)}{D'(p)} - MC(S(p)) = 0
# \end{equation}
# 
# for all positive prices $p$. In simple cases, this function can be found explicitly.  However,
# in more complicated cases, no explicit solution exists.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from compecon import BasisChebyshev, NLP, demo


# ### Model parameters
# 
# Here, the demand elasticity and the marginal cost function parameter are

# In[2]:


alpha, eta = 1.0, 3.5


# In[3]:


D = lambda p: p**(-eta)


# ### Approximation structure
# 
# A degree-25 Chebychev basis on the interval [0.5, 3.0] is selected; also, the associated collocation nodes `p` are computed.

# In[4]:


n, a, b =  25, 0.5, 2.0
S = BasisChebyshev(n, a, b, labels=['price'], y=np.ones(n))
p = S.nodes


# In[5]:


S2 = BasisChebyshev(n, a, b, labels=['price'], l=['supply'])
S2.y = np.ones_like(p)


# ### Residual function
# 
# Suppose, for example, that
# 
# \begin{equation}
#     D(p) = p^{-\eta} \quad\text{and}\quad MC(q) = \alpha\sqrt{q} + q^2
# \end{equation}
# 
# Then the functional equation to be solved for S(p),
# 
# \begin{equation}
#  \left[p - \frac{S(p)p^{\eta+1}}{\eta}\right] -\left[\alpha\sqrt{S(p)} + S(p)^2\right] = 0
# \end{equation}
# 
# has no known closed-form solution.

# In[6]:


def resid(c):
    S.c = c  # update interpolation coefficients
    q = S(p) # compute quantity supplied at price nodes
    return p - q * (p ** (eta+1) / eta) - alpha * np.sqrt(q) - q ** 2


# Notice that `resid` only takes one argument. The other parameters (`Q`, `p`, `eta`, `alpha`) should be declared as such in the main script, were Python's scoping rules will find them.

# ### Solve for effective supply function
# 
# Class `NLP` defines nonlinear problems. It can be used to solve `resid` by Broyden's method.

# In[7]:


cournot = NLP(resid)
S.c = cournot.broyden(S.c, tol=1e-12)


# ### Plot demand and effective supply for m=5 firms

# In[8]:


prices = np.linspace(a, b, 501)
fig1 = demo.figure('Cournot Effective Firm Supply Function', 
            'Quantity', 'Price', [0, 4], [0.5, 2])
plt.plot(5 * S(prices), prices, D(prices), prices)
plt.legend(('Supply','Demand'))


# ### Plot residual
# 
# Notice that `resid` does not take explicit parameters, so to evaluate it when prices are `prices` we need to assign `p = prices`.

# In order to assess the quality of the approximation, one plots the residual function over the approximation domain. Here, the residual function is plotted by computing the residual at a refined grid of 501 equally spaced points.

# In[9]:


p = prices
fig2 = demo.figure('Residual Function for Cournot Problem',
            'Quantity', 'Residual')
plt.hlines(0, a, b, 'k', '--', lw=2)
plt.plot(prices, resid(S.c))


# ### Plot demand and effective supply for m=1, 3, 5, 10, 15, 20 firms

# In[10]:


fig3 = demo.figure('Industry Supply and Demand Functions', 
            'Quantity', 'Price', [0, 12], figsize=[9,4])

lcolor = [z['color']  for z in plt.rcParams['axes.prop_cycle']]

for i, m in enumerate([1, 3, 5, 10, 15, 20]):
    plt.plot(m*S(prices), prices) # supply
    demo.annotate(m*S(1.2)-0.25,1.4-i/12,f'm={m:d}',color=lcolor[i],ms=0,fs=12)
    
plt.plot(D(prices), prices, linewidth=4, color='grey') # demand
demo.annotate(10,0.5,'demand',color='grey', ms=0, fs=12)


# ### Plot equilibrium price as a function of number of firms m

# In[14]:


pp = (b + a) / 2
dp = (b - a) / 2
m  = np.arange(1, 26)
for i in range(50):
    dp /= 2
    pp = pp - np.sign(S(pp) * m - pp ** (-eta)) * dp

fig4 = demo.figure('Cournot Equilibrium Price as Function of Industry Size', 
            'Number of Firms', 'Price')
plt.plot(m, pp)


# In[12]:


demo.savefig([fig1,fig2,fig3,fig4])

