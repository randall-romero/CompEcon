#!/usr/bin/env python
# coding: utf-8

# # Deterministic Optimal Economic Growth Model
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demdp06.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## About
# 
# Welfare maximizing social planner must decide how much society should consume and invest.  Model is of special interest because it has a known closed-form solution.
# 
# - States
#     -     s       stock of wealth
# - Actions
#     -     k       capital investment
# - Parameters
#     -     beta    capital production elasticity
#     -     delta   discount factor

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from compecon import BasisChebyshev, DPmodel, DPoptions, qnwnorm, demo


# ### Model parameters
# 
# Assuming that the marginal productivity of capital is $\beta=0.5$ and the discount factor is $\delta=0.9$

# In[2]:


β, δ = 0.5, 0.9


# ## Analytic results
# 
# The steady-state values for this model are

# In[3]:


sstar = (β * δ) ** (β / (1 - β))   # steady-state wealth
kstar = β * δ * sstar                    # steady-state capital investment
vstar = np.log(sstar - kstar) / (1 - δ)     # steady-state value
pstar = 1 / (sstar * (1 - β * δ))        # steady-state shadow price
b = 1 / (1 - δ * β)

print('\n\nSteady-State')
for var, value in zip(['Wealth','Investment','Value','Shadow price'], [sstar,kstar,vstar,pstar]):
    print(f'\t{var:12s} = {value:8.4f}')


# The true value function is
# \begin{equation}
# V(s) = v^* + \frac{1}{1-\delta\beta}\left(\log(s) -\log(s^*)\right)
# \end{equation}

# In[4]:


def vtrue(wealth): # analytic value function
    return vstar + b * (np.log(wealth) - np.log(sstar))


# The true policy function is
# \begin{equation}
# k(s) = \delta\beta s
# \end{equation}

# In[5]:


def ktrue(wealth): #analytic policy function
    return δ*β*wealth


# ## Numeric results

# ### State space
# The state variable is s="Wealth", which we restrict to $0\in[0.2, 1.0]$. 
# 
# Here, we represent it with a Chebyshev basis, with $n=15$ nodes.

# In[6]:


n, smin, smax = 15, 0.2, 1.0
basis = BasisChebyshev(n, smin, smax, labels=['Wealth'])


# ### Action space
# The choice variable k="Investment" must be nonnegative.

# In[7]:


def bounds(s, i=None, j=None):
    return np.zeros_like(s), s[:]


# ### Reward function
# The reward function is the utility of consumption=$s-k$.

# In[8]:


def reward(s, k, i=None, j=None):
    sk = s - k
    u = np.log(sk)
    ux= - sk ** -1
    uxx = - sk ** -2
    return u, ux, uxx


# ### State transition function
# Next period, wealth will be equal to production from available initial capital $k$, that is $s' = k^\beta$

# In[9]:


def transition(s, k, i=None, j=None, in_=None, e=None):
    g = k ** β
    gx = β * k **(β - 1)
    gxx = (β - 1) * β * k ** (β - 2)
    return g, gx, gxx


# ### Model structure
# The value of wealth $s$ satisfies the Bellman equation
# \begin{equation*}
# V(s) = \max_k\left\{\log(s-k) + \delta V(k^\beta)  \right\}
# \end{equation*}
# 
# To solve and simulate this model,use the CompEcon class `DPmodel`

# In[10]:


growth_model = DPmodel(basis, reward, transition, bounds,
                       x=['Investment'],
                       discount=δ)


# ### Solving the model
# 
# Solving the growth model by collocation, using *Newton* algorithm and a maximum of 20 iterations

# In[11]:


options = dict(show=True,
               algorithm='newton',
               maxit=20)

snodes = growth_model.Value.nodes
S = growth_model.solve(vtrue(snodes), ktrue(snodes), **options)


# `DPmodel.solve` returns a pandas `DataFrame` with the following data:

# In[12]:


S.head()


# We are also interested in the shadow price of wealth (the first derivative of the value function) and the approximation error.
# 
# To analyze the dynamics of the model, it also helps to compute the optimal change of wealth.

# In[13]:


S['shadow price'] = growth_model.Value(S['Wealth'],1)
S['error'] = S['value'] - vtrue(S['Wealth'])
S['D.Wealth'] = transition(S['Wealth'], S['Investment'])[0] - S['Wealth']
S.head()


# ### Solving the model by Linear-Quadratic Approximation
# 
# The `DPmodel.lqapprox` solves the linear-quadratic approximation, in this case arround the steady-state. It returns a LQmodel which works similar to the DPmodel object.
# 
# We also compute the shadow price and the approximation error to compare these results to the collocation results.

# In[14]:


growth_lq = growth_model.lqapprox(sstar, kstar)
L = growth_lq.solution(basis.nodes)
L['shadow price'] = L['value_Wealth']
L['error'] = L['value'] - vtrue(L['Wealth'])
L['D.Wealth'] = L['Wealth_next']- L['Wealth']
L.head()


# In[15]:


growth_lq.G


# ## Plotting the results

# ### Optimal Policy

# In[16]:


fig1 = demo.figure('Optimal Investment Policy', 'Wealth', 'Investment')
plt.plot(S.Investment, label='Chebychev Collocation')
plt.plot(L.Investment, label='L-Q Approximation')
demo.annotate(sstar, kstar,'$s^*$ = %.2f\n$V^*$ = %.2f' % (sstar, kstar),'bo', (10, -17),ms=10)
plt.legend(loc= 'upper left')


# ### Value Function

# In[17]:


fig2 = demo.figure('Value Function', 'Wealth', 'Value')
plt.plot(S.value, label='Chebychev Collocation')
plt.plot(L.value, label='L-Q Approximation')

demo.annotate(sstar, vstar, f'$s^* = {sstar:.2f}$\n$V^* = {vstar:.2f}$', 'bo', (10, -17),ms=10)
plt.legend(loc= 'upper left')


# ### Shadow Price Function

# In[18]:


fig3 = demo.figure('Shadow Price Function', 'Wealth', 'Shadow Price')
plt.plot(S['shadow price'], label='Chebychev Collocation')
plt.plot(L['shadow price'], label='L-Q Approximation')
demo.annotate(sstar, pstar,f'$s^* = {sstar:.2f}$\n$\lambda^* = {pstar:.2f}$', 'bo', (10, 17),ms=10)
plt.legend(loc= 'upper right')


# ### Chebychev Collocation Residual and Approximation Error vs. Linear-Quadratic Approximation Error

# In[19]:


fig4 = plt.figure(figsize=[12, 6])
demo.subplot(1, 2, 1, 'Chebychev Collocation Residual\nand Approximation Error', 'Wealth', 'Residual/Error')
plt.hlines(0,smin,smax,'k',linestyles='--')
plt.plot(S[['resid', 'error']])
plt.legend(['Residual','Error'], loc='lower right')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,1))

demo.subplot(1, 2, 2, 'Linear-Quadratic Approximation Error', 'Wealth', 'Error')
plt.hlines(0,smin,smax,'k',linestyles='--')
plt.plot(L['error'], label='Error')
plt.legend(loc='upper left')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,1))


# ### Wealth dynamics
# 
# Notice how the steady-state is stable in the Chebyshev collocation solution, but unstable in the linear-quadratic approximation. In particular, simulated paths of wealth in the L-Q approximation will converge to zero, unless the initial states is within a small neighborhood of the steady-state.

# In[20]:


fig5 = demo.figure('Wealth dynamics', 'Wealth', 'Wealth change', figsize=[8,5])
plt.plot(S['D.Wealth'], label='Chebychev Collocation')
plt.plot(L['D.Wealth'], label='L-Q Approximation')
plt.hlines(0,smin,smax,linestyles=':')

demo.annotate(sstar, 0, f'$s^* = {sstar:.2f}$\n$\Delta s^* = {0:.2f}$', 'bo', (10, 10),ms=10,fs=11)
plt.legend(loc= 'lower left')


# ## Simulating the model
# 
# We simulate 20 periods of the model starting from $s=s_{\min}$

# In[21]:


T = 20
data = growth_model.simulate(T, smin)


# ### Simulated State and Policy Paths

# In[22]:


opts = dict(spec='r*', offset=(-5, -5), fs=11, ha='right')

fig6 = demo.figure('State and Policy Paths','Period', 'Wealth/Investment',[0, T + 0.5])
plt.plot(data[['Wealth', 'Investment']])
demo.annotate(T, sstar, 'steady-state wealth\n = %.2f' % sstar, **opts)
demo.annotate(T, kstar, 'steady-state investment\n = %.2f' % kstar, **opts)


# In[23]:


#demo.savefig([fig1,fig2,fig3,fig4,fig5,fig6])

