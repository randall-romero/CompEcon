#!/usr/bin/env python
# coding: utf-8

# # Production Management Model
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demdp01.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# **WARNING** This demo is not running. Problem with dpmodel.
# 
# TODO: Fix error in dpmodel.
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## About
# 
# Profit maximizing entrepeneur must decide how much to produce, subject to production adjustment costs.
# 
# - States
#     -     i       market price (discrete)
#     -     s       lagged production (continuous)
# - Actions
#     -     x       current production
# - Parameters
#     - $\alpha$    -- marginal adjustment cost
#     - $\beta$     -- marginal production cost parameters
#     - pbar       -- long-run average market price 
#     - $\mu$      -- mean log price
#     - $\sigma$     -- market price shock standard deviation
#     - $\delta$     -- discount factor
#    
#     

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from compecon import BasisSpline, DPmodel, DPoptions, qnwlogn, demo, BasisChebyshev
import seaborn as sns
import pandas as pd


# ### Model parameters
# 

# In[2]:


α, β0, β1, pbar = 0.01, 0.8, 0.03, 1.0 
σ, δ = 0.2, 0.9
μ = np.log(pbar) - σ**2 / 2


# In[3]:


α, β0, β1


# ### Continuous state shock distribution

# In[4]:


m = 3  #number of market price shocks
p, w = qnwlogn(m, μ, σ**2) 
q = np.repeat(w,3).reshape(3,3).T


# ### State space
# The state variable is s="lagged production", which we restrict to $s\in[0, 20]$. 
# 
# Here, we represent it with a cubic spline basis, with $n=50$ nodes.

# In[5]:


n, smin, smax = 5, 0.0, 20.0
basis = BasisChebyshev(n, smin, smax, labels=['lagged production'])


# The discrete state is given by the price *p*

# In[6]:


prices = ['p_low', 'p_mid', 'p_high']


# ### Action space
# The choice variable x="current production" must be nonnegative.

# In[7]:


def bounds(s, i, j=None):
    return np.zeros_like(s), np.inf + np.zeros_like(s)


# ### Reward function
# The reward function is 

# In[8]:


def reward(s, q, i, j=None):
    u = p[i]*q - (β0*q + 0.5*β1*q**2) - 0.5*α*((q-s)**2)
    ux = p[i] - β0 - β1*q - α*(q-s)
    uxx = (-β1-α)*np.ones_like(s)    
    return u, ux, uxx


# ### State transition function
# Next period, reservoir level wealth will be equal to current level minus irrigation plus random rainfall:

# In[9]:


def transition(s, q, i, j=None, in_=None, e=None):
    g = q
    gx = np.ones_like(q)
    gxx = np.zeros_like(q)
    return g, gx, gxx


# ### Model structure
# # TODO:  CORREGIR ESTA ECUACION
# 
# The value of wealth $s$ satisfies the Bellman equation 
# \begin{equation*}
# V(s) = \max_x\left\{\left(\frac{a_0}{1+a_1}\right)x^{1+a1} + \left(\frac{b_0}{1+b_1}\right)(s-x)^{1+b1}+ \delta V(s-x+e)  \right\}
# \end{equation*}
# 
# To solve and simulate this model,use the CompEcon class `DPmodel`

# In[10]:


firm = DPmodel(basis, reward, transition, bounds,q=q,                      
               i=prices, x=['Production'],discount=δ )


# In[11]:


firm


# ## Solving the model

# Solving the growth model by collocation. 

# In[12]:


S = firm.solve()
S.head()


# In[ ]:


firm.Policy_j(firm.Policy.nodes,dropdim=True).shape


# `DPmodel.solve` returns a pandas `DataFrame` with the following data:

# We are also interested in the shadow price of wealth (the first derivative of the value function).

# In[ ]:


S['shadow price'] = water_model.Value(S['Reservoir'],1)
S.head()


# ## Plotting the results

# ### Optimal Policy

# In[ ]:


fig1 = demo.figure('Optimal Irrigation Policy', 'Reservoir Level', 'Irrigation')
plt.plot(S['Irrigation'])
demo.annotate(sstar, xstar,f'$s^*$ = {sstar:.2f}\n$x^*$ = {xstar:.2f}','bo', (10, -6),ms=10,fs=11)


# ### Value Function

# In[ ]:


fig2 = demo.figure('Value Function', 'Reservoir Level', 'Value')
plt.plot(S['value'])


# ### Shadow Price Function

# In[ ]:


fig3 = demo.figure('Shadow Price Function', 'Reservoir Level', 'Shadow Price')
plt.plot(S['shadow price'])


# ### Chebychev Collocation Residual

# In[ ]:


fig4 = demo.figure('Bellman Equation Residual', 'Reservoir Level', 'Residual')
plt.hlines(0,smin,smax,'k',linestyles='--')
plt.plot(S[['resid']])
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,1))


# ## Simulating the model
# 
# We simulate 21 periods of the model starting from $s=s_{\min}$

# In[ ]:


T = 31
nrep = 100_000
data = water_model.simulate(T, np.tile(smin,nrep))


# ### Simulated State and Policy Paths

# In[ ]:


subdata = data[data['_rep'].isin(range(3))]
opts = dict(spec='r*', offset=(0, -15), fs=11, ha='right')


# In[ ]:


fig6 = demo.figure('Simulated and Expected Reservoir Level','Year', 'Reservoir Level',[0, T + 0.5])
plt.plot(data[['time','Reservoir']].groupby('time').mean())
plt.plot(subdata.pivot('time','_rep','Reservoir'),lw=1)
demo.annotate(T, sstar, f'steady-state reservoir\n = {sstar:.2f}', **opts)


# In[ ]:


fig7 = demo.figure('Simulated and Expected Irrigation','Year', 'Irrigation',[0, T + 0.5])
plt.plot(data[['time','Irrigation']].groupby('time').mean())
plt.plot(subdata.pivot('time','_rep','Irrigation'),lw=1)
demo.annotate(T, xstar, f'steady-state irrigation\n = {xstar:.2f}', **opts)


# ### Ergodic Wealth Distribution

# In[ ]:


subdata = data[data['time']==T][['Reservoir','Irrigation']]
stats =pd.DataFrame({'Deterministic Steady-State': [sstar, xstar],
              'Ergodic Means': subdata.mean(),
              'Ergodic Standard Deviations': subdata.std()}).T
stats


# In[ ]:


fig8 = demo.figure('Ergodic Reservoir and Irrigation Distribution','Wealth','Probability')
sns.kdeplot(subdata['Reservoir'])
sns.kdeplot(subdata['Irrigation'])


# In[ ]:


#demo.savefig([fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8])

