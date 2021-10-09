#!/usr/bin/env python
# coding: utf-8

# # Monopolist's Effective Supply Function
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demapp10.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## Initial tasks

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from compecon import BasisChebyshev, NLP, demo


# ### Residual Function

# In[2]:


def resid(c):
    Q.c = c
    q = Q(p)
    marginal_income = p + q / (-3.5 * p **(-4.5))
    marginal_cost = np.sqrt(q) + q ** 2
    return  marginal_income - marginal_cost 


# ### Approximation structure

# In[3]:


n, a, b = 21, 0.5, 2.5
Q = BasisChebyshev(n, a, b)
c0 = np.zeros(n)
c0[0] = 2
p = Q.nodes


# ### Solve for effective supply function

# In[4]:


monopoly = NLP(resid)
Q.c = monopoly.broyden(c0)


# ### Plot effective supply

# In[5]:


nplot = 1000
p = np.linspace(a, b, nplot)
rplot = resid(Q.c)


# In[6]:


fig1, ax = plt.subplots()
ax.set(title="Monopolist's Effective Supply Curve",
       xlabel='Quantity', 
       ylabel='Price')
ax.plot(Q(p), p);


# ### Plot residual

# In[7]:


fig2, ax = plt.subplots()
ax.set(title='Functional Equation Residual',
       xlabel='Price',
       ylabel='Residual')
ax.hlines(0, a, b, 'k', '--')
ax.plot(p, rplot);


# ### Save all figures to disc

# In[8]:


#demo.savefig([fig1, fig2], name='demapp10')

