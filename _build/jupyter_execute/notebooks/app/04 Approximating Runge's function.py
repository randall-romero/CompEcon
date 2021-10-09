#!/usr/bin/env python
# coding: utf-8

# # Approximating Runge's function
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demapp04.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## About
# 
# Uniform-node and Chebyshev-node polynomial approximation of Runge's function and compute condition numbers of associated interpolation matrices

# ## Initial tasks

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, cond
from compecon import BasisChebyshev, demo
import warnings

warnings.simplefilter('ignore')


# ### Runge function

# In[2]:


runge = lambda x: 1 / (1 + 25 * x ** 2)


# Set points of approximation interval

# In[3]:


a, b = -1, 1


# Construct plotting grid

# In[4]:


nplot = 1001
x = np.linspace(a, b, nplot)
y = runge(x)


# ## Plot Runge's Function

# Initialize data matrices

# In[5]:


n = np.arange(3, 33, 2)
nn = n.size
errunif, errcheb = (np.zeros([nn, nplot]) for k in range(2))
nrmunif, nrmcheb, conunif, concheb = (np.zeros(nn) for k in range(4))


# Compute approximation errors on refined grid and interpolation matrix condition numbers

# In[6]:


for i in range(nn):
    # Uniform-node monomial-basis approximant
    xnodes = np.linspace(a, b, n[i])
    c = np.polyfit(xnodes, runge(xnodes), n[i])
    yfit = np.polyval(c, x)
    phi = xnodes.reshape(-1, 1) ** np.arange(n[i])

    errunif[i] = yfit - y
    nrmunif[i] = np.log10(norm(yfit - y, np.inf))
    conunif[i] = np.log10(cond(phi, 2))

    # Chebychev-node Chebychev-basis approximant
    yapprox = BasisChebyshev(n[i], a, b, f=runge)
    yfit = yapprox(x)  # [0] no longer needed?  # index zero is to eliminate one dimension
    phi = yapprox.Phi()
    errcheb[i] = yfit - y
    nrmcheb[i] = np.log10(norm(yfit - y, np.inf))
    concheb[i] = np.log10(cond(phi, 2))


# Plot Chebychev- and uniform node polynomial approximation errors

# In[7]:


figs = []
fig1, ax = plt.subplots()
ax.plot(x, y)
ax.text(-0.8, 0.8, r'$y = \frac{1}{1+25x^2}$', fontsize=18)
ax.set(xticks=[], title="Runge's Function", xlabel='', ylabel='y');
figs.append(fig1)


# In[8]:


fig2, ax = plt.subplots()
ax.hlines(0, a, b, 'gray', '--')
ax.plot(x, errcheb[4], label='Chebychev Nodes')
ax.plot(x, errunif[4], label='Uniform Nodes')
ax.legend(loc='upper center')
ax.set(title="Runge's Function $11^{th}$-Degree\nPolynomial Approximation Error.", xlabel='x', ylabel='Error')
figs.append(fig2)


# Plot approximation error per degree of approximation

# In[9]:


fig3, ax = plt.subplots()
ax.plot(n, nrmcheb, label='Chebychev Nodes')
ax.plot(n, nrmunif, label='Uniform Nodes')
ax.legend(loc='upper left')
ax.set(title="Log10 Polynomial Approximation Error for Runge's Function",xlabel='', ylabel='Log10 Error', xticks=[])
figs.append(fig3)


# In[10]:


fig4, ax = plt.subplots()
ax.plot(n, concheb, label='Chebychev Polynomial Basis')
ax.plot(n, conunif, label='Mononomial Basis')
ax.legend(loc='upper left')
ax.set(title="Log10 Interpolation Matrix Condition Number",
       xlabel='Degree of Approximating Polynomial',
       ylabel='Log10 Condition Number')
figs.append(fig4)


# ### Save all figures to disc

# In[11]:


#demo.savefig(figs, name='demapp04')

