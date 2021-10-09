#!/usr/bin/env python
# coding: utf-8

# # Constrained optimization using scipy
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demopt08.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## About
# 
# The problem is
# 
# \begin{equation*}
# \max\{-x_0^2 - (x_1-1)^2 - 3x_0 + 2\}
# \end{equation*}
# 
# subject to
# 
# \begin{align*}
# 4x_0 + x_1 &\leq 0.5\\
# x_0^2 + x_0x_1 &\leq 2.0\\
# x_0 &\geq 0 \\
# x_1 &\geq 0
# \end{align*}

# ## Using scipy
# 
# The **scipy.optimize.minimize** function minimizes functions subject to equality constraints, inequality constraints, and bounds on the choice variables. 

# In[1]:


import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=4,suppress=True)


# * First, we define the objective function, changing its sign so we can minimize it

# In[2]:


def f(x):
    return x[0]**2 + (x[1]-1)**2 + 3*x[0] - 2


# * Second, we specify the inequality constraints using a tuple of two dictionaries (one per constraint), writing each of them in the form $g_i(x) \geq 0$, that is
# \begin{align*}
# 0.5 - 4x_0 - x_1 &\geq 0\\
# 2.0 - x_0^2 - x_0x_1 &\geq 0
# \end{align*}

# In[3]:


cons = ({'type': 'ineq', 'fun': lambda x: 0.5 - 4*x[0] - x[1]},
       {'type': 'ineq', 'fun': lambda x: 2.0 - x[0]**2 - x[0]*x[1]})


# * Third, we specify the bounds on $x$:
# \begin{align*}
# 0 &\leq x_0 \leq \infty\\
# 0 &\leq x_1 \leq \infty
# \end{align*}

# In[4]:


bnds = ((0, None), (0, None))


# * Finally, we minimize the problem, using the SLSQP method, starting from $x=[0,1]$

# In[5]:


x0 = [0.0, 1.0]
res = minimize(f, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(res)

