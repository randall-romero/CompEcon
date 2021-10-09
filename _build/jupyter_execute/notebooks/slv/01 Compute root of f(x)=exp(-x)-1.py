#!/usr/bin/env python
# coding: utf-8

# # Compute root of $f(x)=\exp(-x)-1$
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv01.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>
# 
# 

# ## About
# 
# Compute root of $f(x)=\exp(-x)-1$ using Newton and secant methods. Initial value generated randomly. True root is $x=0$.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compecon import NLP, tic, toc


# ### Set up the problem

# In[2]:


def f(x):
    fval = np.exp(-x) - 1
    fjac = -np.exp(-x)
    return fval, fjac

problem = NLP(f, all_x=True)


# ### Randomly generate starting point

# In[3]:


problem.x0 = 10 * np.random.randn(1)


# ### Compute root using Newton method

# In[4]:


t0 = tic()
x1 = problem.newton()
t1 = 100 * toc(t0)
n1, x_newton = problem.fnorm, problem.x_sequence


# ### Compute root using Broyden method

# In[5]:


t0 = tic()
x2 = problem.broyden()
t2 = 100 * toc(t0)
n2, x_broyden = problem.fnorm, problem.x_sequence


# ### Print results

# In[6]:


print('Hundredths of seconds required to compute root of exp(-x)-1,')
print('via Newton and Broyden methods, starting at x = %4.2f.' % problem.x0)

pd.DataFrame({
    'Time': [t1, t2],
    'Norm of f': [n1, n2],
    'Final x': [x1, x2]},
    index=['Newton', 'Broyden']
)


# ### View current options for solver

# In[7]:


print(problem.opts)


# # Describe the options

# In[8]:


print(problem.opts.__doc__)


# ### Plot the convergence

# In[9]:


b = -abs(problem.x0)
a = -b
xx = np.linspace(a, b, 100)

fig, ax = plt.subplots()
ax.hlines(0, a, b, 'gray')
ax.plot(xx, f(xx)[0], 'b-', alpha=0.4)
ax.plot(x_newton.T,f(x_newton)[0].T,'ro:', alpha=0.4);

