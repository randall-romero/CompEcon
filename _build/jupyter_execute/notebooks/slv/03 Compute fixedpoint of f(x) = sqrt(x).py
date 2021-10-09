#!/usr/bin/env python
# coding: utf-8

# # Compute fixedpoint of $f(x) = x^{0.5}$
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv03.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>
# 

# ## About
# 
# Compute fixedpoint of $f(x) = x^{0.5}$ using Newton, Broyden, and function iteration methods.
# 
# Initial values generated randomly. Some alrorithms may fail to converge, depending on the initial value. 
# 
# True fixedpoint is $x=1$.

# In[1]:


import numpy as np
import pandas as pd
from compecon import tic, toc, NLP


# ### Randomly generate starting point

# In[2]:


xinit = np.random.rand(1) + 0.5


# ### Set up the problem

# In[3]:


def g(x):
    return np.sqrt(x)

problem_as_fixpoint = NLP(g, xinit)


# ### Equivalent Rootfinding Formulation

# In[4]:


def f(x):
    fval = x - np.sqrt(x)
    fjac = 1-0.5 / np.sqrt(x)
    return fval, fjac

problem_as_zero = NLP(f, xinit)


# ### Compute fixed-point using Newton method

# In[5]:


t0 = tic()
x1 = problem_as_zero.newton()
t1 = 100 * toc(t0)
n1 = problem_as_zero.fnorm


# ### Compute fixed-point using Broyden method

# In[6]:


t0 = tic()
x2 = problem_as_zero.broyden()
t2 = 100 * toc(t0)
n2 = problem_as_zero.fnorm


# ### Compute fixed-point using function iteration

# In[7]:


t0 = tic()
x3 = problem_as_fixpoint.fixpoint()
t3 = 100 * toc(t0)
n3 = np.linalg.norm(problem_as_fixpoint.fx - x3)


# ### Print results

# In[8]:


print('Hundredths of seconds required to compute fixed-point of g(x)=sqrt(x)')
print('using Newton, Broyden, and function iteration methods, starting at')
print('x = %4.2f\n' % xinit)

pd.DataFrame({
    'Time': [t1, t2, t3],
    'Norm of f': [n1, n2, n3],
    'x': [x1, x2, x3]},
    index=['Newton', 'Broyden', 'Function']
)

