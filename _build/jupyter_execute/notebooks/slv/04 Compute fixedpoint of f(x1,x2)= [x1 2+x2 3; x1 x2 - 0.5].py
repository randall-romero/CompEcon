#!/usr/bin/env python
# coding: utf-8

# # Compute fixedpoint of $f(x, y)= [x^2 + y^3; xy - 0.5]$
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv04.m**
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
# Compute fixedpoint of 
# 
# \begin{equation}
# f(x, y)= \begin{bmatrix}x^2 + y^3 \\ xy - 0.5 \end{bmatrix}
# \end{equation}
# 
# using Newton, Broyden, and function iteration methods.
# 
# Initial values generated randomly.  Some algorithms may fail to converge, depending on the initial value.
# 
# True fixedpoint is $x = -0.09$,  $y=-0.46$.

# In[1]:


import numpy as np
import pandas as pd
from compecon import NLP, tic, toc
np.random.seed(12)


# ### Set up the problem

# In[2]:


def g(z):
    x, y = z
    return np.array([x **2 + y ** 3, x * y - 0.5])

problem_as_fixpoint = NLP(g, maxit=1500)


# ### Equivalent Rootfinding Formulation

# In[3]:


def f(z):
    x, y = z
    fval = [x - x ** 2 - y ** 3,
            y - x * y + 0.5]
    fjac = [[1 - 2 * x, -3 * y **2],
            [-y, 1 - x]]

    return np.array(fval), np.array(fjac)

problem_as_zero = NLP(f, maxit=1500)


# ### Randomly generate starting point

# In[4]:


xinit = np.random.randn(2)


# ### Compute fixed-point using Newton method

# In[5]:


t0 = tic()
z1 = problem_as_zero.newton(xinit)
t1 = 100 * toc(t0)
n1 = problem_as_zero.fnorm


# ### Compute fixed-point using Broyden method

# In[6]:


t0 = tic()
z2 = problem_as_zero.broyden(xinit)
t2 = 100 * toc(t0)
n2 = problem_as_zero.fnorm


# ### Compute fixed-point using function iteration

# In[7]:


t0 = tic()
z3 = problem_as_fixpoint.fixpoint(xinit)
t3 = 100 * toc(t0)
n3 = np.linalg.norm(problem_as_fixpoint.fx - z3)


# 

# In[8]:


print('Hundredths of seconds required to compute fixed-point of ')
print('\n\t\tg(x1,x2)=[x1^2+x2^3; x1*x2-0.5]')
print('\nusing Newton, Broyden, and function iteration methods, starting at')
print('\n\t\tx1 = {:4.2f}  x2 = {:4.2f}\\n\\n'.format(*xinit))

pd.DataFrame({
    'Time': [t1, t2, t3],
    'Norm of f': [n1, n2, n3],
    'x1': [z1[0], z2[0], z3[0]],
    'x2': [z1[1], z2[1], z3[1]]},
    index=['Newton', 'Broyden', 'Function']
)

