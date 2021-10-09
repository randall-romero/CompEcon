#!/usr/bin/env python
# coding: utf-8

# # Compute root of Rosencrantz function
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv02.m**
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
# Compute root of 
# 
# \begin{equation}
# f(x_1,x_2)= \begin{bmatrix}200x_1(x_2-x_1^2) + 1-x_1 \\ 100(x_1^2-x_2)\end{bmatrix}
# \end{equation}
# 
# using Newton and Broyden methods. Initial values generated randomly.  True root is $x_1=1, \quad x_2=1$.

# In[1]:


import numpy as np
import pandas as pd
from compecon import NLP, tic, toc


# ### Set up the problem

# In[2]:


def f(x):
    x1, x2 = x
    fval = [200 * x1 * (x2 - x1 ** 2) + 1 - x1, 100 * (x1 ** 2 - x2)]
    fjac = [[200 * (x2 - x1 ** 2) - 400 * x1 ** 2 - 1, 200 * x1],
            [200 * x1, -100]]
    return np.array(fval), np.array(fjac)

problem = NLP(f)


# ### Randomly generate starting point

# In[3]:


problem.x0 = np.random.randn(2)


# ### Compute root using Newton method

# In[4]:


t0 = tic()
x1 = problem.newton()
t1 = 100 * toc(t0)
n1 = problem.fnorm


# ### Compute root using Broyden method

# In[5]:


t0 = tic()
x2 = problem.broyden()
t2 = 100 * toc(t0)
n2 = problem.fnorm


# ### Print results

# In[6]:


print('Hundreds of seconds required to compute root of Rosencrantz function')
print('f(x1,x2)= [200*x1*(x2-x1^2)+1-x1;100*(x1^2-x2)] via Newton and Broyden')
print('methods, starting at x1 = {:4.2f} x2 = {:4.2f}'.format(*problem.x0))

pd.DataFrame({
    'Time': [t1, t2],
    'Norm of f': [n1, n2],
    'x1': [x1[0], x2[0]],
    'x2': [x1[1], x2[1]]},
    index=['Newton', 'Broyden']
)

