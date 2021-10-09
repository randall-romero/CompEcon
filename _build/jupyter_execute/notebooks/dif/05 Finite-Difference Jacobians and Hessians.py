#!/usr/bin/env python
# coding: utf-8

# # Finite-Difference Jacobians and Hessians
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demdif05.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## Initial tasks
# 

# In[1]:


from compecon import jacobian, hessian
import numpy as np

np.set_printoptions(precision=15)


# ## Example 1
# 
# The exact Jacobian of
# \begin{equation*}
# f(x_1,x_2) = \begin{bmatrix}\exp(x_1)-x_2 \\ x_1+x_2^2 \\ (1-x_1)\log(x_2)\end{bmatrix}
# \end{equation*}
# at $(0,1)$ is
# \begin{equation*}
# f'(x_1,x_2) = \begin{bmatrix}1 & -1 \\ 1 & 2 \\ 0 & 1\end{bmatrix}
# \end{equation*}

# In[2]:


def f(x):
    x1, x2 = x
    y = [np.exp(x1)-x2,
         x1 + x2**2,
         (1-x1)*np.log(x2)]
    return np.array(y)

jacobian(f, [0, 1])


# ## Ejemple 2
# 
# The exact Hessian of
# \begin{equation*}
# f(x_1,x_2) = x_1^2 \exp(-x_2)
# \end{equation*}
# at $(1,0)$ is
# \begin{equation*}
# f''(x_1,x_2) = \begin{bmatrix}2 & -2 \\ -2 &  1\end{bmatrix}.
# \end{equation*}

# In[3]:


def f(x):
    x1, x2 = x
    return x1**2 * np.exp(-x2)

hessian(f,[1, 0])

