#!/usr/bin/env python
# coding: utf-8

# # Simple nonlinear complementarity problem
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv13.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## About
# 
# The problem is to solve
# 
# \begin{equation}
# f(x, y) = \begin{bmatrix}1+xy -2x^3-x\\ 2x^2-y\end{bmatrix}
# \end{equation}
# 
# subject to $0 \leq x, y \leq 1$

# In[1]:


from compecon import MCP, jacobian
import numpy as np

x0 = [0.5, 0.5]    


# ### Solving the problem without the Jacobian
# To solve this problem we create a **MCP** object using a lambda function.

# In[2]:


def func(z):
    x, y = z
    return np.array([1 + x*y - 2*x**3 - x, 2*x**2 - y])
                      
F = MCP(func, [0, 0], [1,1])


# Solve for initial guess $x_0 = [0.5, 0.5]$

# In[3]:


x = F.zero(x0, transform='minmax', show=True)
#x = F.zero(x0, transform='ssmooth', show=True) # FIXME: generates error
print(f'Solution is {x=}.')


# ### Solving the problem with the Jacobian

# In[4]:


def func2(z):
    x, y = z
    f = [1 + x*y - 2*x**3 - x, 2*x**2 - y]
    J = [[y-6*x**2-1, x],[4*x, -1]]
    return np.array(f), np.array(J)                  

F2 = MCP(func2, [0, 0], [1,1])


# In[5]:


x = F2.zero(x0, transform='minmax', show=True)
print(f'Solution is {x=}.')

