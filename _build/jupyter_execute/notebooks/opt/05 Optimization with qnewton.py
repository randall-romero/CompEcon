#!/usr/bin/env python
# coding: utf-8

# # Optimization with qnewton
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demopt05.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# In[1]:


from compecon import OP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

np.set_printoptions(precision=4,suppress=True)
plt.style.use('seaborn')


# ## Example 1
# Find the optimal value of 
# $$f(x) = x^3 - 12x^2 + 36x + 8$$

# In[2]:


def f(x):
    return x ** 3 - 12 * x ** 2 + 36 * x + 8

F = OP(f)

x = F.qnewton(x0=4.0)
J = F.jacobian(x)
E = np.linalg.eig(F.hessian(x))[0]

print('x = ', x, '\nJ = ', J, '\nE = ', E)


# In[3]:


fig, ax = plt.subplots()
xx = np.linspace(0,8.2,100)
ax.plot(xx,f(xx))
ax.plot(4,f(4),'b.',ms=10)
ax.plot(x,f(x),'r.',ms=18)


# ## Find the optimum for 
# $$g(x,y) = 5 - 4x^2 - 2y^2 - 4xy - 2y$$

# In[4]:


def g(z):
    x, y = z
    return 5 - 4*x**2 - 2*y**2 - 4*x*y - 2*y
    
G = OP(g, print=True)
x = G.qnewton(x0=[-1, 1])
J = G.jacobian(x)
E = np.linalg.eig(G.hessian(x))[0]
print('x = ', x, '\nJ = ', J, '\nE = ', E)


# In[5]:


xx0 = np.linspace(-1.0,2.0,25)
xx1 = np.linspace(-2.5,0.5,25)
x0, x1 = np.meshgrid(xx0,xx1)

fig, ax = plt.subplots()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x0, x1, g([x0, x1]), rstride=1, cstride=1, 
                cmap=cm.Spectral, linewidth=0, antialiased=False)
ax.set_xlabel('$x_0$')
ax.set_xticks(np.linspace(-1.0,2.0,4))
ax.set_ylabel('$x_1$')
ax.set_yticks(np.linspace(-2.5,0.5,4))
ax.set_zlabel('$g(x_0,x_1)$')

