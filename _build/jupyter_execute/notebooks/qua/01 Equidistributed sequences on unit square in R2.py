#!/usr/bin/env python
# coding: utf-8

# # Equidistributed sequences on unit square in $R^2$
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demqua01.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>

# ## Initial tasks

# In[1]:


from compecon import qnwequi, demo
import matplotlib.pyplot as plt


# In[2]:


methods = [['N', 'Neiderreiter Sequence'],
           ['W', 'Weyl Sequence'],
           ['R','Pseudo-Random Sequence']]


# In[3]:


def equiplot(method):
    x, w = qnwequi(2500, [0, 0], [1, 1], method[0])
    fig, ax = plt.subplots(figsize=[5,5])
    ax.set(title=method[1],
           xlabel='$x_1$',
           ylabel='$x_2$',
           xlim=[0, 1],
           ylim=[0, 1],
           xticks=[0, 1],
           yticks=[0,1])
    ax.axis('equal')
    ax.plot(*x,'.')
    return fig


# In[4]:


figs = [equiplot(k) for k in methods]
#demo.savefig(figs, name='demqua01')    

