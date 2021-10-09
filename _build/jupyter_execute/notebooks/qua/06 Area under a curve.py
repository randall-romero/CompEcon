#!/usr/bin/env python
# coding: utf-8

# # Area under a curve
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


from numpy import cos, pi, linspace, array
import matplotlib.pyplot as plt


# In[2]:


def f(x):
    return 25 - cos(pi*x)*(2*pi*x - pi + 0.5)**2


# In[3]:


x_range = array([0, 1])
a_b = array([0.25, 0.75])
n = 401

z = linspace(*a_b, n)
x = linspace(*x_range, n)


# In[4]:


fig, ax = plt.subplots(figsize=[8,4])
ax.fill_between(z, 0, f(z), alpha=0.35, color='LightSkyBlue')
ax.hlines(0, *x_range, 'k', linewidth=2)
ax.vlines(a_b, 0, f(a_b), color='tab:orange',linestyle='--',linewidth=2)
ax.plot(x,f(x), linewidth=3)
ax.set(xlim=x_range, xticks=a_b,
       ylim=[-5, f(x).max()+2], yticks=[0])
       
ax.set_yticklabels(['0'], size=20)
ax.set_xticklabels(['$a$', '$b$'], size=20)

ax.annotate(r'$f(x)$', [x_range[1] - 0.1, f(x_range[1])-5], fontsize=18, color='C0', va='top')
ax.annotate(r'$A = \int_a^bf(x)dx$', [a_b.mean(), 10] ,fontsize=18, ha='center');

#demo.savefig([fig], name='demqua06')

