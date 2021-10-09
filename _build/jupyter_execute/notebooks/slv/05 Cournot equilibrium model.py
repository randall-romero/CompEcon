#!/usr/bin/env python
# coding: utf-8

# # Cournot Equilibrium Model
# 
# **Randall Romero Aguilar, PhD**
# 
# This demo is based on the original Matlab demo accompanying the  <a href="https://mitpress.mit.edu/books/applied-computational-economics-and-finance">Computational Economics and Finance</a> 2001 textbook by Mario Miranda and Paul Fackler.
# 
# Original (Matlab) CompEcon file: **demslv05.m**
# 
# Running this file requires the Python version of CompEcon. This can be installed with pip by running
# 
#     !pip install compecon --upgrade
# 
# <i>Last updated: 2021-Oct-01</i>
# <hr>
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from compecon import NLP, gridmake


# ### Parameters and initial value

# In[2]:


alpha = 0.625
beta = np.array([0.6, 0.8])


# ### Set up the Cournot function

# In[3]:


def market(q):
    quantity = q.sum()
    price = quantity ** (-alpha)
    return price, quantity


# In[4]:


def cournot(q):
    P, Q = market(q)
    P1 = -alpha * P/Q
    P2 = (-alpha - 1) * P1 / Q
    fval = P + (P1 - beta) * q
    fjac = np.diag(2 * P1 + P2 * q - beta) + np.fliplr(np.diag(P1 + P2 * q))
    return fval, fjac


# ### Compute equilibrium using Newton method (explicitly)

# In[5]:


q = np.array([0.2, 0.2])

for it in range(40):
    f, J = cournot(q)
    step = -np.linalg.solve(J, f)
    q += step
    if np.linalg.norm(step) < 1.e-10: break

price, quantity = market(q)
print(f'Company 1 produces {q[0]:.4f} units, while company 2 produces {q[1]:.4f} units.')
print(f'Total production is {quantity:.4f} and price is {price:.4f}')


# ### Using a NLP object

# In[6]:


q0 = np.array([0.2, 0.2])
cournot_problem = NLP(cournot)
q = cournot_problem.newton(q0, show=True)

price, quantity = market(q)
print(f'\nCompany 1 produces {q[0]:.4f} units, while company 2 produces {q[1]:.4f} units.')
print(f'Total production is {quantity:.4f} and price is {price:.4f}')


# ### Generate data for contour plot

# In[7]:


n = 100
q1 = np.linspace(0.1, 1.5, n)
q2 = np.linspace(0.1, 1.5, n)
z = np.array([cournot(q)[0] for q in gridmake(q1, q2).T]).T


# ### Plot figures

# In[8]:


steps_options = {'marker': 'o',
                 'color': (0.2, 0.2, .81),
                 'linewidth': 1.0,
                 'markersize': 9,
                 'markerfacecolor': 'white',
                 'markeredgecolor': 'red'}

contour_options = {'levels': [0.0],
                   'colors': '0.25',
                   'linewidths': 0.5}


Q1, Q2 = np.meshgrid(q1, q2)
Z0 = np.reshape(z[0], (n,n), order='F')
Z1 = np.reshape(z[1], (n,n), order='F')

methods = ['newton', 'broyden']
cournot_problem.opts['maxit', 'maxsteps', 'all_x'] = 10, 0, True

qmin, qmax = 0.1, 1.3

fig, axs = plt.subplots(1,2,figsize=[12,6])
for ax, method in zip(axs, methods):
    x = cournot_problem.zero(method=method)
    ax.set(title=method.capitalize() + "'s method",
           xlabel='$q_1$',
           ylabel='$q_2$',
           xlim=[qmin, qmax],
           ylim=[qmin, qmax])
    ax.contour(Q1, Q2, Z0, **contour_options)
    ax.contour(Q1, Q2, Z1, **contour_options)
    ax.plot(*cournot_problem.x_sequence, **steps_options)

    ax.annotate('$\pi_1 = 0$', (0.85, qmax), ha='left', va='top')
    ax.annotate('$\pi_2 = 0$', (qmax, 0.55),  ha='right', va='center')

