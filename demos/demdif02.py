
# coding: utf-8

# ### DEMDIF02
# # Demonstrates accuracy of one- and two-sided finite-difference derivatives of $e^x$ at $x=1$ as a function of step size $h$.

# In[1]:


import numpy as np
from compecon import demo
import matplotlib.pyplot as plt


# In[2]:


n, x = 18, 1.0
c = np.linspace(-15,0,n)
h = 10 ** c


# In[3]:


exp = np.exp
eps = np.finfo(float).eps

def deriv_error(l, u):
    dd = (exp(u) - exp(l)) / (u-l)
    return np.log10(np.abs(dd - exp(x)))


# ## One-sided finite difference derivative

# In[4]:


d1 = deriv_error(x, x+h)
e1 = np.log10(eps**(1/2))


# ## Two-sided finite difference derivative

# In[5]:


d2 = deriv_error(x-h, x+h)
e2 = np.log10(eps**(1/3))


# ## Plot finite difference derivatives

# In[6]:


ylim = [-15, 5]
xlim = [-15, 0]
lcolor = [z['color']  for z in plt.rcParams['axes.prop_cycle']]


demo.figure('Error in Numerical Derivatives','$\log_{10}(h)$','$\log_{10}$ Approximation Error',xlim,ylim)
plt.plot(c,d1, label='One-Sided')
plt.plot(c,d2, label='Two-Sided')
plt.vlines([e1,e2],*ylim, lcolor,linestyle=':')
plt.xticks(np.arange(-15,5,5))
plt.yticks(np.arange(-15,10,5))
demo.annotate(e1,2,'$\sqrt{\epsilon}$',color=lcolor[0],ms=0)
demo.annotate(e2,2,'$\sqrt[3]{\epsilon}$',color=lcolor[1],ms=0)
plt.legend(loc='lower left')


# In[7]:


demo.savefig([plt.gcf()])

