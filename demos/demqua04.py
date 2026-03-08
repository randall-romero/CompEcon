
# coding: utf-8

# ### DEMQUA04
# # Area under normal pdf using Simpson's rule

# In[1]:


import numpy as np
from compecon import qnwsimp, demo
import matplotlib.pyplot as plt


# In[2]:


n, a, z = 11, 0, 1

def f(x):
    return np.sqrt(1/(2*np.pi))*np.exp(-0.5*x**2)


# In[3]:


x, w = qnwsimp(n, a, z)
prob = 0.5 + w.dot(f(x))


# In[4]:


a, b, n = -4, 4, 500
x = np.linspace(a, b, n)
xz = np.linspace(a, z, n)

plt.figure(figsize=[8,4])
plt.fill_between(xz,f(xz), color='yellow')
plt.hlines(0, a, b,'k','solid')
plt.vlines(z, 0, f(z),'r',linewidth=2)
plt.plot(x,f(x), linewidth=3)
demo.annotate(-1, 0.08,r'$\Pr\left(\tilde Z\leq z\right)$',fs=18,ms=0)
plt.yticks([])
plt.xticks([z],['$z$'],size=20)

demo.savefig([plt.gcf()])