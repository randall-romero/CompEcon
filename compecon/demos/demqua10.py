
# coding: utf-8

# ### DEMQUA10
# # Monte Carlo Simulation of Time Series
# 
# Simulate time series using Monte Carlo Method

# In[1]:


import numpy as np
from compecon import demo
from scipy.stats import norm
import matplotlib.pyplot as plt


# In[2]:


m, n = 3, 40
mu, sigma = 0.005, 0.02
e = norm.rvs(mu,sigma,size=[n,m])
logp = np.zeros([n+1,m])
logp[0] = np.log(2)
for t in range(40):
    logp[t+1] = logp[t] + e[t]


# In[3]:


demo.figure('','Week','Price', [0,n])
plt.plot(np.exp(logp))
demo.savefig([plt.gcf()])
