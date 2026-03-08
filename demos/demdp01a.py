
# coding: utf-8

# ### DEMDP01a
# # Timber Harvesting -- using 2 nodes

# In[1]:


import numpy as np
from compecon import NLP, demo
import matplotlib.pyplot as plt


# In[2]:


price = 1.0  # price of biomass
kappa = 0.2  # clearcut-replant cost
smax  = 0.5  # stand carrying capacity
gamma = 0.1  # biomass growth parameter
delta = 0.9  # discount factor


# ### Code the growth function

# In[3]:


snodes = np.array([0.2, 0.4])


# In[4]:


def h(s): return s + gamma*(smax - s)


# ## SOLUTION
# 
# ### Code the approximant and the residual

# In[5]:


def vhat(c, s): return c[0] + c[1]*s
def vhat1(c,s): return price*s - kappa + delta * vhat(c,h(0))
def vhat0(c,s): return delta * vhat(c, h(s))

def resid(c,s=snodes): return vhat(c,s) - np.maximum(vhat0(c,s), vhat1(c,s))


# ### Solve collocation equation

# In[6]:


cc = NLP(resid).broyden(np.zeros(2))


# ### Compute critical biomass

# In[7]:


scrit = NLP(lambda s: vhat0(cc,s)-vhat1(cc,s)).broyden(0.0)[0]


# In[8]:


scrit


# ## ANALYSIS

# ### Compute refined state grid

# In[9]:


ss = np.linspace(0,smax,1000)


# ### Plot Conditional Value Functions

# In[10]:


fig1 =demo.figure('Conditional Value Functions','Biomass','Value of Stand')
plt.plot(ss,vhat0(cc,ss),label='Grow')
plt.plot(ss,vhat1(cc,ss),label='Clear-Cut')
plt.legend()

vcrit = vhat(cc,scrit)
ymin = plt.ylim()[0]
plt.vlines(scrit, ymin,vcrit,'grey',linestyles='--')
demo.annotate(scrit,ymin,'$s^*$',ms=10)
demo.bullet(scrit,vcrit)
print(f'Optimal Biomass Harvest Level = {scrit:.4f}') 


# ### Plot Value Function Residual

# In[ ]:


fig2 = demo.figure('Bellman Equation Residual', 'Biomass', 'Percent Residual')
plt.plot(ss, 100*resid(cc,ss) / vhat(cc,ss))
plt.hlines(0,0,smax,linestyles='--')
plt.plot(snodes,resid(cc),'ro')


# ### Compute ergodic mean annual harvest 

# In[ ]:


s = h(0)
for n in range(100):
    if s > scrit: break
    s = h(s)
    
print(f'Ergodic Mean Annual Harvest = {s/n:.4f} after {n+1} iterations') 


# In[ ]:


demo.savefig([fig1,fig2])

