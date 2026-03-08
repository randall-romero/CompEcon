
# coding: utf-8

# ### DEMSLV10

# # Illustrates linear complementarity problem

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


from compecon.demos import demo

plt.style.use('seaborn')


# In[2]:


def basicsubplot(i, title, yvals,solution):
    ax = demo.subplot(1,3,i,title,'','',[-0.05,1.05],[-2,2])
    ax.plot([0,1],[0,0],'k-',linewidth=1.5)
    ax.plot([0,0],[-2,2],'k:',linewidth=2.5)
    ax.plot([1,1],[-2,2],'k:',linewidth=2.5)
    ax.plot([0, 1],yvals)
    demo.bullet(solution[0], solution[1],'r.',18)
    plt.xticks([0,1],['a','b'])
    plt.yticks([-2,0,2],['','0',''])


# ## Possible Solutions to Complementarity Problem, $f$ Strictly Decreasing

# In[3]:


figs = []


# In[4]:


figs.append(plt.figure(figsize=[9,4]))
basicsubplot(1,'f(a) > f(b) > 0', [1.5, 0.5], [1.0,0.5])
basicsubplot(2,'f(a) > 0 > f(b)', [0.5, -0.5], [0.5,0.0])
basicsubplot(3,'0 > f(a) > f(b)', [-0.5, -1.5],[0.0,-0.5])
plt.show()


# ## Possible Solutions to Complementarity Problem, $f$ Strictly Increasing

# In[5]:


figs.append(plt.figure(figsize=[9,4]))
basicsubplot(1,'f(a) < f(b) < 0', [-1.5, -0.5], [0.0,-1.5])
basicsubplot(2,'f(a) < 0 < f(b)', [-0.5, 0.5], [0.5,0.0])
demo.bullet(0.0,-0.5,'r.',18)
demo.bullet(1.0,0.5,'r.',18)
basicsubplot(3,'0 < f(a) < f(b)', [0.5, 1.5],[1.0,1.5])
plt.show()


# In[6]:


demo.savefig(figs)

