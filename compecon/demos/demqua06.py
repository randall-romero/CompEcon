
# coding: utf-8

# ### DEMQUA06
# # Area under a curve

# In[1]:


from numpy import cos, pi, linspace
from compecon import qnwsimp, demo
import matplotlib.pyplot as plt


# In[2]:


def f(x):
    return 50 - cos(pi*x)*(2*pi*x - pi + 0.5)**2


# In[3]:


xmin, xmax = 0, 1
ymin, ymax = 25, 65
a, b, n = 0.25, 0.75, 401


# In[4]:


z = linspace(a,b,n)
x = linspace(xmin, xmax, n)

plt.figure(figsize=[8,4])
plt.fill_between(z,ymin,f(z), color='yellow')
plt.hlines(ymin, xmin, xmax, 'k',linewidth=2)
plt.vlines(a, ymin, f(a), 'r',linestyle='--',linewidth=2)
plt.vlines(b, ymin, f(b), 'r',linestyle='--',linewidth=2)
plt.plot(x,f(x), linewidth=3)
plt.xlim([xmin, xmax])
plt.ylim([ymin-5, ymax])
plt.yticks([ymin], ['0'], size=20)
plt.xticks([a, b],['$a$', '$b$'],size=20)
demo.annotate(xmax-0.1, f(xmax)-9, r'$f(x)$',fs=18,ms=0)
demo.annotate((a+b)/2, ymin+10 ,r'$A = \int_a^bf(x)dx$',fs=18,ms=0)

demo.savefig([plt.gcf()])