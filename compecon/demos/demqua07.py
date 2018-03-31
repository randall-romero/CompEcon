
# coding: utf-8

# ### DEMQUA07
# # Illustrates integration using Trapezoidal rule
# 

# In[1]:


from numpy import poly1d, linspace
from compecon import qnwtrap, demo
import matplotlib.pyplot as plt


# In[2]:


n = 1001
xmin, xmax = -1, 1
xwid = xmax-xmin
x = linspace(xmin, xmax, n)


# In[3]:


f = poly1d([2.0, -1.0, 0.5, 5.0])

# In[4]:


def plot_trap(n):
    xi, wi = qnwtrap(n+1, xmin, xmax)
    
    fig = plt.figure()
    plt.fill_between(xi, f(xi), color='yellow')
    plt.plot(x, f(x), linewidth=3, label=r'$f(x)$')
    plt.plot(xi, f(xi),'r--', label=r'$\tilde{f}_{%d}(x)$' % (n+1))
    plt.vlines(xi, 0, f(xi),'k', linestyle=':')
    plt.hlines(0,xmin-0.1, xmax+0.1,'k',linewidth=2)
    plt.xlim(xmin-0.1, xmax+0.1)
    xtl = ['$x_{%d}$' % i for i in range(n+1)]
    xtl[0] += '=a'
    xtl[n] += '=b'
    plt.xticks(xi, xtl)
    plt.yticks([0],['0'])
    plt.legend()
    return fig


# In[5]:


figs = [plot_trap(n) for n in [2, 4, 8]]

demo.savefig(figs)