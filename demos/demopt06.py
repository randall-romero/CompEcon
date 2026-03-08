
# coding: utf-8

# ### DEMOPT06 
# # KKT conditions for constrained optimization problems

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from compecon.demos import demo

plt.style.use('seaborn')


# In[2]:


x = np.linspace(-0.5,1.5, 100)
a, b = 0.1, 1.1
ylim = [0.5, 2]

fig = plt.figure(figsize=[10,4])
demo.subplot(1,2,1,'Internal Maximum', '$x$', '$f (x)$', [a-0.05, b+0.05], ylim)
f = lambda x: 1.5 - 2*(x-0.75)**2
plt.plot(x, f(x))
plt.plot([a, a], ylim,'g--',linewidth=4)
plt.plot([b, b], ylim,'g--',linewidth=4)
plt.xticks([a, b], ['a', 'b'])
plt.yticks(ylim,['', ''])
xstar = 0.75
ystar = f(xstar)
demo.bullet(xstar,ystar,'ro',10)
demo.annotate(0.65,0.75,"$x-a>0\Rightarrow\lambda=0$\n$b-x>0\Rightarrow\mu=0$\n$\Rightarrow f\,'(x)=0$", ms=0)


demo.subplot(1,2,2,'Corner Maximum', '$x$', '$f (x)$', [a-0.05, b+0.05], ylim)
g = lambda x: 2 - 0.75*(x + 0.25)**2
plt.plot(x, g(x))
plt.plot([a, a], ylim,'g--',linewidth=4)
plt.plot([b, b], ylim,'g--',linewidth=4)
plt.xticks([a, b], ['a', 'b'])
plt.yticks(ylim,['', ''])
demo.bullet(a,g(a),'ro',10)
demo.annotate(0.45,0.75,"$x=a\Rightarrow\lambda\geq0$\n$b-x>0\Rightarrow\mu=0$\n$\Rightarrow f\,'(x)=-\lambda\leq0$", ms=0)

demo.savefig([fig])
