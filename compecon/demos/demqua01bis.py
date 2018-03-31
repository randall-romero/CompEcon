
# coding: utf-8

# ### DEMQUA01bis
# # Computing integral with quasi-Monte Carlo methods
# 

# To seven significant digits,
# \begin{align*}
# A &=\int_{-1}^1\int_{-1}^1  e^{-x_1}\cos^2(x_2)dx _1dx_2\\
#  &=\int_{-1}^1 e^{-x_1} dx _1 \times \int_{-1}^1 \cos^2(x_2) dx_2\\
#  &=\left(e - \tfrac{1}{e}\right) \times \left(1+\tfrac{1}{2}\sin(2)\right)
#  &\approx 3.4190098
# \end{align*}

# In[1]:


import numpy as np
from compecon import qnwtrap, qnwequi, demo
import matplotlib.pyplot as plt
import pandas as pd


# ### Make support function

# In[2]:


f1 = lambda x1: np.exp(-x1)
f2 = lambda x2: np.cos(x2)**2
f = lambda x1, x2: f1(x1) * f2(x2)


# In[3]:


def quad(method, n):
    (x1, x2), w = qnwequi(n,[-1, -1], [1, 1],method)
    return w.dot(f(x1, x2))


# ## Compute the approximation errors

# In[4]:


nlist = range(3,7)
quadmethods = ['Random', 'Neiderreiter','Weyl']

f_quad = np.array([[quad(qnw[0], 10**ni) for qnw in quadmethods] for ni in nlist])
f_true = (np.exp(1) - np.exp(-1)) * (1+0.5*np.sin(2))
f_error = np.log10(np.abs(f_quad/f_true - 1))


# ## Make table with results

# In[5]:


results = pd.DataFrame(f_error, columns=quadmethods)
results['Nodes'] = ['$10^%d$' % n for n in nlist]
results.set_index('Nodes', inplace=True)


# In[6]:


results


# In[7]:


results.to_latex('figures/demqua01bis.tex', escape=False, float_format='%.1f')

