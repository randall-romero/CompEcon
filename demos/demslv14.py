
# coding: utf-8

# ### DemSlv14
# # Spacial Equilibrium Model
# 
# * See textbook page 56 for description

# * Demand and supply equations
#     Country     Demand        Supply
#        1      p = 42 - 2q    p =  9 + 1q
#        2      p = 54 - 3q    p =  3 + 2q
#        3      p = 51 - 1q    p = 18 + 1q
# 
# * Transportation costs:
#         From        To country 1 To country 2 To country 3
#         Country 1        0             3            9
#         Country 2        3             0            3
#         Country 3        6             3            0
# 

# In[1]:


import numpy as np
from compecon import MCP
np.set_printoptions(precision=4, suppress=True)


# In[2]:


A = np.array
as_ = A([9, 3, 18])
bs = A([1, 2, 1])
ad = A([42, 54, 51])
bd = A([3, 2, 1])
c = A([[0, 3, 9], [3, 0, 3],[6, 3, 0]])


# In[3]:


def market(x, jac=False):
    quantities = x.reshape((3,3))
    ps = as_ + bs * quantities.sum(0)
    pd = ad - bd * quantities.sum(1)
    ps, pd = np.meshgrid(ps, pd)
    fval = (pd - ps - c).flatten()
    return (fval, None) if jac else fval


# In[4]:


a = np.zeros(9)
b = np.full(9, np.inf)
Market = MCP(market, a, b)


# In[5]:


x0 = np.zeros(9)
x = Market.zero(x0, transform='minmax')


# In[6]:


quantities = x.reshape(3,3)
prices = as_ + bs * quantities.sum(0)
print('Quantities = \n', quantities)
print('Prices = \n', prices)
print('Net exports =\n', quantities.sum(0) - quantities.sum(1))


# * In autarky

# In[7]:


quantities = (ad - as_) / (bs + bd)
prices = as_ + bs * quantities


# In[8]:


print('Quantities = \n', quantities)
print('Prices = \n', prices)

