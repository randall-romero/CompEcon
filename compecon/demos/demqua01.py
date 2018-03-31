
# coding: utf-8

# ###  DEMQUA01 
# # Equidistributed sequences on unit square in $R^2$
# 

# In[1]:


from compecon import qnwequi, demo
import matplotlib.pyplot as plt


# In[2]:

methods = [['N', 'Neiderreiter Sequence'],
           ['W', 'Weyl Sequence'],
           ['R','Pseudo-Random Sequence']]


# In[3]:


def equiplot(method):
    x, w = qnwequi(2500, [0, 0], [1, 1], method[0])
    fig = demo.figure(method[1], '$x_1$', '$x_2$',[0, 1], [0, 1], figsize=[5,5])
    plt.plot(*x,'.')
    plt.xticks([0, 1])
    plt.yticks([0,1])
    plt.axis('equal')
    return fig


# In[4]:


figs = [equiplot(k) for k in methods]
demo.savefig(figs)    

