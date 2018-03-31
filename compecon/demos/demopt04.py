
# coding: utf-8

# ### DEMOPT04
# # Maximization of banana function by various methods

# $$f(x,y)=-100*(y-x^2)^2-(1-x)^2$$
# starting at \[0,1\].
# 

# In[1]:


import numpy as np
from compecon import OP
np.set_printoptions(4, linewidth=120, suppress=True)
import matplotlib.pyplot as plt

from compecon.demos import demo


# In[2]:


''' Set up the problem '''
x0 = [1, 0]
banana = OP(lambda x: -100 * (x[1] - x[0] ** 2)**2 - (1 - x[0]) ** 2,
            x0, maxit=250, print=True, all_x=True)


# In[3]:


x = banana.qnewton()
J = banana.jacobian(x)
E = np.linalg.eig(banana.hessian(x))[0]
print('x = ', x, '\nJ = ', J, '\nE = ', E)


# In[4]:


''' Plots options '''
steps_options = {'marker': 'o',
                 'color': (0.2, 0.2, .81),
                 'linewidth': 1.0,
                 'markersize': 3,
                 'markerfacecolor': 'white',
                 'markeredgecolor': 'red'}

contour_options = {'levels': -np.exp(np.arange(7,0.25,-0.5)),
                   'colors': '0.25',
                   'linewidths': 0.5}


# In[5]:


''' Data for coutours '''
n = 40  # number of grid points for plot per dimension
xmin = [-0.7, -0.2]
xmax = [ 1.2,  1.2]

X0, X1 = np.meshgrid(*[np.linspace(a, b, n) for a, b in zip(xmin, xmax)])
Y = banana.f([X0.flatten(), X1.flatten()])
Y.shape = (n, n)

fig = plt.figure(figsize=[12,4])

for it, method in enumerate(banana.search_methods.keys()):
    ''' Solve problem with given method '''
    print('\n\nMaximization with method ' + method.upper())
    x = banana.qnewton(SearchMeth=method)
    print('x =', x)

    ''' Plot the result '''
    demo.subplot(1, 3, it + 1, method.upper(),'x','y')
    
    plt.contour(X0, X1, Y, **contour_options)
    plt.plot(*banana.x_sequence, **steps_options)
    plt.plot(1, 1, 'r*', markersize=15)
    plt.title(method.upper() + " search")
    plt.xlabel('x', verticalalignment='top')
    plt.ylabel('y', verticalalignment= 'bottom')
    plt.axis((xmin[0], xmax[0], xmin[1], xmax[1]))
    
plt.show()


# ## Using Scipy
# 
# As of this version of CompEcon, the Nelder Mead method has not been implemented. However, we can still use it with the help of the **scipy.optimize.minimize** function. To this end, we must rewrite the banana function (change its sign) so that we switch from our original maximization problem to one of minimization.
# 
# 
# 

# In[6]:


from scipy.optimize import minimize


# In[7]:


x0 = [1, 0]
def banana2(x):
    return 100 * (x[1] - x[0] ** 2)**2 + (1 - x[0]) ** 2


# In[8]:


res = minimize(banana2, x0, method='Nelder-Mead')
print(res)

demo.savefig([fig])