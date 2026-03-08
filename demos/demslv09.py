
# coding: utf-8

# ###DemSlv09:
# # Hard nonlinear complementarity problem with Billup's function
# Solve hard nonlinear complementarity problem on R using semismooth and minmax methods.  Problem involves Billup's function.  Minmax formulation fails semismooth formulation suceeds.
#
# The function to be solved is $$f(x) = 1.01 - (1- x)^2$$
# where $x \geq 0$. Notice that $f(x)$ has roots $1\pm\sqrt{1.01}$

# * Preliminary tasks

# In[1]:
from demos.setup import np, plt, tic, toc
from compecon import MCP



# * Billup's function roots are

# In[2]:

roots = 1 + np.sqrt(1.01), 1 - np.sqrt(1.01)
print(roots)


# ### Set up the problem
# The class **MCP** is used to represent mixed-complementarity problems. To create one instance, we define the objective function and the boundaries $a$ and $b$ such that for $a \leq x \leq b$

# In[3]:

def billups(x):
    fval = 1.01 - (1 - x) ** 2
    fjac = 2 * (1 - x )
    return fval, fjac


a = 0
b = np.inf

Billups = MCP(billups, a, b)


# ### Solve by applying Newton method
# * Using minmax formulation
# Initial guess is $x=0$
Billups.x0 = 0.0


# In[4]:

t1 = tic()
x1 =  Billups.zero(transform='minmax')
t1 = 100*toc(t1)
n1 = Billups.fnorm

# * Using semismooth formulation

# In[5]:

t2 = tic()
x2 = Billups.zero(transform='ssmooth')
t2 = 100*toc(t2)
n2 = Billups.fnorm

# ### Print results
# Hundreds of seconds required to solve hard nonlinear complementarity problem using Newton minmax and semismooth formulations

# In[6]:

frm = '{:21} {:6.3f}  {:8.1e}     {:7.6f}'
prt = lambda d, t, n, x: print(frm.format(d, t, n, *x))

print('{:21} {:^6}  {:^8}     {:^7}\n{}'.format('Algorithm','Time','Norm','x','-' * 51));
prt('Newton minmax', t1, n1, x1)
prt('Newton semismooth', t2, n2, x2)


# ### Plot results
# Here we use the methods *ssmooth* and *minmax* from class **MCP** to compute the semi-smooth and minimax transformations.

# In[7]:

fig = plt.figure()
original = {'label':'Original', 'alpha':0.5, 'color':'gray'}
x = np.linspace(-0.5, 2.5, 500)

ax1 = fig.add_subplot(121, title='Difficult NCP', aspect=1,
                     xlabel='x', xlim=[-0.5, 2.5], ylim=[-1, 1.5])
ax1.axhline(ls='--', color='gray')
ax1.plot(x, billups(x)[0], **original)
ax1.plot(x, Billups.ssmooth(x), label='Semismooth')
ax1.plot(x, Billups.minmax(x), label='Minmax')
ax1.legend(loc='best')

x = np.linspace(-0.03, 0.03, 500)
ax2 = fig.add_subplot(122, title='Difficult NCP Magnified', aspect=1,
                      xlabel='x', xlim = [-.035, .035], ylim=[ -.01, .06])
ax2.axhline(ls='--', color='gray')
ax2.plot(x, Billups.original(x), **original)
ax2.plot(x, Billups.ssmooth(x), label='Semismooth')
ax2.plot(x, Billups.minmax(x), label='Minmax')
ax2.legend(loc='best')

plt.show()