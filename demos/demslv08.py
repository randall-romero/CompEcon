
# coding: utf-8

# ### DemSlv08:
# # Nonlinear complementarity problem methods
# Solve nonlinear complementarity problem on $R^2$ using semismooth and minmax methods.
#
# Function to be solved is
# $$f(x,y) = \begin{array}
# {}200x(y - x^2) + 1 - x \\
# 100(x^2 - y)
# \end{array}$$

# * Preliminary tasks

# In[1]:

from demos.setup import np, tic, toc
from compecon import MCP


# ### Set up the problem
# The class **MCP** is used to represent mixed-complementarity problems. To create one instance, we define the objective function and the boundaries $a$ and $b$ such that for $a \leq x \leq b$. Apart from the required parameters, we can specify options to be used when solving the problem.

def f(z):
    x, y = z
    fval = np.array([200 * x * (y - x ** 2) + 1 - x,
                     100 * (x ** 2 - y)])
    fjac = np.array([[200 * (y - x ** 2) - 400 * x ** 2 - 1, 200 * x],
                         [200 * x, -100]])

    return fval, fjac


# * Generate problem test data
z = 2 * np.random.randn(2, 2)
a = 1 + np.min(z, 0)
b = 1 + np.max(z, 0)

F = MCP(f, a, b, maxit=1500)


# ### Solve by applying Newton method
# We'll use a random initial guess $x$

# In[4]:

F.x0 = np.random.randn(2)


''' Check the Jacobian '''
F.check_jacobian()



# * Using minmax formulation

# In[5]:

t0 = tic()
x1 = F.zero(transform='minmax')
t1 = 100 * toc(t0)
n1 = F.fnorm

# * Using semismooth formulation

# In[6]:

t0 = tic()
x2 = F.zero(transform='ssmooth')
t2 = 100*toc(t0)
n2 = F.fnorm

# * Print results

# In[7]:

hdr = 'Hundreds of seconds required to solve nonlinear complementarity \n' +\
      'problem on R^2 using minmax and semismooth formulations, with \n'  +\
      'randomly generated bounds \n' +\
      '\ta = [{:4.2f}, {:4.2f}] \n' + \
      '\tb = [{:4.2f}, {:4.2f}]'

frm = '{:20} {:6.3f}  {:8.0e}     {:5.2f}  {:5.2f}'
prt = lambda method, t, n, x: print(frm.format(method, t, n, *x))

print(hdr.format(a[0], a[1], b[0], b[1]))
print('\nAlgorithm           Time      Norm        x1     x2\n{}'.format('-' * 56));
prt('Newton minmax',     t1, n1, x1)
prt('Newton semismooth', t2, n2, x2)


