__author__ = 'Randall'


## DEMDP09 Private Non-Renewable Resource Model
#
#  Profit maximizing mine owner must decide how much ore to extract.
#
# States
#     s       ore stock
# Actions
#     q       ore extracted and sold
# Parameters
#     a       demand function parameters
#     b       cost function parameters
#     delta   discount factor

# Preliminary tasks
from compecon import BasisSpline, DPmodel
from compecon.quad import qnwlogn
from demos.setup import np, plt, demo

## FORMULATION
 
# Model Parameters
a = [6, 0.8]                            # demand function parameters
b = [7, 1.0]                            # cost function parameters
delta = 0.9                            # discount factor

# Approximation Structure
n    = 51                             # number of collocation nodes
smin =   0                             # minimum state
smax =  10                             # maximum state
basis = BasisSpline(n, smin, smax, labels=['stock'])    # basis functions


def bounds(s, i, j):
    return np.zeros_like(s), s.copy()

def reward(s, q, i, j):
    a0, a1 = a
    b0, b1 = b
    f = (a0 - b0 + b1 * s) * q - (a1 + b1 / 2) * q ** 2
    fx = (a0 - b0 + b1 * s) - 2*(a1 + b1 / 2) * q
    fxx = -2 * (a1 + b1 / 2) * np.ones_like(s)
    return f, fx, fxx

def transition(s,q,i,j,in_,e):
    g = s - q
    gx = - np.ones_like(s)
    gxx = np.zeros_like(s)
    return g, gx, gxx


model = DPmodel(basis, reward, transition, bounds,
                x=['extracted'], discount=delta)


# Check Model Derivatives
# dpcheck(model,smax,0)


## SOLUTION

# Solve Bellman Equation
model.solve()
resid, s, v, q = model.solution()

# Compute and print abandonment point
sstar = (b[0] - a[0]) / b[1]
print('Abandonment Point = %5.2f' % sstar)

# Plot Optimal Policy
demo.figure('Optimal Extraction', 'Ore Stock', 'Ore Extracted')
plt.plot(s, q.T)

# Plot Value Function
demo.figure('Value Function', 'Ore Stock', 'Value')
plt.plot(s, v.T)

# Plot Shadow Price Function
demo.figure('Shadow Price Function', 'Ore Stock', 'Shadow Price')
plt.plot(s, model.Value(s, 1))

# Plot Residual
demo.figure('Bellman Equation Residual', 'Ore Stock', 'Residual')
plt.plot(s, resid.T)
plt.hlines(0, 0, smax,'k', '--')


## SIMULATION

# Simulate Model
T = 20
data = model.simulate(T, smax)

# Plot State and Policy Paths
data[['stock', 'extracted']].plot()
plt.title('State and Policy Paths')
plt.legend(['Stock','Extraction'])
plt.hlines(sstar, 0, T, 'k', '--')
plt.xlabel('Period')
plt.ylabel('Stock / Extraction')

plt.show()
