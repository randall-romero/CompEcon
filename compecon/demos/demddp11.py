__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import getindex

## DEMDDP11 Stochastic optimal growth model

# Model Parameters
delta =  0.9                 # discount factor
alpha =  0.2                 # utility parameter
beta  =  0.5                 # production parameter
gamma =  0.9                 # capital survival rate

# State Space
smin  =  1.0                 # minimum state
smax  = 10.0                 # maximum state
n = 200                      # number of states
S = np.linspace(smin, smax, n)    # vector of states

# Action Space
xmin  = 0.65                 # minimum action
xmax  = 0.99                 # maximum action
m =  500                     # number of actions
X = np.linspace(xmin, xmax, m)    # vector of actions


# Reward Function
f = np.empty((m ,n))
for k in range(m):
    f[k] = ((S * (1 - X[k])) ** (1-alpha)) / (1 - alpha)


# State Transition Function
g = np.empty_like(f)
for k in range(m):
    snext = gamma * X[k] * S + (X[k] * S) ** beta
    g[k] = getindex(snext, S)

# Model Structure
model = DDPmodel(f, g, delta).solve()

## Analysis

# Plot Optimal Policy
demo.figure('Optimal Investment', 'Wealth', 'Investment')
plt.plot(S, X[model.policy] * S)

# Plot Optimal Policy
demo.figure('Optimal Consumption', 'Wealth', 'Consumption')
plt.plot(S, S - X[model.policy] * S)

# Plot Value Function
demo.figure('Optimal Value Function', 'Wealth', 'Value')
plt.plot(S, model.value)


# Simulate Model
nyrs = 20
t = np.arange(0, nyrs + 1)
st, xt = model.simulate(smin, nyrs)

# Plot State Path
demo.figure('Optimal State Path', 'Year', 'Wealth')
plt.plot(t, S[st])

# Compute Steady State Distribution and Mean
pi = model.markov()
avgstock = np.dot(S, pi)
print('Steady-state Wealth     {:8.2f},  {:8.2f}'.format(*avgstock))

plt.show()