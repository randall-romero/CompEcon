__author__ = 'Randall'

from demos.setup import np
from compecon import DDPmodel

## DEMDDP08 Job search model

# Model Parameters
u     =  50                  # weekly unemp. benefit
v     =  60                  # weekly value of leisure
pfind = 0.90                 # prob. of finding job
pfire = 0.10                 # prob. of being fired
delta = 0.99                 # discount factor

# State Space
S = np.array([1, 2])         # vector of states
n = S.size                   # number of states

# Action Space (idle=1, active=2)
X = ['idle', 'active']          	# vector of actions
m = len(X)               	# number of actions

# Reward Function
f = np.zeros((m, n))
f[0] = v                   # gets leisure
f[1, 0] = u                   # gets benefit


# State Transition Probability Matrix
P = np.zeros((m, n, n))
P[0, :, 0] = 1                 # remains unemployed
P[1, 0, 0] = 1 - pfind           # finds no job
P[1, 0, 1] = pfind             # finds job
P[1, 1, 0] = pfire             # gets fired
P[1, 1, 1] = 1 - pfire           # keeps job

# Model Structure
model = DDPmodel(f, P, delta)

## Solution

# Solve Bellman Equation
wage = np.arange(55, 66)
xtable = np.zeros((wage.size, 2), dtype=int)
for i, w in enumerate(wage):
    model.reward[1, 1] = w  # vary wage
    xtable[i] = model.solve().policy  # solve via policy iteration

## Analysis

# Display Optimal Policy
print('  Optimal Job Search Strategy')
print('   (1=innactive, 2=active)   ')
print('  Wage  Unemployed  Employed ')


print(*['{:4d}  {:10s}{:10s}\n'.format(w, X[u], X[e]) for w, (u, e) in zip(wage, xtable)])

