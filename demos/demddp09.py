__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel


## DEMDDP09 Deterministic cow replacement model


# Model Parameters
delta = 0.9                  # discount factor
cost  = 500                  # replacement cost
price = 150                  # milk price

# State Space
S = np.arange(1, 11)         # lactation states
n = S.size                   # number of states

# Action Space (keep='K', replace='R')
X = ['Keep', 'Replace']      # keep or replace
m = len(X)                   # number of actions

# Reward Function
f = np.empty((m, n))
y = -0.2 * S ** 2 + 2 * S + 8  # yield per lactation
f[0] = price * y
f[1] = f[0] - cost
f[0, -1] = -np.inf               # force replace at lactation 10

# State Transition Function
g = np.ones_like(f)
g[0] = np.minimum(np.arange(n) + 1, n - 1)  # Raise lactation number by 1, if keep


# Model Structure

model = DDPmodel(f, g, delta)
model.solve()

# Plot Optimal Policy
demo.figure('Optimal Replacement', 'Age', 'Optimal Decision', [0, n + 1], [-0.5, 1.5])
plt.plot(S, model.policy, '*', markersize=15)
plt.yticks((0, 1), X)

# Plot Value Function
demo.figure('Optimal Value in Cow Replacement', 'Age', 'Value (thousands)')
plt.plot(S, model.value / 1000)

plt.show()
