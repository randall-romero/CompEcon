__author__ = 'Randall'


from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import getindex

# DEMDDP01 Mine management model

# Model Parameters
price = 1                    # price of ore
sbar  = 100                  # initial ore stock
delta = 0.9                  # discount factor

# State Space
S = np.arange(sbar + 1)      # vector of states
n = S.size                   # number of states

# Action Space
X = np.arange(sbar + 1)      # vector of actions
m = X.size                   # number of actions

# Reward Function
f = np.full((m, n), -np.inf)
for c, s in enumerate(S):
    for r, x in enumerate(X):
        if x <= s:
            f[r, c] = price * x - (x ** 2) / (1 + s)

# State Transition Function
g = np.empty_like(f)
for r, x in enumerate(X):
    snext = S - x
    g[r] = getindex(snext, S)


# Model Structure
model = DDPmodel(f, g, delta)
model.solve()

# Analysis

# Simulate Model
sinit = S.max()
nyrs = 15
t = np.arange(nyrs + 1)
spath, xpath = model.simulate(sinit, nyrs)

# Plot Optimal Policy
demo.figure('Optimal Extraction Policy', 'Stock', 'Extraction')
plt.plot(S, X[model.policy])

# Plot Value Function
demo.figure('Optimal Value Function', 'Stock', 'Value')
plt.plot(S, model.value)

# Plot State Path
demo.figure('Optimal State Path', 'Year', 'Stock')
plt.plot(t, S[spath])

plt.show()
