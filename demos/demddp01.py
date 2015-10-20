__author__ = 'Randall'


from demos.setup import np, plt
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
f = np.full((n, m), -np.inf)
for r, s in enumerate(S):
    for c, x in enumerate(X):
        if x <= s:
            f[r, c] = price * x - (x ** 2) /(1 + s)

# State Transition Function
g = np.empty_like(f)
for r, s in enumerate(S):
    snext = s - X
    g[r] = getindex(snext, S)


# Model Structure
model = DDPmodel(f, g, delta)
model.solve()

# Analysis
# Plot Optimal Policy
plt.figure()
plt.axes(title='Optimal Extraction Policy', xlabel='Stock', ylabel='Extraction')
plt.plot(S, X[model.policy])

# Plot Value Function
plt.figure()
plt.axes(title='Optimal Value Function', xlabel='Stock', ylabel='Value')
plt.plot(S, model.value)

# Simulate Model
sinit = S.max()
nyrs = 15
t = np.arange(nyrs + 1)
spath, xpath = model.simulate(sinit, nyrs)

# Plot State Path
plt.figure()
plt.axes(title='Optimal State Path', xlabel='Year', ylabel='Stock')
plt.plot(t, S[spath])


plt.show()
