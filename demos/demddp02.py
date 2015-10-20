__author__ = 'Randall'


from demos.setup import np, plt
from compecon import DDPmodel


## DEMDDP02 Asset replacement model

# Model Parameters
maxage  = 5                  # maximum machine age
repcost = 75               	# replacement cost
delta   = 0.9                # discount factor

# State Space
S = np.arange(1, 1 + maxage)  # machine age
n = S.size                	# number of states

# Action Space (keep=1, replace=2)
X = np.array([1,2])     	# vector of actions
m = X.size               	# number of actions



# Reward Function
f = np.c_[50 - 2.5 * S - 2.5 * S ** 2, np.repeat(50 - repcost, n)]
f[-1, 0] = -np.inf

# State Transition Function
g = np.empty_like(f)
for i in range(n):
    g[i] = np.minimum(1 + i, n - 1), 0  # keep, replace


# Model Structure
model = DDPmodel(f, g, delta)
model.solve()


## Analysis

# Plot Optimal Value
plt.figure()
plt.axes(title='Optimal Value Function', xlabel='Age of Machine', ylabel='Value')
plt.plot(S, model.value)

# Simulate Model
sinit, nyrs = S.min() - 1, 12
t = np.arange(1 + nyrs)
spath, xpath = model.simulate(sinit, nyrs)

# Plot State Path
plt.figure()
plt.axes(title='Optimal State Path', xlabel='Year', ylabel='Age of Machine', xlim=[0, 12])
plt.plot(t, S[spath])
plt.show()