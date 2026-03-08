__author__ = 'Randall'


from demos.setup import np, plt, demo
from compecon import DDPmodel


## DEMDDP02 Asset replacement model

# Model Parameters
maxage  = 5                  # maximum machine age
repcost = 75               	# replacement cost
delta   = 0.9                # discount factor

# State Space
S = np.arange(1, 1 + maxage)  # machine age
n = S.size                	  # number of states

# Action Space (keep=1, replace=2)
X = ['keep', 'replace']     	# vector of actions
m = len(X)                  	# number of actions


# Reward Function
f = np.zeros((m, n))
f[0] = 50 - 2.5 * S - 2.5 * S ** 2
f[1] = 50 - repcost
f[0, -1] = -np.inf

# State Transition Function
g = np.zeros_like(f)
g[0] = np.arange(1, n + 1)
g[0, -1] = n - 1  # adjust last state so it doesn't go out of bounds

# Model Structure
model = DDPmodel(f, g, delta)
model.solve()


## Analysis

# Simulate Model
sinit, nyrs = S.min() - 1, 12
t = np.arange(1 + nyrs)
spath, xpath = model.simulate(sinit, nyrs)

# Plot Optimal Value
demo.figure('Optimal Value Function', 'Age of Machine', 'Value')
plt.plot(S, model.value)

# Plot State Path
demo.figure('Optimal State Path', 'Year', 'Age of Machine', [0, 12])
plt.plot(t, S[spath])

plt.show()