__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import gridmake, getindex


# DEMDDP05 Water management model

# Model Parameters
alpha1 = 14                  # producer benefit function parameter
beta1  = 0.8                 # producer benefit function parameter
alpha2 = 10                  # recreational user benefit function parameter
beta2  = 0.4                 # recreational user benefit function parameter
maxcap = 30                  # maximum dam capacity
r = np.array([0, 1, 2, 3, 4])  # rain levels
p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])    # rain probabilities
delta  = 0.9                 # discount factor

# State Space
S = np.arange(1 + maxcap)    # vector of states
n = S.size                   # number of states

# Action Space
X = np.arange(1 + maxcap)    # vector of actions
m = X.size                   # number of actions

# Reward Function
f = np.full((m, n), -np.inf)
for k in range(m):
    f[k, k:] = alpha1 * X[k] ** beta1 + alpha2 * (S[k:] - X[k]) ** beta2


# State Transition Probability Matrix
P = np.zeros((m, n, n))

for k in range(m):
    for i in range(n):
        for j in range(r.size):
            snext = min(S[i] - X[k] + r[j], maxcap)
            inext = getindex(snext, S)
            P[k, i, inext] = P[k, i, inext] + p[j]


# Model Structure
model = DDPmodel(f, P, delta)
model.solve()

## Analysis

# Plot Optimal Policy
demo.figure('Optimal Irrigation Policy', 'Water Level', 'Irrigation', [-1, 31], [0, 6])
plt.plot(S, X[model.policy], '*')

# Plot Value Function
demo.figure('Optimal Value Function', 'Water Level', 'Value')
plt.plot(S, model.value)

# Simulate Model
sinit = np.zeros(10000)
nyrs  = 30
t = np.arange(1 + nyrs)
spath, xpath = model.simulate(sinit, nyrs)

# Plot State Path
demo.figure('Optimal State Path', 'Year', 'Water Level')
plt.plot(t, S[spath].mean(1))


# Compute Steady-State Distribution of Water Level
pi = model.markov()
demo.figure('Steady State Distribution', 'Water Level', 'Probability', [-1, 31], [0, 0.16])
plt.bar(S, pi, 1)

plt.show()

# Compute Steady-State Mean Water Level
avgstock = np.inner(pi.flatten(), S)
print('\nSteady-state Stock        {:8.2f}\n'.format(avgstock))