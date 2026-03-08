__author__ = 'Randall'


from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import gridmake, getindex

## DEMDDP07 Renewable resource model

# Model Parameters
delta =  0.9                  # discount factor
alpha =  4.0                  # growth function parameter
beta  =  1.0                  # growth function parameter
gamma =  0.5                  # demand function parameter
cost  =  0.2                  # unit cost of harvest

# State Space
smin = 0                      # minimum state
smax = 8                      # maximum state
n = 200                    # number of states
S = np.linspace(smin, smax, n)  # vector of states

# Action Space
xmin = 0                      # minimum action
xmax = 6                      # maximum action
m = 100                    # number of actions
X = np.linspace(xmin, xmax, m)  # vector of actions


# Reward Function
f = np.full((m, n), -np.inf)
for k in range(m):
    f[k, S >= X[k]] = (X[k] ** (1 - gamma)) / (1 - gamma) - cost * X[k]

# State Transition Function

g = np.zeros_like(f)
for k in range(m):
    snext = alpha * (S - X[k]) - 0.5 * beta * (S - X[k]) ** 2
    g[k] = getindex(snext, S)



# Model Structure
model = DDPmodel(f, g, delta)
model.solve()
   

## Analysis

# Plot Optimal Policy
demo.figure('Optimal Harvest Policy', 'Stock', 'Harvest')
plt.plot(S,X[model.policy])


# Plot Value Function
demo.figure('Optimal Value Function', 'Stock', 'Value')
plt.plot(S,model.value)


# Simulate Model
nyrs = 20
t = np.arange(nyrs + 1)
spath, xpath = model.simulate(n - 1, nyrs)

# Plot State Path
demo.figure('Optimal State Path', 'Year', 'Stock')
plt.plot(t, S[spath])

# Plot Optimal Transition Function
demo.figure('Optimal State Transition Function', 'S(t)', 'S(t+1)')
ii, jj = np.where(model.transition)
plt.plot(S[ii], S[jj], S, S, '--')

plt.show()