__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import gridmake, getindex

# DEMDDP03 Asset replacement model with maintenance

# Model Parameters
maxage  = 5                  # maximum asset age
repcost = 75                 # replacement cost
mancost = 10               	# maintenance cost
delta   = 0.9                # discount factor

# State Space
s1 = np.arange(1, 1 + maxage)   # asset age
s2 = s1 - 1   	                # servicings
S  = gridmake(s1,s2)     	# combined state grid
S1, S2 = S
n = S1.size                  	# total number of states

# Action Space
X = ['no action', 'service', 'replace']	 # vector of actions
m = len(X)              	# number of actions

# Reward Function
f = np.zeros((m, n))
q = 50 - 2.5 * S1 - 2.5 * S1 ** 2
f[0] = q * np.minimum(1, 1 - (S1 - S2) / maxage)
f[1] = q * np.minimum(1, 1 - (S1 - S2 - 1) / maxage) - mancost
f[2] = 50 - repcost


# State Transition Function
g = np.empty_like(f)
g[0] = getindex(np.c_[S1 + 1, S2], S)
g[1] = getindex(np.c_[S1 + 1, S2 + 1], S)
g[2] = getindex(np.c_[1, 0], S)

# Model Structure
model = DDPmodel(f, g, delta)
model.solve()
   


## Analysis

# Simulate Model
sinit = 0
nyrs = 12
t = np.arange(nyrs + 1)
spath, xpath = model.simulate(sinit, nyrs)

# Plot State Path (Age)
demo.figure('Optimal State Path', 'Year', 'Age of Asset', [0, 12])
plt.plot(t, S1[spath])

# Plot State Path (Servicings)
demo.figure('Optimal State Path', 'Year', 'Number of Servicings', [0, 12], [0, 2.25])
plt.plot(t, S2[spath])

plt.show()