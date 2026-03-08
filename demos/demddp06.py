__author__ = 'Randall'


from demos.setup import np, plt, demo.figure
from compecon import DDPmodel
from compecon.tools import gridmake, getindex



## DEMDDP06 Bioeconomic model

# Model Parameters
T     = 10                   # foraging periods
emax  =  8                   # energy capacity
e     = [2,  4,  4]            # energy from foraging
p     = [1.0, 0.7, 0.8]        # predation survival probabilities
q     = [0.5, 0.8, 0.7]        # foraging success probabilities

# State Space
S = np.arange(emax + 1)                # energy levels
n = S.size                # number of states

# Action Space
X = np.arange(3) + 1        # vector of actions
m = X.size               	# number of actions

# Reward Function
f = np.zeros((m, n))

# State Transition Probability Matrix
P = np.zeros((m, n, n))
for k in range(m):
    P[k, 0, 0] = 1
    i = range(1, n)
    # does not survive predation
    snext = 0
    j = getindex(snext, S)
    P[k, i, j] += 1 - p[k]
    # survives predation, finds food
    snext = S[i] - 1 + e[k]
    j = getindex(snext, S)
    P[k, i, j] += p[k] * q[k]
    # survives predation, finds no food
    snext = S[i] - 1
    j = getindex(snext, S)
    P[k, i, j] += p[k] * (1 - q[k])


# Terminal Value Function
vterm = np.ones(n)            # terminal value: survive
vterm[0] = 0                 # terminal value: death

# Model Structure
model = DDPmodel(f, P, 1, T, vterm=vterm)
model.solve()

## Analysis

lims = [-0.5, emax + 0.5], [0, 1]

# Plot Survial Probabilities, Period 0
demo.figure('Survival Probability (Period 0)', 'Stock of Energy', 'Probability', *lims)
plt.bar(S, model.value[0], 1)

# Plot Survial Probabilities, Period 5
demo.figure('Survival Probability (Period 5)', 'Stock of Energy', 'Probability', *lims)
plt.bar(S, model.value[5], 1)

plt.show()
