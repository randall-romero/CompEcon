__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel
from compecon.tools import gridmake


## DEMDDP10 Stochastic cow replacement model

# Model Parameters
delta = 0.9                  # discount factor
cost = 500                  # replacement cost
price = 150                  # milk price

# State Space
s1 = np.arange(10) + 1     # lactation states
s2 = np.array([0.8, 1.0, 1.2])           # productivity states
n1 = s1.size              # number of lactation states
n2 = s2.size              # number of productivity states
S1, S2 = gridmake(s1,s2)    # combined state grid
n = n1 * n2                    # total number of states

# Action Space (keep='K', replace='R')
X = np.array(['Keep','Replace'])          # keep or replace
m = X.size                 	# number of actions

# Reward Function
f = np.empty((m, n))
y = (-0.2 * S1 ** 2 + 2 * S1 +8) * S2  # yield per lactation
f[0] = price * y
f[1] = f[0] - cost
f[0, S1 == 10] = -np.inf           # force replace at lactation 10

# State Transition Probability Matrix
P = np.zeros((2, n1, n2, n1, n2))
for i in range(n1):
    for j in range(n2):
        if i < 9:
            P[0, i, j, i+1, j] = 1     # Raise lactation number by 1, if keep
        else:
            P[0, i, j, 0] = 0.2, 0.6, 0.2

        P[1, i, j, 0] = 0.2, 0.6, 0.2       # Optional replacement

P.shape = 2, n, n


# Model Structure
model = DDPmodel(f, P, delta).solve(print=True)


## Analysis

# Display Optimal Policy
xtemp = model.policy.reshape((n1, n2))
header = '{:^8s} {:^8s}  {:^8s}  {:^8s}'.format('Age','Lo', 'Med', 'Hi')

print('Optimal Policy')
print(header)
print(*('{:^8d} {:^8s}  {:^8s}  {:^8s}\n'.format(s, *X[x]) for s, x in zip(s1, xtemp)))


# Plot Value Function
demo.figure('Optimal Replacement Value', 'Age', 'Optimal Value (thousands)')
plt.plot(s1, model.value.reshape((n1,n2)) / 1000)
plt.legend(['Low','Med','Hi'])



# Compute Steady-State distribution
pi = model.markov().reshape((n1, n2))

# Display Steady-State distribution
print('          Invariant Distribution     ')
print(header)
print(*('{:^8d} {:8.3f}  {:8.3f}  {:8.3f}\n'.format(s, *x) for s, x in zip(s1, pi)))


# Compute Steady-State Mean Cow Age and Productivity
pi = pi.flatten()
avgage = np.dot(pi.T, S1)
avgpri = np.dot(pi.T, S2)
print('Steady-state Age          = {:8.2f}'.format(avgage))
print('Steady-state Productivity = {:8.2f}'.format(avgpri))


plt.show()