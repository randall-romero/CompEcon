__author__ = 'Randall Romero'


from compecon import DPmodel, BasisSpline
from demos.setup import np, plt, demo



## DEMDP01 Timber Harvesting Model - Cubic Spline Approximation
#
# Profit maximizing owner of a commercial tree stand must decide when to
# clearcut the stand.
#
# States
#     s       stand biomass
# Actions
#     j       clear cut (2) or do not clear cut (1)
# Parameters
#     price   unit price of biomass
#     kappa   clearcut-replant cost
#     smax    stand carrying capacity
#     gamma   biomass growth parameter
#     delta   discount factor



## FORMULATION

# Model Parameters
price = 1.0 / 2                        # unit price of biomass
kappa = 0.2                            # clearcut-replant cost
smax  = 0.5                            # stand carrying capacity
gamma = 0.1                            # biomass growth parameter
delta = 0.9                            # discount factor

# Approximation Structure
n = 200                                # number of collocation nodes
basis = BasisSpline(n, 0, smax, labels=['stand biomass'])        # basis functions


# Model Structure

def reward(s, x, i , j):
    return (price * s - kappa) * j

def transition(s, x, i, j, in_, e):
    if j:
        return np.full_like(s, gamma * smax)
    else:
        return s + gamma * (smax - s)


model = DPmodel(basis, reward, transition,
                discount=delta,
                j=['keep', 'clear cut'])

model.solve()
resid, sr, vr = model.residuals()

# Plot Action-Contingent Value Functions


demo.figure('Action-Contingent Value Functions', 'Biomass', 'Value of Stand')
plt.plot(sr, vr.T)
plt.legend(model.labels.j, loc='upper center')

# Compute and Plot Optimal Harvesting Stock Level
scrit = np.interp(0.0, vr[1] - vr[0], sr)
vcrit = np.interp(scrit, sr,vr[0])
demo.annotate(scrit, vcrit, '$s^* = {:.2f}$'.format(scrit), 'wo', (-5, 5), fs=12)

print('Optimal Biomass Harvesting Level = {:5.2f}'.format(scrit))




# Plot Residual
demo.figure('Bellman Equation Residual', 'Biomass', 'Percent Residual')
plt.plot(sr, 100 * resid.T / vr.max(0).T)





## SIMULATION

# Simulate Model
T = 50      # Number of periods simulated
sinit = 0.0      # Initial value of continuous state
data = model.simulate(T, sinit)

# Compute Optimal Rotation Cycle
print('Optimal Rotation Cycle = ', np.min(data.time[data.j == 'clear cut']))

# Plot State Path

data.plot('time', 'stand biomass', legend=False, title='Rotation cycle')
plt.ylabel('Biomass')

plt.show()
