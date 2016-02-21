__author__ = 'Randall Romero'


## DEMDP02 Asset Replacement Model
#
# Profit-maximizing entrepreneur must decide when to replace an aging asset.
#
# States
#     p       unit profit
#     a       asset age (1..A)
# Actions
#     j       keep(1) or replace(2) asset
# Parameters
#     A       maximum asset age 
#     alpha   production function coefficients
#     kappa   net replacement cost
#     pbar    long-run mean unit profit
#     gamma   unit profit autoregression coefficient
#     sigma   standard deviation of unit profit shock
#     delta   discount factor 

# Preliminary tasks

from compecon import BasisSpline, DPmodel
from compecon.quad import qnwnorm
from demos.setup import np, plt, demo
from warnings import simplefilter
simplefilter('ignore')


## FORMULATION
  
# Model Parameters
A       = 6                                # maximum asset age 
alpha   = np.array([50, -2.5, -2.5])       # production function coefficients
kappa   = 40                               # net replacement cost
pbar    = 1                                # long-run mean unit profit
gamma   = 0.5                              # unit profit autoregression coefficient
sigma   = 0.15                             # standard deviation of unit profit shock
delta   = 0.9                              # discount factor 

# Continuous State Shock Distribution
m = 5                                      # number of unit profit shocks
[e,w] = qnwnorm(m,0,sigma ** 2)               # unit profit shocks and probabilities

# Deterministic Discrete State Transitions
h = np.zeros((2, A))
h[0, :-1] = np.arange(1, A)

# Approximation Structure
n  = 200                                   # number of collocation nodes
pmin = 0                                   # minimum unit profit
pmax = 2                                   # maximum unit profit
basis = BasisSpline(n, pmin, pmax, labels=['unit profit'])        # basis functions

# Model Structure

def profit(p, x, i, j):
    a = i + 1
    if j or a == A:
        return p * 50 - kappa
    else:
        return p * (alpha[0] + alpha[1] * a + alpha[2] * a ** 2 )

def transition(p, x, i, j, in_, e):
    return pbar + gamma * (p - pbar) + e

model = DPmodel(basis, profit, transition,
                # i=['a={}'.format(a+1) for a in range(A)],
                i=[a + 1 for a in range(A)],
                j=['keep', 'replace'],
                discount=delta, e=e, w=w, h=h)

# SOLUTION

S = model.solve()

pr = np.linspace(pmin, pmax, 10 * n)

# Plot Action-Contingent Value Functions

pp = demo.qplot('unit profit', 'value_j', 'i',
      data=S,
      main='Action-Contingent Value Functions',
      xlab='Net Unit Profit',
      ylab='Value')


print(pp)



'''

# color_list = plt.cm.Set3(np.linspace(0, 1, 8))

demo.figure('Action-Contingent Value Functions', 'Net Unit Profit', 'Value')
plt.plot(pr, model.Value_j[:, 'keep'](pr).T)
plt.legend(model.labels.i, loc='upper left')


# Compute and Plot Critical Unit Profit Contributions
vr = model.Value_j(pr)

print('Critical Replacement Profit\n')
for a in range(A-1):
    pcrit = np.interp(0.0, vr[a, 1] - vr[a, 0], pr, np.nan, np.nan)  # interp only works with increasing x data
    vcrit = np.interp(pcrit, pr, vr[a, 0])
    if np.isnan(pcrit):
        continue

    demo.annotate(pcrit, vcrit, '$p^*_' + str(a+1) + '$', 'wo',
                  (0, 0), fs=11, ms=18)
    print('   Age {:2d}  Profit {:5.2f}'.format(a, pcrit))

'''
# Plot Residual
S['resid2'] = 100*S.resid / S.value
print(demo.qplot('unit profit', 'resid2','i',
            data=S,
            geom='line',
            main='Bellman Equation Residual',
            xlab='Net Unit Profit',
            ylab='Percent Residual'))




'''
demo.figure('Bellman Equation Residual', 'Net Unit Profit', 'Percent Residual')
plt.plot(pr, 100 * (resid / model.Value(pr)).T)
# plot(pr,0*pr,'k--')
plt.legend(model.labels.i, loc='upper right')
'''

## SIMULATION

# Simulate Model

T = 50
nrep = 10000
sinit = np.full(nrep, pbar)
iinit = 0
data = model.simulate(T,sinit,iinit, seed=945)

# Print Ergodic Moments
frm = '\t{:<10s} = {:5.2f}'

print('\nErgodic Means')
print(frm.format('Price', data['unit profit'].mean()))
print(frm.format('Age', data.i.mean()))
print('\nErgodic Standard Deviations')
print(frm.format('Price', data['unit profit'].std()))
print(frm.format('Age', data.i.std()))


# Plot Simulated and Expected Continuous State Path
print(demo.qplot('time', 'unit profit', '_rep',
      data=data[data['_rep'] < 3],
            geom='line',
      main='Simulated and Expected Price',
      ylab='Net Unit Profit',
      xlab='Period'))

# Plot Expected Discrete State Path
data[['time', 'i']].groupby('time').mean().plot(legend=False)
plt.title('Expected Machine Age')
plt.xlabel('Period')
plt.ylabel('Age')
plt.show()
