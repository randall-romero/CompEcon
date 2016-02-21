__author__ = 'Randall'

## DEMDP03 Industry Entry-Exit Model
#
# A profit maximizing firm must decide whether to operate or shut down, 
# given its short-run profitability, subject to transactions costs.
#
# States
#     p       current short-run profit
#     i       active (1) or idle (0) last period
# Actions
#     j       active (1) or idle (0) this period
# Parameters
#     pbar    long-run mean profit
#     gamma   profit autoregressive coefficient
#     kappa   cost of reopenning idle firm
#     sigma   standard deviation of profit shock
#     delta   discount factor
  
# Preliminary tasks
from compecon import BasisSpline, DPmodel
from compecon.quad import qnwnorm
from demos.setup import np, plt, demo
import pandas as pd
from warnings import simplefilter
simplefilter('ignore')
from ggplot import ggplot, aes, geom_line, geom_point

## FORMULATION
  
# Model Parameters
pbar   = 1.0                               # long-run mean profit
gamma  = 0.7                               # profit autoregressive coefficient
kappa  =  10                             	# cost of reopenning idle firm
sigma  = 1.0                               # standard deviation of profit shock
delta  = 0.9                             	# discount factor
  
# Continuous State Shock Distribution
m = 5                                      # number of profit shocks
[e,w] = qnwnorm(m,0,sigma **2)              # profit shocks and probabilities

# Approximation Structure
n = 250                                    # number of collocation nodes
pmin = -20                                 # minimum profit
pmax =  20                                 # maximum profit
basis = BasisSpline(n, pmin, pmax)        # basis functions

def profit(p, x, i, j):
  return p * j - kappa * (1 - i) * j

def transition(p, x, i, j, in_, e):
  return pbar + gamma * (p - pbar) + e


# Model
model = DPmodel(BasisSpline(n, pmin, pmax, labels=['profit']),
                profit, transition,
                i=['idle', 'active'],
                j=['idle', 'active'],
                discount=delta, e=e, w=w,
                h=[[0, 0], [1, 1]])


## SOLUTION

# Solve Bellman Equation
S = model.solve(print=True)

# Plot Action-Contingent Value Functions

S['ij'] = pd.Categorical(S.i.astype(np.ndarray) + '--' + S.j.astype(np.ndarray))
S.ij.cat.categories = ['Keep active firm open', 'Shut down', 'Reopen Firm', 'Remain idle']
print(demo.qplot('profit', 'value_j', 'ij',
                 data=S[S.ij != 'Remain idle'],
                 geom='line')
     )



'''
demo.figure('Action-Contingent Value Functions', 'Potential Profit', 'Value of Firm')

#  Compute and Plot Critical Profit
offs = [(4, -6), (-8, 5)]

for i in range(2):
    pcrit = np.interp(0, vr[i, 1] - vr[i, 0], sr)
    vcrit  = np.interp(pcrit, sr, vr[i,0])
    demo.annotate(pcrit, vcrit, '$p^*_{} = {:.2f}$'.format(i, pcrit), 'wo', offs[i], fs=11, ms=16,)
    if i:
        print('Profit Exit  = {:5.2f}'.format(pcrit))
    else:
        print('Profit Entry = {:5.2f}'.format(pcrit))

'''

# Plot Residual
S['resid2'] = 100 * (S.resid / S.value)
print(demo.qplot('profit','resid2','i',
                 data=S,
                 geom='line',
                 main='Bellman Equation Residual',
                 xlab='Potential Profit',
                 ylab='Percent Residual'))


# SIMULATION

# Simulate Model
# rand('seed',0.945)
T = 50
nrep = 500#00
pinit = pbar * np.ones((1, nrep))
iinit = 1
data = model.simulate(T, pinit, iinit, seed=945)


# Print Ergodic Moments
f = '\t{:18s} = {:5.2f}'
print('\nErgodic Means')
print(f.format('Profit Contribution', data['profit'].mean()))
print(f.format('Activity', (data['i'] == 'active').mean()))
print('\nErgodic Standard Deviations\n')
print(f.format('Profit Contribution', data['profit'].std()))
print(f.format('Activity', (data['i'] == 'active').std()))


# Plot Simulated and Expected Continuous State Path
data2 = data[['time', 'profit']].groupby('time').mean()
data2['time'] = data2.index

print(data2)
print(data2.columns)

ppp = ggplot(aes('time','profit'),
             data=data2) + \
      geom_line()

print(ppp)

ppp = ggplot(aes('time','profit','_rep'),
             data=data[data._rep <3]) + \
      geom_point() + \
      geom_line(aes('time','profit'), data=data2)

print(ppp)

print(demo.qplot('time','profit','_rep',
                 data=data[data._rep <3],
                 geom='line') +
      geom_line(aes('time','profit'),data=data2
                 ))

'''
subdata = data[data['_rep'] < 3][['time', 'profit', '_rep']]
subdata.pivot(index='time', columns='_rep', values='profit').plot(legend=False, lw=1)
plt.hlines(data['profit'].mean(), 0, T)
plt.title('Simulated and Expected Profit Contribution')
plt.xlabel('Period')
plt.ylabel('Profit Contribution')


# Plot Expected Discrete State Path

subdata = data[['time', 'i']]
subdata['i'] = subdata['i'] == 'active'
subdata.groupby('time').mean().plot(legend=False)
plt.title('Probability of Operation')
plt.xlabel('Period')
plt.ylabel('Probability')

plt.show()
'''