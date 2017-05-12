__author__ = 'Randall'

## DEMDP04 Job Search Model
#
# Infinitely-lived worker must decide whether to quit, if employed, or 
# search for a job, if unemployed, given prevailing market wages.
#
# States
#     w       prevailing wage
#     i       unemployed (0) or employed (1) at beginning of period
# Actions
#     j       idle (0) or active (i.e., work or search) (1) this period
# Parameters
#     v        benefit of pure leisure
#     wbar     long-run mean wage
#     gamma    wage reversion rate
#     p0       probability of finding job
#     p1       probability of keeping job
#     sigma    standard deviation of wage shock
#     delta    discount factor

# Preliminary tasks
from compecon import BasisSpline, DPmodel
from compecon.quad import qnwnorm
from demos.setup import np, plt, demo


## FORMULATION
  
# Model Parameters
u     =  90                            # unemployment benefit
v     =  95                            # benefit of pure leisure
wbar  = 100                           	# long-run mean wage
gamma = 0.40                           # wage reversion rate
p0    = 0.20                         	# probability of finding job
p1    = 0.90                           # probability of keeping job
sigma = 5                             	# standard deviation of wage shock
delta = 0.95                           # discount factor
   
# Continuous State Shock Distribution
m = 15                                	# number of wage shocks
e, w = qnwnorm(m, 0, sigma ** 2)        # wage shocks

# Stochastic Discrete State Transition Probabilities
q = np.zeros((2, 2, 2))
q[1, 0, 1] = p0
q[1, 1, 1] = p1
q[:, :, 0] = 1 - q[:, :, 1]

# Model Structure

# Approximation Structure
n = 150                                # number of collocation nodes
wmin =   0                             # minimum wage
wmax = 200                             # maximum wage
basis = BasisSpline(n, wmin, wmax, labels=['wage'])     # basis functions


def reward(w, x, employed, active):
    if active:
        return w.copy() if employed else np.full_like(w, u)  # the copy is critical!!! otherwise it passes the pointer to w!!
    else:
        return np.full_like(w, v)


def transition(w, x, i, j, in_, e):
    return wbar + gamma * (w - wbar) + e


model = DPmodel(basis, reward, transition,
                i =['unemployed', 'employed'],
                j = ['idle', 'active'],
                discount=delta, e=e, w=w, q=q)

## SOLUTION

# Solve Bellman Equation

S = model.solve(print=True)
ni, nj = model.dims['ni', 'nj']
vr = S['value_j'].reshape(ni, nj, -1)
sr = S['wage'].reshape(ni, nj, -1)[0, 0]

#resid, sr, vr = model.residuals(10)

# Compute and Print Critical Action Wages

wcrit0 = np.interp(0, vr[0, 1] - vr[0, 0], sr)
vcrit0 = np.interp(wcrit0, sr, vr[0,0])
print('Critical Search Wage = {:5.1f}'.format(wcrit0))

wcrit1 = np.interp(0, vr[1, 1] - vr[1, 0], sr)
vcrit1 = np.interp(wcrit1, sr, vr[1,0])
print('Critical Quit Wage   = {:5.1f}'.format(wcrit1))


# Plot Action-Contingent Value Function - Unemployed

demo.figure('Action-Contingent Value, Unemployed', 'Wage', 'Value')
plt.plot(sr, vr[0].T)
demo.annotate(wcrit0, vcrit0, '$w^*_0 = {:.1f}$'.format(wcrit0), 'wo', (5, -5), fs=12)
plt.legend(['Do Not Search', 'Search'], loc='upper left')

# Plot Action-Contingent Value Function - Unemployed

demo.figure('Action-Contingent Value, Employed', 'Wage', 'Value')
plt.plot(sr, vr[1].T)
demo.annotate(wcrit1, vcrit1, '$w^*_0 = {:.1f}$'.format(wcrit1), 'wo',
              (5, -5), fs=12)
plt.legend(['Quit', 'Work'], loc='upper left')

# Plot Residual

demo.figure('Bellman Equation Residual', 'Wage', 'Percent Residual')
S['resid2'] = 100 * (S.resid / S.value)
fig= demo.qplot('wage', 'resid2', 'i',
           data=S,
           geom='line',
           main='Bellman Equation Residual',
           xlab='Wage',
           ylab='Percent Residual')
fig.draw()

resid = S['resid2'].reshape(ni, nj, -1)[0].T
demo.figure('Bellman Equation Residual', 'Wage', 'Percent Residual')
plt.plot(sr,resid)


# SIMULATION

# Simulate Model

T = 40
nrep = 10000
sinit = np.full((1, nrep), wbar)
iinit = 0
data = model.simulate(T, sinit, iinit, seed=945)


# Print Ergodic Moments
ff = '\t{:12s} = {:5.2f}'

print('\nErgodic Means')
print(ff.format('Wage', data['wage'].mean()))
print(ff.format('Employment', (data['i'] == 'employed').mean()))
print('\nErgodic Standard Deviations')
print(ff.format('Wage',data['wage'].std()))
print(ff.format('Employment', (data['i'] == 'employed').std()))


# Plot Expected Discrete State Path

subdata = data[['time', 'i']]
subdata['i'] = subdata['i'] == 'employed'
subdata.groupby('time').mean().plot(legend=False)
plt.title('Probability of Employment')
plt.xlabel('Period')
plt.ylabel('Probability')

# Plot Simulated and Expected Continuous State Path

subdata = data[data['_rep'] < 3][['time', 'wage', '_rep']]
subdata.pivot(index='time', columns='_rep', values='wage').plot(legend=False, lw=1)
plt.hlines(data['wage'].mean(), 0, T)
plt.title('Simulated and Expected Wage')
plt.xlabel('Period')
plt.ylabel('Wage')


plt.show()