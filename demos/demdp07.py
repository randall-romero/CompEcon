__author__ = 'Randall'

## DEMDP07 Stochastic Optimal Economic Growth Model
#
# Welfare maximizing social planner must decide how much society should
# consume and invest. Unlike the deterministic model, this model allows
# arbitrary constant relative risk aversion, capital depreciation, and 
# stochastic production shocks.  It lacks a known closed-form solution.
#
# States
#     s       stock of wealth
# Actions
#     k       capital investment
# Parameters
#     alpha   relative risk aversion
#     beta    capital production elasticity
#     gamma   capital survival rate
#     sigma   production shock volatility
#     delta   discount factor

# Preliminary tasks
from compecon import BasisChebyshev, DPmodel
from compecon.quad import qnwlogn
from demos.setup import np, plt, demo


## FORMULATION
  
# Model Parameters
alpha = 0.2                             # relative risk aversion
beta  = 0.5                             # capital production elasticity
gamma = 0.9                           	# capital survival rate
sigma = 0.1                             # production shock volatility
delta = 0.9                             # discount factor


# Approximation Structure
n     = 10                              # number of collocation nodes
smin  =  5                              # minimum wealth
smax  = 10                              # maximum wealth
basis = BasisChebyshev(n, smin, smax, labels=['wealth'])     # basis functions

# Continuous State Shock Distribution
m = 5                                  	# number of production shocks
e, w = qnwlogn(m,-sigma ** 2/2,sigma ** 2)  	# production shocks and probabilities

# Model specification

def bounds(s, i, j):
    return np.zeros_like(s), 0.99 * s

def reward(s, k, i, j):
    sk = s - k
    f = (sk ** (1 - alpha)) / (1 - alpha)
    fx = - sk ** -alpha
    fxx = -alpha * sk ** (-alpha - 1)
    return f, fx, fxx

def transition(s, k, i, j, in_, e):
    g = gamma * k + e * k ** beta
    gx = gamma + beta * e * k **(beta - 1)
    gxx = (beta - 1) * beta * e * k ** (beta - 2)
    return g, gx, gxx


growth = DPmodel(basis, reward, transition, bounds,
                 x=['investment'], discount=delta, e=e, w=w)


# Deterministic Steady-State
kstar = ((1 - delta * gamma) / (delta * beta)) ** (1 / (beta - 1))  # determistic steady-state capital investment
sstar = gamma * kstar + kstar ** beta       	# deterministic steady-state wealth

# Check Model Derivatives
# dpcheck(model,sstar,kstar)


## SOLUTION

# Solve Bellman Equation
growth.solve()
resid, s, v, k = growth.solution()

# Plot Optimal Policy
demo.figure('Optimal Investment Policy',  'Wealth', 'Investment')
plt.plot(s, k.T)

# Plot Value Function
demo.figure('Value Function', 'Wealth', 'Value')
plt.plot(s, v.T)

# Plot Shadow Price Function
demo.figure('Shadow Price Function', 'Wealth', 'Shadow Price')
plt.plot(s, growth.Value(s, order=1).T)



# Plot Residual
demo.figure('Bellman Equation Residual', 'Wealth', 'Residual')
plt.plot(s, resid.T)
plt.hlines(0, smin, smax, 'k', '--')


## SIMULATION

# Simulate Model
T = 20
nrep = 50000
sinit = np.full((1, nrep), smin)

data = growth.simulate(T, sinit)




# Plot Simulated State Path

subdata = data[data['_rep'] < 3][['time', 'wealth', '_rep']]
subdata.pivot(index='time', columns='_rep', values='wealth').plot(legend=False, lw=1)
data.groupby('time')['wealth'].mean().plot(color='k', linestyle='--')
plt.title('Simulated and Expected Wealth')
plt.xlabel('Period')
plt.ylabel('Wealth')


# Plot Simulated Policy Path

subdata = data[data['_rep'] < 3][['time', 'investment', '_rep']]
subdata.pivot(index='time', columns='_rep', values='investment').plot(legend=False, lw=1)
plt.plot(data.groupby('time')['investment'].mean(), 'k-')
plt.title('Simulated and Expected Investment')
plt.xlabel('Period')
plt.ylabel('Investment')


# Print Steady-State and Ergodic Moments
ff = '\t%15s = %5.2f'
D = data[data['time'] == T][['wealth', 'investment']]

print('\nDeterministic Steady-State')
print(ff % ('Wealth', sstar))
print(ff % ('Investment', kstar))
print('\nErgodic Means\n')
print(ff % ('Wealth', D['wealth'].mean()))
print(ff % ('Investment', D['investment'].mean()))
print('\nErgodic Standard Deviations\n')
print(ff % ('Wealth', D['wealth'].std()))
print(ff % ('Investment', D['investment'].std()))


# Compute and Plot Ergodic Wealth Distribution

demo.figure('Ergodic Wealth Distribution', 'Wealth', 'Probability', [smin, smax])
D['wealth'].plot.kde()

plt.show()
