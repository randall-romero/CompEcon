__author__ = 'Randall'

## DEMDP10 Water Resource Management Model
#
# Public authority must decide how much water to release from a reservoir so 
# as to maximize benefits derived from agricultural and recreational uses.
#
# States
#     s       reservoiur level at beginning of summer
# Actions
#     x       quantity of water released for irrigation
# Parameters
#     a       producer benefit function parameters
#     b       recreational user benefit function parameters
#     delta   discount factor

# Preliminary tasks
from compecon import BasisChebyshev, DPmodel
from demos.setup import np, plt, demo
from compecon.quad import qnwlogn

## FORMULATION

# Model Parameters
a = [1, -2]                                 # producer benefit function parameters
b = [2, -3]                                 # recreational user benefit function parameters
ymean = 1.0                                # mean rainfall
sigma = 0.2                                # rainfall volatility
delta = 0.9                                # discount factor

# Continuous State Shock Distribution
m = 3                                      # number of rainfall shocks
e, w = qnwlogn(m,np.log(ymean)-sigma ** 2/2,sigma ** 2) # rainfall shocks and proabilities



# Approximation Structure
n    = 15                                  # number of collocation nodes
smin =  2                                  # minimum state
smax =  10                                  # maximum state
basis = BasisChebyshev(n, smin, smax, labels=['reservoir'])        # basis functions


def bounds(s, i, j):
  return np.zeros_like(s), s.copy()


def reward(s, x, i, j):
    a0, a1 = a
    b0, b1 = b
    f = 1.2 * (a0 / (1 + a1)) *x ** (1 + a1) + (b0 / (1 + b1)) * (s-x) ** (1+b1)
    fx = 1.2 * a0 * x ** a1 - b0 * (s-x) ** b1
    fxx = 1.2 * a0 * a1 * x ** (a1-1) + b0 * b1 * (s-x) ** (b1 - 1)
    return f, fx, fxx

def transition(s, x, i, j, in_, e):
    g = s - x + e
    gx = -np.ones_like(s)
    gxx = np.zeros_like(s)
    return g, gx, gxx

# Model object

model = DPmodel(basis, reward, transition, bounds,
                x=['released'], discount=delta, e=e, w=w)

# Deterministic Steady-State
xstar = 1                                  # deterministic steady-state action
sstar = 1 + (a[0] *(1-delta)/b[0]) ** (1 / b[1])   # deterministic steady-state stock

# Check Model Derivatives
# dpcheck(model,sstar,xstar)


## SOLUTION

# Compute Linear-Quadratic Approximation at Collocation Nodes
model.lqapprox(sstar, xstar)

# Solve Bellman Equation
model.solve() # no need to pass LQ to model, it's already there
resid, s, v, x = model.solution()

# Plot Optimal Policy
demo.figure('Optimal Irrigation Policy', 'Reservoir Level', 'Irrigation')
plt.plot(s, x.T)

# Plot Value Function
demo.figure('Value Function', 'Reservoir Level', 'Value')
plt.plot(s, v.T)

# Plot Shadow Price Function
demo.figure('Shadow Price Function', 'Reservoir Level', 'Shadow Price')
plt.plot(s, model.Value(s, 1).T)

# Plot Residual
demo.figure('Bellman Equation Residual', 'Reservoir Level', 'Residual')
plt.plot(s,resid.T)
plt.hlines(0, smin, smax, 'k', '--')


## SIMULATION

# Simulate Model
T = 30
nrep = 100000
sinit = np.full((1, nrep), smin)
data = model.simulate(T, sinit, seed=945)

# Plot Simulated State Path
D = data[data['_rep'] < 3][['time', 'reservoir', '_rep']]
D.pivot(index='time', columns='_rep', values='reservoir').plot(legend=False, lw=1)
data.groupby('time')['reservoir'].mean().plot(color='k', linestyle='--')
plt.title('Simulated and Expected Reservoir Level')
plt.xlabel('Year')
plt.ylabel('Reservoir Level')

# Plot Simulated Policy Path
D = data[data['_rep'] < 3][['time', 'released', '_rep']]
D.pivot('time', '_rep', 'released').plot(legend=False, lw=1)
data.groupby('time')['released'].mean().plot(color='k', linestyle='--')
plt.title('Simulated and Expected Irrigation')
plt.xlabel('Year')
plt.ylabel('Irrigation')

# Print Steady-State and Ergodic Moments
ff = '\t%15s = %5.2f'

D = data[data['time'] == T]

print('\nDeterministic Steady-State')
print(ff % ('Reservoir Level', sstar))
print(ff % ('Irrigation', xstar))
print('\nErgodic Means')
print(ff % ('Reservoir Level', D['reservoir'].mean()))
print(ff % ('Irrigation', D['released'].mean()))
print('\nErgodic Standard Deviations')
print(ff % ('Reservoir Level', D['reservoir'].std()))
print(ff % ('Irrigation', D['released'].std()))


# Compute and Plot Ergodic State Distribution

demo.figure('Ergodic Reservoir Level Distribution', 'Reservoir Level', 'Probability', [2, 6])
D['reservoir'].plot.kde()


plt.show()