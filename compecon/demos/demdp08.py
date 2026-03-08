__author__ = 'Randall'

## DEMDP08 Public Renewable Resource Model
#
# Welfare maximizing social planner must decide how much of a renewable 
# resource to harvest.
#
# States
#     s       quantity of stock available
# Actions
#     q       quantity of stock harvested
# Parameters
#     alpha   growth function parameter
#     beta    growth function parameter
#     gamma   relative risk aversion
#     kappa   unit cost of harvest
#     delta   discount factor

# Preliminary tasks
from compecon import BasisChebyshev, DPmodel
from demos.setup import np, plt, demo

## FORMULATION
  
# Model Parameters
alpha = 4.0                                    # growth function parameter
beta  = 1.0                                  	# growth function parameter
gamma = 0.5                                   	# relative risk aversion
kappa = 0.2                                    # unit cost of harvest
delta = 0.9                                  	# discount factor


# Approximation Structure
n    = 8                                     	# number of collocation nodes
smin = 6                                    	# minimum state
smax = 9                                      # maximum state
basis = BasisChebyshev(n,smin,smax, labels=['available'])            # basis functions


# Model Structure


def bounds(s, i, j):
    return np.zeros_like(s), s[:]  # important!!, pass a copy of s


def reward(s, q, i, j):
  f = (q ** (1 - gamma)) / (1-gamma) - kappa * q
  fx = q ** ( - gamma) - kappa
  fxx = - gamma * q ** (-gamma - 1)
  return f, fx, fxx


def transition(s, q, i, j, in_, e):
    g = alpha * (s - q) - 0.5 * beta * (s - q) ** 2
    gx = -alpha + beta * (s - q)
    gxx = -beta* np.zeros_like(s)
    return g, gx, gxx


model = DPmodel(basis, reward, transition, bounds,
                x=['harvest'], discount=delta)

# Steady-State

sstar = (alpha ** 2 - 1 / delta ** 2) / (2 * beta)      # steady-state stock
qstar = sstar - (delta * alpha - 1) / (delta * beta) 	# steady-state action

# Print Steady-States
print('Steady States')
print('\tStock   = %5.2f' % sstar)
print('\tHarvest = %5.2f' % qstar)

# Check Model Derivatives
# dpcheck(model,sstar,qstar)


## SOLUTION

# Solve Bellman Equation
model.solve()
resid, s, v, q = model.solution()

# Plot Optimal Policy
demo.figure('Optimal Harvest Policy', 'Stock', 'Harvest')
plt.plot(s, q.T)

# Plot Value Function
demo.figure('Value Function', 'Stock', 'Value')
plt.plot(s, v.T)

# Plot Shadow Price Function
demo.figure('Shadow Price Function', 'Stock', 'Shadow Price')
plt.plot(s, model.Value(s, 1).T)

# Plot Residual
demo.figure('Bellman Equation Residual','Stock', 'Residual')
plt.plot(s, resid.T)
plt.hlines(0, smin, smax, 'k', '--')


## SIMULATION

# Simulate Model
T = 15
data = model.simulate(T, smin)
print(data)

# Plot State and Policy Paths
opts = dict(spec='r*', offset=(0,-5), fs=11, ha='right')

data[['available', 'harvest']].plot()
demo.annotate(T, sstar, 'steady-state stock = %.2f' % sstar, **opts)
demo.annotate(T, qstar, 'steady-state harvest = %.2f' % qstar, **opts)
plt.xlim([0, T + 0.25])
plt.title('State and Policy Paths')
plt.xlabel('Period')
plt.ylabel('Stock / Harvest')
plt.legend(['Stock','Harvest'], loc='right')

plt.show()