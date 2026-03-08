__author__ = 'Randall'

from demos.setup import np, plt, demo
from compecon import DDPmodel

# DEMDDP04 Binomial American put option model

# Model Parameters
T = 0.5                 # years to expiration
sigma = 0.2                 # annual volatility
r = 0.05                # annual interest rate
strike = 2.1                 # option strike price
p0 = 2.0                 # current asset price

# Discretization Parameters
N = 100                 # number of time intervals
tau = T / N              	# length of time intervals
delta = np.exp(-r * tau)   	# discount factor
u = np.exp(sigma * np.sqrt(tau))	# up jump factor
q = 0.5 + np.sqrt(tau) * (r - (sigma**2) / 2) / (2 * sigma) # up jump probability

# State Space
price = p0 * u ** np.arange(-N, N+1)      # asset prices
n = price.size        # number of states

# Action Space (hold=1, exercise=2)
X = ['hold', 'exercise']   	# vector of actions
m = len(X)               	# number of actions

# Reward Function
f = np.zeros((m,n))
f[1] = strike - price

# State Transition Probability Matrix
P = np.zeros((m, n, n))

for i in range(n):
    P[0, i, min(i + 1, n - 1)] = q
    P[0, i, max(i - 1, 0)] = 1 - q

# Model Structure
model = DDPmodel(f, P, delta, horizon=N)
model.solve()
   
## Analysis

# Plot Optimal Exercise Boundary
i, j = np.where(np.diff(model.policy[:-1], 1))
temp = (i * tau)[::-1]
demo.figure('Put Option Optimal Exercise Boundary', 'Time to Maturity', 'Asset Price')
plt.plot(temp, price[j])

# Plot Option Premium vs. Asset Price
demo.figure('Put Option Value', 'Asset Price', 'Premium', [0, 2 * strike])
plt.plot([0, strike],[strike, 0], 'k--', lw=2)
plt.plot(price, model.value[0], lw=3)

plt.show()