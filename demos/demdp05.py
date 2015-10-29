__author__ = 'Randall'


## DEMDP05 American Put Option Pricing Model
#
# Compute the critical exercise price for an American put option in terms 
# of time to expiration.
#
# States
#     p       underlying asset price
# Actions
#     j       exercize (2) or do not exercize (1) option
# Parameters
#     K       option strike price
#     N       number of periods to expiration
#     mu      mean of log price innovation
#     sigma   standard devitation of log price innovation
#     delta   discount factor

# Preliminary tasks
from compecon import BasisSpline, NLP
from compecon.quad import qnwnorm
from demos.setup import np, plt, demo

## FORMULATION

# Continuous Time Discretization
# sigma = 0.2                            # annual volatility
# T     = 0.5                            # years to expiration
# K     = 1.0                            # option strike price
# r     = 0.1                            # annual interest rate
# N     = 300                            # number of time intervals
# dt    = T/N                            # length of time interval
# delta = exp(-r*dt)                     # per period discount factor
# mu    = dt*(r-sigma ** 2/2)               # mean of log price innovation
  
# Model Parameters
K     = 1.0                            # option strike price
N     = 300                            # number of periods to expiration
mu    = 0.0001                         # mean of log price innovation
sigma = 0.0080                         # standard devitation of log price innovation
delta = 0.9998                         # discount factor
  
# Continuous State Shock Distribution
m = 15                                 # number of price shocks
[e,w] = qnwnorm(m,mu,sigma ** 2)          # price shocks and probabilities
  
# Approximation Structure
n    = 500                             # number of collocation nodes
pmin  = -1                             # minimum log price
pmax  =  1                             # maximum log price
Value = BasisSpline(n, pmin, pmax,
                    labels=['logprice'], l=['value'])   # basis functions
# p     = funnode(basis)                 # collocaton nodes
# Phi   = funbase(basis)                 # interpolation matrix


## SOLUTION
  
# Intialize Value Function
# c = zeros(n,1)                         # conditional value function basis coefficients

# Solve Bellman Equation and Compute Critical Exercise Prices
f = NLP(lambda p: K - np.exp(p) - delta * Value(p))

pcrit = np.empty(N + 1)

pcrit[0] = f.zero(0.0)

for t in range(N):
    v = np.zeros((1, n))
    for k in range(m):
        pnext = Value.nodes + e[k]
        v += w[k] * np.maximum(K - np.exp(pnext), delta * Value(pnext))

    Value[:] = v
    pcrit[t + 1] = f.broyden(pcrit[t])


# Print Critical Exercise Price 300 Periods to Expiration

print('Critical Exercise Price 300 Periods to Expiration')
print('   Critical Price  = {:5.2f}'.format(np.exp(pcrit[-1])))

# Plot Critical Exercise Prices
demo.figure('American Put Option Optimal Exercise Boundary', 'Periods Remaining Until Expiration', 'Exercise Price')
plt.plot(np.exp(pcrit))
plt.show()
