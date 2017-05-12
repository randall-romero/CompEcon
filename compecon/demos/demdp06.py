"""
 DEMDP06 Deterministic Optimal Economic Growth Model

 Welfare maximizing social planner must decide how much society should
 consume and invest.  Model is of special interest because it has a known 
 closed-form solution.

 States
     s       stock of wealth
 Actions
     k       capital investment
 Parameters
     beta    capital production elasticity
     delta   discount factor
"""


from demos.setup import demo, np, plt
from compecon import BasisChebyshev, DPmodel, DPoptions, qnwnorm


# =========== Approximation Structure
n, smin, smax = 15, 0.2, 1.0
basis = BasisChebyshev(n, smin, smax, labels=['Wealth'])
snodes = basis.nodes

# =========== Model specification

beta, delta = 0.5, 0.9


def bounds(s, i, j):
    return np.zeros_like(s), s[:]


def reward(s, k, i, j):
    sk = s - k
    f = np.log(sk)
    fx= - sk ** -1
    fxx = - sk ** -2
    return f, fx, fxx


def transition(s, k, i, j, in_, e):
    g = k ** beta
    gx = beta * k **(beta - 1)
    gxx = (beta - 1) * beta * k ** (beta - 2)
    return g, gx, gxx


sigma = 0.1
m=5
e,w = qnwnorm(m, -sigma**2 / 2, sigma**2)



growth_model = DPmodel(basis, reward, transition, bounds,
                       x=['Investment'],
                       discount=delta)


# ======== Steady-State
sstar = (beta * delta) ** (beta / (1 - beta))   # steady-state wealth
kstar = beta * delta * sstar                    # steady-state capital investment
vstar = np.log(sstar - kstar) / (1 - delta)     # steady-state value
pstar = 1 / (sstar * (1 - beta * delta))        # steady-state shadow price
b = 1 / (1 - delta * beta)


# ===========   Compute Analytic Solution at Collocation Nodes
vtrue = vstar + b * (np.log(snodes) - np.log(sstar))
ktrue = delta * beta * snodes


# Set a refined grid to evaluate the functions
#Wealth = np.linspace(smin, smax, n * 10)
order = np.atleast_2d([0, 1])

# ===========  Solve Bellman Equation
options = dict(print=True,
               algorithm='newton',
               maxit=253)

S = growth_model.solve(vtrue, ktrue, print=True, algorithm='newton', maxit=120)
v, pr = growth_model.Value(S.Wealth, order)
k = growth_model.Policy(S.Wealth)
# resid = growth_model.residuals(s)

# ============  Simulate Model
T = 20
data = growth_model.simulate(T, np.atleast_2d(smin))


# ============  Compute Linear-Quadratic Approximation
growth_model.lqapprox(sstar, kstar)
vlq, plq = growth_model.Value(S.Wealth, order)
klq = growth_model.Policy(S.Wealth)

# ============   Compute Analytic Solution
vtrue = vstar + b * (np.log(S.Wealth) - np.log(sstar))


# ==============   Make plots:
Wealth = S.Wealth.T


# Plot Optimal Policy
demo.figure('Optimal Investment Policy', 'Wealth', 'Investment')
plt.plot(Wealth, np.c_[k, klq])
demo.annotate(sstar, kstar,'$s^*$ = %.2f\n$k^*$ = %.2f' % (sstar, kstar), 'bo', (10, -7))
plt.legend(['Chebychev Collocation','L-Q Approximation'], loc = 'upper left')


# Plot Value Function
demo.figure('Value Function', 'Wealth', 'Value')
plt.plot(Wealth, np.c_[v, vlq])
demo.annotate(sstar, vstar,'$s^*$ = %.2f\n$V^*$ = %.2f' % (sstar, vstar),'bo', (10, -7))
plt.legend(['Chebychev Collocation','L-Q Approximation'], loc= 'upper left')



# Plot Shadow Price Function
demo.figure('Shadow Price Function', 'Wealth', 'Shadow Price')
plt.plot(Wealth, np.c_[pr, plq])
demo.annotate(sstar, pstar,'$s^*$ = %.2f\n$\lambda^*$ = %.2f' % (sstar, pstar), 'bo', (10, 7))
plt.legend(['Chebychev Collocation','L-Q Approximation'])


# Plot Chebychev Collocation Residual and Approximation Error
plt.figure(figsize=[12, 6])
demo.subplot(1, 2, 1, 'Chebychev Collocation Residual\nand Approximation Error', 'Wealth', 'Residual/Error')
plt.plot(Wealth, np.c_[S.resid, v - vtrue], Wealth, np.zeros_like(Wealth), 'k--')
plt.legend(['Residual','Error'], loc='lower right')

# Plot Linear-Quadratic Approximation Error
demo.subplot(1, 2, 2, 'Linear-Quadratic Approximation Error', 'Wealth', 'Error')
plt.plot(Wealth, vlq - vtrue)


# Plot State and Policy Paths
opts = dict(spec='r*', offset=(-5, -5), fs=11, ha='right')

data[['Wealth', 'Investment']].plot()
plt.title('State and Policy Paths')
demo.annotate(T, sstar, 'steady-state wealth\n = %.2f' % sstar, **opts)
demo.annotate(T, kstar, 'steady-state investment\n = %.2f' % kstar, **opts)
plt.xlabel('Period')
plt.ylabel('Wealth/Investment')
plt.xlim([0, T + 0.5])


# Print Steady-State
print('\n\nSteady-State')
print('   Wealth       = %5.4f' % sstar)
print('   Investment   = %5.4f' % kstar)

plt.show()


