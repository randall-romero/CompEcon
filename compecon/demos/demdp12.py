__author__ = 'Randall'


## DEMDP12 Production Management Model
#
# Profit maximizing entrepeneur must decide how much to produce, subject to 
# production adjustment costs.
#
# States
#     i       market price (discrete)
#     s       lagged production (continuous)
# Actions
#     x       current production
# Parameters
#     alpha   marginal adjustment cost
#     beta    marginal production cost parameters
#     pbar    long-run average market price
#     sigma   market price shock standard deviation
#     delta   discount factor

# Preliminary tasks
from demos.setup import demo, plt, np
from compecon import BasisSpline, DPmodel
from compecon.quad import qnwlogn

## FORMULATION

# Model Parameters
alpha = 0.01                               # marginal adjustment cost
beta  = [0.8, 0.03]                         # marginal production cost parameters
pbar  = 1.0                                # long-run average market price
sigma = 0.2                                # market price shock standard deviation
delta = 0.9                                # discount factor


# Deterministic Steady-State
sstar = (pbar - beta[0]) / beta[1]             # deterministic steady-state state



# Continuous State Shock Distribution
m = 3                                      # number of market price shocks
mu = np.log(pbar) - sigma ** 2 / 2                   # mean log price
p, w = qnwlogn(m, mu, sigma ** 2)              # market price shocks and probabilities
q = np.tile(w, (m, 1))                      # market price transition probabilities

# Approximation Structure
n = 50                                     # number of collocation nodes
smin =  0                                  # minimum state
smax = 20                                  # maximum state
basis = BasisSpline(n, smin, smax, labels=['lagged production'])        # basis functions


def bounds(s, i, j):
    return np.zeros_like(s), np.full(s.shape, np.inf)


def reward(s, q, i, j):
    f = p[i] * q - (beta[0] * q + 0.5 * beta[1] * q ** 2) - 0.5 * alpha * ((q - s) ** 2)
    fx = p[i] - beta[0] - beta[1] * q - alpha * (q - s)
    fxx = (-beta[1] - alpha) * np.ones_like(s)
    return f, fx, fxx


def transition(s, q ,i, j, in_, e):
    return q.copy(), np.ones_like(q), np.zeros_like(q)



# Model Structure

model = DPmodel(basis, reward, transition, bounds,
                i=['Low price', 'Average price', 'High Price'],
                x=['Current production'],
                discount=delta, q=q)


# Check Model Derivatives
# dpcheck(model,sstar,sstar)


## SOLUTION

# Solve Bellman Equation
model.solve()
resid, s, v, x = model.solution()

"""





# Plot Optimal Policy
figure
plot(s,x)
legend('Low Price','Average Price','High Price')
legend('Location','Best')
legend('boxoff')
title('Optimal Production Policy')
xlabel('Lagged Production')
ylabel('Production')

# Plot Value Function
figure
plot(s,v)
legend('Low Price','Average Price','High Price')
legend('Location','Best')
legend('boxoff')
title('Value Function')
xlabel('Lagged Production')
ylabel('Value')

# Plot Shadow Price Function
figure
lambda = funeval(c,basis,s,1)
plot(s,lambda)
legend('Low Price','Average Price','High Price')
legend('Location','Best')
legend('boxoff')
title('Shadow Price of Lagged Production')
xlabel('Lagged Production')
ylabel('Shadow Price')

# Plot Residual
figure
hold on
plot(s,resid)
legend('Low Price','Average Price','High Price')
legend('Location','Best')
legend('boxoff')
plot(s,0*resid,'k--','LineWidth',2)
title('Bellman Equation Residual')
xlabel('Lagged Production')
ylabel('Residual')


## SIMULATION

# Simulate Model
rand('seed',0.945)
nper = 26
nrep = 10000
sinit = smin(ones(nrep,1),:)
iinit = 2*ones(nrep,1)
[ssim,xsim,isim] = dpsimul(model,basis,nper,sinit,iinit,s,v,x)
psim = p(isim)

# Compute Ergodic Moments
savg = mean(ssim(:))
pavg = mean(psim(:))
sstd = std(ssim(:))
pstd = std(psim(:))

# Print Ergodic Mean and Standard Deviation
fprintf('Deterministic Steady-State\n')
fprintf('   Price       = #5.2f\n'  ,pbar)
fprintf('   Production  = #5.2f\n\n',sstar)
fprintf('Ergodic Means\n')
fprintf('   Price       = #5.2f\n',  pavg)
fprintf('   Production  = #5.2f\n\n',savg)
fprintf('Ergodic Standard Deviations\n')
fprintf('   Price       = #5.2f\n'  ,pstd)
fprintf('   Production  = #5.2f\n\n',sstd)

# Plot Simulated and Expected Policy Path
figure
hold on
plot(0:nper-1,xsim[1:3,:])
plot(0:nper-1,mean(xsim),'k')
title('Simulated and Expected Production')
xlabel('Period')
ylabel('Production')


##
"""


