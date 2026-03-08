__author__ = 'Randall'

## DEMDP11 Monetary Policy Model
#
# A central bank must set nominal interest rate so as to minimize
# deviations of inflation rate and GDP gap from established targets.
#
# States
#     s1      GDP gap
#     s2      inflation rate
# Actions
#     x       nominal interest rate
# Parameters
#     alpha   transition function constant coefficients
#     beta    transition function state coefficients
#     gamma   transition function action coefficients
#     omega   central banker's preference weights
#     sbar    equilibrium targets
#     delta   discount factor

# Preliminary tasks
from compecon import BasisChebyshev, DPmodel
from demos.setup import np, plt, demo
from compecon.quad import qnwnorm


## FORMULATION

# Model Parameters
alpha   = np.array([[0.9, -0.1]]).T             # transition function constant coefficients
beta    = np.array([[-0.5, 0.2], [0.3, -0.4]])  # transition function state coefficients
gamma   = np.array([[-0.1, 0.0]]).T             # transition function action coefficients
omega   = np.identity(2)                        # central banker's preference weights
sbar    = np.array([[1, 0]]).T                  # equilibrium targets
sigma   = 0.08 * np.identity(2),                # shock covariance matrix
delta   = 0.9                                   # discount factor


# Continuous State Shock Distribution
m   = [3, 3]                            # number of shocks
mu  = [0, 0]                            # means of shocks
[e,w] = qnwnorm(m,mu,sigma)            # shocks and probabilities

# Approximation Structure
n = 21                                 # number of collocation coordinates, per dimension
smin = [-2, -3]                         # minimum states
smax = [ 2,  3]                         # maximum states

basis = BasisChebyshev(n, smin, smax, method='complete',
                       labels=['GDP gap', 'inflation'])    # basis functions

print(basis)

def bounds(s, i, j):
    lb  = np.zeros_like(s[0])
    ub  = np.full(lb.shape, np.inf)
    return lb, ub


def reward(s, x, i, j):
    s = s - sbar  #  todo make sure they broadcast (:,ones(1,n))'
    f = np.zeros_like(s[0])
    for ii in range(2):
        for jj in range(2):
            f -= 0.5 * omega[ii, jj] * s[ii] * s[jj]
    fx = np.zeros_like(x)
    fxx = np.zeros_like(x)
    return f, fx, fxx


def transition(s, x, i, j, in_, e):
    g = alpha + beta @ s + gamma @ x + e
    gx = np.tile(gamma, (1, x.size))
    gxx = np.zeros_like(s)
    return g, gx, gxx


# Model Structure


bank = DPmodel(basis, reward, transition, bounds,
               x=['interest'], discount=delta, e=e, w=w)

# Solve Bellman Equation

#sol = bank.solve()
# resid, s, v, x = bank.residuals(nr=5)  # fixme takes huge amount of memory to deal with vmax over the refined grid!

bank.check_derivatives()


"""
''' this is a temp fix '''




# Compute Unconstrained Deterministic Steady-State
sstar, xstar, pstar = bank.lqapprox(sbar, [0])

# If Nonnegativity Constraint Violated, Re-Compute Deterministic Steady-State
if xstar < 0:
    I = np.identity(2)
    xstar = 0
    sstar = np.linalg.solve(np.identity(2) - beta, alpha)


print(sstar, xstar, pstar)
#
# # Reorient Deterministic Steady State
# sstar   = sstar'
# xstar   = xstar'

# Check Model Derivatives
# dpcheck(bank,sstar,xstar)


## SOLUTION

"""


"""
# Reshape Output for Plotting
n = n*5
s1 = reshape(s(:,1),n)
s2 = reshape(s(:,2),n)
v = reshape(v,n)
x = reshape(x,n)
resid = reshape(resid,n)

# Compute Shadow Prices
p1 = funeval(c,basis,s,[1 0])
p2 = funeval(c,basis,s,[0 1])
p1 = reshape(p1,n)
p2 = reshape(p2,n)

# Plot Optimal Policy
figure
surf(s1,s2,x,'FaceColor','interp','EdgeColor','interp')
title('Optimal Monetary Policy')
xlabel('GDP Gap')
ylabel('Inflation Rate')
zlabel('Nominal Interest Rate')
  
# Plot Value Function
figure
surf(s1,s2,v,'FaceColor','interp','EdgeColor','interp')
title('Value Function')
xlabel('GDP Gap')
ylabel('Inflation Rate')
zlabel('Value')
   
# Plot Shadow Price Function 1
figure
surf(s1,s2,p1,'FaceColor','interp','EdgeColor','interp')
title('Shadow Price of GDP Gap')
xlabel('GDP Gap')
ylabel('Inflation Rate')
zlabel('Price')
# Plot Shadow Price Function 2
figure
surf(s1,s2,p2,'FaceColor','interp','EdgeColor','interp')
title('Shadow Price of Inflation Rate')
xlabel('GDP Gap')
ylabel('Inflation Rate')
zlabel('Price')

# Plot Residual
figure
surf(s1,s2,resid,'FaceColor','interp','EdgeColor','interp')
title('Bellman Equation Residual')
xlabel('GDP Gap')
ylabel('Inflation Rate')
zlabel('Residual')


"""

## SIMULATION

# Simulate Model
#rand('seed',0.945)
nper, nrep = 21, 10000
sinit = np.tile(smax, (nrep, 1)).T
S = bank.simulate(nper, sinit)
print(S.mean())
"""

s1sim = ssim(:,:,1)
s2sim = ssim(:,:,2)

# Compute Ergodic Moments
s1avg = mean(s1sim(:))
s2avg = mean(s2sim(:))
xavg = mean(xsim(:))
s1std = std(s1sim(:))
s2std = std(s2sim(:))
xstd = std(xsim(:))

# Print Steady-State and Ergodic Moments
fprintf('Deterministic Steady-State\n') 
fprintf('   GDP Gap               = #5.2f\n'    ,sstar(1))
fprintf('   Inflation Rate        = #5.2f\n'    ,sstar(2))
fprintf('   Nominal Interest Rate = #5.2f\n\n'  ,xstar)
fprintf('Ergodic Means\n') 
fprintf('   GDP Gap               = #5.2f\n'    ,s1avg)
fprintf('   Inflation Rate        = #5.2f\n'    ,s2avg)
fprintf('   Nominal Interest Rate = #5.2f\n\n'  ,xavg)
fprintf('Ergodic Standard Deviations\n') 
fprintf('   GDP Gap               = #5.2f\n'    ,s1std)
fprintf('   Inflation Rate        = #5.2f\n'    ,s2std)
fprintf('   Nominal Interest Rate = #5.2f\n\n'  ,xstd)

# Plot Simulated and Expected State Paths 1
figure
hold on
plot(0:nper-1,s1sim(1:3,:))
plot(0:nper-1,mean(s1sim),'k')
plot(nper-1,s1avg,'k*')
title('Simulated and Expected State Paths')
xlabel('Period')
ylabel('GDP Gap')

# Plot Simulated and Expected State Paths 2
figure
hold on
plot(0:nper-1,s2sim(1:3,:))
plot(0:nper-1,mean(s2sim),'k')
plot(nper-1,s2avg,'k*')
title('Simulated and Expected State Paths')
xlabel('Period')
ylabel('Inflation Rate')

# Plot Simulated and Expected Policy Paths
figure
hold on
plot(0:nper-1,xsim(1:3,:))
plot(0:nper-1,mean(xsim),'k')
plot(nper-1,xavg,'k*')
title('Simulated and Expected Policy Paths')
xlabel('Period')
ylabel('Nominal Interest Rate')


"""