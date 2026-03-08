__author__ = 'Randall'

from demos.setup import np, plt
# from quantecon import MarkovChain





""" Simulate Simple Markov Chain """
''' FORMULATION '''

# Model Parameters
gamma = 0.07     # aggregate unemployment rate
eta = 2.0        # expected duration of unemployment
y = [0.51, 1.0]  # income per employment state
delta = 0.90     # discount factor

# Employment Transition Probabilities
q = np.zeros([2, 2])
q[0, 0] = 1 - 1 / eta
q[1, 0] = gamma * (1 - q[0, 0]) / (1 - gamma)
q[0, 1] = 1 - q[0, 0]
q[1, 1] = 1 - q[1, 0]

# Compute Expected Lifetime Income
e = np.linalg.solve(np.identity(2) - delta * q, y)

# Compute Stationary Distribution of Employment Expected Employment State Durations
# p = MarkovChain(q)  # not exactly what markov is in matlab


# TODO: Finish this demo!!  Markov chain not imported yet from quantecon