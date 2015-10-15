import numpy as np
from numpy.linalg import norm, cond, solve
import matplotlib.pyplot as plt
np.set_printoptions(3)


""" Ill-conditioning of Vandermonde matrix"""
#todo: Review this demo, result not the same as in Miranda's
# Compute approximation error and matrix condition number
n = np.arange(6, 51)
nn = n.size

errv = np.zeros(nn)
conv = np.zeros(nn)

for i in range(nn):
    v = np.vander(1 + np.arange(n[i]))
    errv[i] = np.log10(norm(np.identity(n[i]) - solve(v, v)))
    conv[i] = np.log10(cond(v))

print('errv = ', errv)
# Smooth using quadratic function
X = np.vstack([np.ones(nn), n]).T
b = np.linalg.lstsq(X, errv)[0]
print('b = ', b)
errv = np.dot(X, b)
b = np.linalg.lstsq(X, conv)[0]
print('b = ', b)
conv = np.dot(X, b)

# Plot matrix condition numbers
plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(n, conv)
plt.xlabel('n')
plt.ylabel('Log_{10} Condition Number')
plt.title('Vandermonde Matrix Condition Numbers')

# Plot approximation errors
plt.subplot(1, 2, 2)
plt.plot(n,errv)
plt.xlabel('n')
plt.ylabel('Log_{10} Error')
plt.title(r'Approximation Error for I - V\V')

plt.show()
