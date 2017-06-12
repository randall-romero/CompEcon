from compecon.demos import np, plt, demo
from numpy.linalg import norm, cond, solve

""" Ill-conditioning of Vandermonde matrix"""
#todo: Review this demo, result not the same as in Miranda's
#fixme: There seems to be a problem with the computation of eigenvalues by numpy.linalg

# Compute approximation error and matrix condition number
n = np.arange(6, 12)
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
fig = plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(n, conv)
plt.xlabel('n')
plt.ylabel('$\log_{10}$ Condition Number')
plt.title('Vandermonde Matrix Condition Numbers')

# Plot approximation errors
plt.subplot(1, 2, 2)
plt.plot(n,errv)
plt.xlabel('n')
plt.ylabel('$\log_{10}$ Error')
plt.title(r'Approximation Error for $I - V^{-1}V$')

plt.show()
demo.savefig([fig])