from demos.setup import np, tic, toc
from compecon import LCP

# demslv07():
"""Linear complementarity problem methods    """

# Generate problem test data
n = 8
z = np.random.randn(n, 2) - 1

# Boundaries
a = np.min(z, 1)
b = np.max(z, 1)

# Objective function
q = np.random.randn(n)
M = np.random.randn(n, n)
M = - np.dot(M.T, M)


# Define the problem by creating an LCP instance
L = LCP(M, q, a, b)

# Set 100 random initial points
nrep = 100
x0 = np.random.randn(nrep, n)

# Solve by applying Newton method to semi-smooth formulation
t0 = tic()
it1 = 0
L.opts.transform = 'ssmooth'
for k in range(nrep):
    L.newton(x0[k])
    it1 += L.it
t1 = toc(t0)
n1 = L.fnorm

# Solve by applying Newton method to minmax formulation
t0 = tic()
it2 = 0
L.opts.transform = 'minmax'
for k in range(nrep):
    L.newton(x0[k])
    it2 += L.it
t2 = toc(t0)
n2 = L.fnorm


print('Hundredths of seconds required to solve randomly generated linear \n',
      'complementarity problem on R^8 using Newton and Lemke methods')
print('\nAlgorithm           Time      Norm   Iterations  Iters/second\n' + '-' * 60)
print('Newton semismooth {:6.2f}  {:8.0e}   {:8d}  {:8.1f}'.format(t1, n1, it1, it1/t1))
print('Newton minmax     {:6.2f}  {:8.0e}   {:8d}  {:8.1f}'.format(t2, n2, it2, it2/t2))


