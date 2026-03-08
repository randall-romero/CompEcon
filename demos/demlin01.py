from demos.setup import np, tic, toc
from numpy.linalg import solve

""" Solving linear equations by different methods """

# Print table header
print('Hundreds of seconds required to solve n by n linear equation Ax=b')
print('m times using solve(A, b) and dot(inv(A), b), computing inverse only once.\n')
print('{:>5s} {:>5s} {:>12s} {:>12s}'.format('m', 'n', 'solve(A,b)', 'dot(inv(A), b)'))
print('-' * 40)

for m in [1, 100]:
    for n in [50, 500]:
        A = np.random.rand(n, n)
        b = np.random.rand(n, 1)

        tt = tic()
        for j in range(m):
            x = solve(A, b)

        f1 = 100 * toc(tt)

        tt = tic()
        Ainv = np.linalg.inv(A)
        for j in range(m):
            x = np.dot(Ainv, b)

        f2 = 100 * toc(tt)
        print('{:5d} {:5d} {:12.2f} {:12.2f}'.format(m, n, f1, f2))
