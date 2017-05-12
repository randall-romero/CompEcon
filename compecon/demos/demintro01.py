__author__ = 'Randall'

from demos.setup import np, plt




""" Inverse Demand Problem """

demand = lambda p: 0.5 * p ** -0.2 + 0.5 * p ** -0.5
derivative = lambda p: -0.01 * p ** -1.2 - 0.25 * p ** -1.5

p = 0.25
for it in range(100):
    f = demand(p) - 2
    d = derivative(p)
    s = -f / d
    p += s
    print('iteration {:3d} price {:8.4f}'.format(it, p))
    if np.linalg.norm(s) < 1.0e-8:
        break

# Generate demand function
pstar = p
qstar = demand(pstar)
n, a, b = 100, 0.02, 0.40
p = np.linspace(a, b, n)
q = demand(p)

# Graph demand function
fig = plt.figure()
ax1 = fig.add_subplot(121, title='Demand', aspect=0.1,
                      xlabel='p', xticks=[0.0, 0.2, 0.4], xlim=[0, 0.4],
                      ylabel='q', yticks=[0, 2, 4], ylim=[0, 4])
ax1.plot(p, q, 'b')



# Graph inverse demand function
ax2 = fig.add_subplot(122, title='Inverse Demand', aspect=10,
                      xlabel='q', xticks=[0, 2, 4], xlim=[0, 4],
                      ylabel='p', yticks=[0.0, pstar, 0.2, 0.4], yticklabels=['0.0', '$p^{*}$', '0.2', '0.4'], ylim=[0, 0.4])
ax2.plot(q, p, 'b')
ax2.plot([0, 2, 2], [pstar, pstar, 0], 'r--')
ax2.plot([2], [pstar], 'ro', markersize=12)
plt.show()

