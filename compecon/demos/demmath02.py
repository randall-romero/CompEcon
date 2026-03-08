from demos.setup import np, plt
from scipy.integrate import quad


""" Computing Function Inner Products, Norms & Metrics """

# Compute Inner Product and Angle
a, b = -1, 1
f = lambda x: 2 * x**2 - 1
g = lambda x: 4 * x**3 - 3*x
fg = quad(lambda x: f(x) * g(x), a, b)[0]
ff = quad(lambda x: f(x) * f(x), a, b)[0]
gg = quad(lambda x: g(x) * g(x), a, b)[0]
angle = np.arccos(fg / np.sqrt(ff * gg)) * 180 / np.pi
print('\nCompute inner product and angle')
print('\tfg = {:6.4f},   ff = {:6.4f}, gg = {:6.4f},  angle = {:3.2f}'.format(fg, ff, gg, angle))

# Compute Function Norm
a, b = 0, 2
f = lambda x: x**2 - 1
p1, p2 = 1, 2
q1 = quad(lambda x: np.abs(f(x) - g(x)) ** p1, a, b)[0] ** (1 / p1)
q2 = quad(lambda x: np.abs(f(x) - g(x)) ** p2, a, b)[0] ** (1 / p2)
print('\nCompute function norm')
print('\tnorm 1 = {:6.4f},   norm 2 = {:6.4f}'.format(q1, q2))


# Compute Function Metrics
a, b = 0, 2
f = lambda x: x**3 + x**2 + 1
g = lambda x: x**3 + 2
p1, p2 = 1, 2
q1 = quad(lambda x: np.abs(f(x)-g(x)) ** p1, a, b)[0] ** (1 / p1)
q2 = quad(lambda x: np.abs(f(x)-g(x)) ** p2, a, b)[0] ** (1 / p2)
print('\nCompute function metrics')
print('\tnorm 1 = {:6.4f},   norm 2 = {:6.4f}'.format(q1, q2))

# Illustrate function metrics
x = np.linspace(a, b, 200)
plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
plt.plot([0, 2], [0, 0], 'k:', linewidth=4)
plt.plot(x, f(x) - g(x), 'b', linewidth=4, label='f - g')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2])
plt.yticks([-1, 0, 1, 2, 3])
plt.title('f - g')

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(f(x) - g(x)), 'b', linewidth=4, label='f - g')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2, 3])
plt.title('|f - g|')

plt.show()

# Demonstrate Pythagorean Theorem
a, b = -1, 1
f = lambda x: 2 * x**2 - 1
g = lambda x: 4 * x**3 -3*x
ifsq = quad(lambda x: f(x) ** 2, a, b)[0]
igsq = quad(lambda x: g(x) ** 2, a, b)[0]
ifplusgsq = quad(lambda x: (f(x) + g(x)) ** 2, a, b)[0]
print('\nDemonstrate Pythagorean Theorem')
print(r'    $\int f^2(x) dx$ = {:6.4f}, $\int g^2(x) dx$ = {:6.4f}'.format(ifsq, igsq))
print(r'    $\int f^2(x) dx + \int g^2(x) dx$ = {:6.4f}, $\int (f+g)^2(x) dx$ = {:6.4f}'.format(ifsq + igsq, ifplusgsq))

