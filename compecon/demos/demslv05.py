from compecon.demos import np, plt, demo
from compecon import NLP
from compecon.tools import gridmake


def cournot(z):
    x, y = z
    fval = [y*np.exp(x) - 2*y, x*y - y**3]
    fjac = [[y*np.exp(x), np.exp(x)-2],
            [y, x-3*y**2]]
    return np.array(fval), np.array(fjac)



''' Parameters and initial value '''
alpha = 0.6
beta = np.array([0.6, 0.8])

''' Set up the Cournot function '''



''' Generate data for contour plot '''
n = 100
q1 = np.linspace(0.3, 1.1, n)
q2 = np.linspace(0.4, 1.2, n)
z = np.array([cournot(q)[0] for q in gridmake(q1, q2).T]).T

''' Using a NLP object '''
q = np.array([1.0, 0.5])
cournot_problem = NLP(cournot)#, q)
q_star, fq_star = cournot_problem.newton(q)
print(q_star)


''' Plot figures '''
steps_options = {'marker': 'o',
                 'color': (0.2, 0.2, .81),
                 'linewidth': 1.0,
                 'markersize': 6,
                 'markerfacecolor': 'red',
                 'markeredgecolor': 'red'}

contour_options = {'levels': [0.0],
                   'colors': '0.25',
                   'linewidths': 0.5}


Q1, Q2 = np.meshgrid(q1, q2)
Z0 = np.reshape(z[0], (n,n), order='F')
Z1 = np.reshape(z[1], (n,n), order='F')

methods = ['newton', 'broyden']
cournot_problem.opts['all_x'] =  True


fig = plt.figure()
for it in range(2):
    cournot_problem.zero(x0=q, method=methods[it])
    demo.subplot(1, 2, it + 1, methods[it].capitalize() + "'s method",
                 '$x$', '$y$', [min(q1), max(q1)], [min(q2), max(q2)])
    plt.contour(Q1, Q2, Z0, **contour_options)
    plt.contour(Q1, Q2, Z1, **contour_options)
    plt.plot(*cournot_problem.x_sequence, **steps_options)
    ax = plt.gca()
    ax.set_xticks([0.3, 0.7, 1.1])
    ax.set_yticks([0.4, 0.8, 1.2])

    demo.text(0.7, 0.45, '$f_1(x,y) = 0$', 'left', 'top',fs=12)
    demo.text(0.3, 0.55, '$f_2(x,y) = 0$', 'left', 'center',fs=12)
plt.show()

demo.savefig([fig])