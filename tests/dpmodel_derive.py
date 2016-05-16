from compecon.dpmodel import DPmodel
from compecon.basis import Basis
import numpy as np
import matplotlib.pyplot as plt
import seaborn
np.set_printoptions(precision=3, suppress=True)


B = Basis(7, 0.0, 3.0)

class Modelo(DPmodel):
    def reward(self, s, x):
        return x[0] * x[1]
    def transition(self, s, x):
        return np.vstack([np.cos(x[0]), x[0] * np.sin(x[1])])

M = Modelo(B, ni=2, nj=3, dx=2)
M.ds = 2


for idx in M.Value_j.idx:
    M.Value_j[idx] = np.random.randint(0,25,(1,7))

for idx in M.Policy_j.idx:
    M.Policy_j[idx] = (10*idx[2] + 1) * np.random.randint(1,9,(1,7))





M.__update_value_function()
#print(growth_model.Value_j.y)
#print('\n', growth_model.Value.y)
#print('\n', growth_model.DiscreteAction)

M.__updatePolicy()
print(M.Policy.y)




s0 = B.nodes
x0 = np.vstack([s0, 10*s0])


f, fx, fxx = M.getDerivative('reward', s0, x0)
print('f = ', f.shape)
print('fx = ', fx.shape)
print('fxx = ', fxx.shape)

plt.figure()
plt.plot(x0[0], f)
plt.plot(x0[0], fx[0])
plt.plot(x0[0], fxx[1,0])
plt.show()

g, gx, gxx = M.getDerivative('transition', s0, x0)
print('g = ', g.shape)
print('gx = ', gx.shape)
print('gxx = ', gxx.shape)
#print(gxx)

print('is the Hessian symmetric?')
print(gxx[0, 1] - gxx[1, 0])