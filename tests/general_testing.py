from compecon.interpolator import *
from compecon.basis import Basis
from compecon.ce_util import gridmake
import numpy as np
np.set_printoptions(precision=3, suppress=True)

B = Basis(7, -3, 3)
Value = InterpolatorArray(B, [3, 4])
xx = B.nodes

f = lambda x: np.sin(x)

F = np.array([f(xx)*a + 10*b for a in range(3) for b in range(4)]).reshape([3,4,7])

for idx in Value.idx:
    Value[idx] = F[idx]

#print(Value.y)


G = np.array([lambda x, a=i, b=j: a*np.sin(b*x) for i in range(3) for j in range(4)]).reshape([3,4])

for idx in Value.idx:
    Value[idx] = G[idx](xx)
#print(Value.y)

Vj = Value[0,1]
#print(Vj.y)

#c = np.ones([2, 4, 5])
#Phi = np.random.randint(0,6,[8, 5])
#print(np.dot(c,Phi.T))

xval = np.linspace(-3, 3, 11)
ord = np.array([0,1,2])
#print(Value[1,1](xval,[0, 1, 2]))
print(Value[1,1](xval,1))