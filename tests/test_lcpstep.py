from compecon.lcpstep import arrayinvb, _arrayinvb2, arrayinv
import numpy as np
np.set_printoptions(precision=4)

nx, ns = 2, 4
p = 0.5
a = 0.11

xlx = np.zeros([nx, ns]) + a
xux = xlx + 1
F = np.linspace(-p, p, nx*ns)
Fx = np.linspace(-p, p, nx*nx*ns)

F.shape = [nx, ns]
Fx.shape = [nx, nx, ns]
#Fx = Fx + Fx.swapaxes(0, 1)

args = (xlx, xux, F, Fx)

A1 = arrayinvb(*args)
A2 = _arrayinvb2(*args)

#print(A1)
#print(A2-A1)


''' Test 2 '''
F = np.linspace(-p, p, nx*ns).reshape([nx, ns])
Fx = np.linspace(-p, p, nx*nx*ns).reshape([nx,nx,ns])
#F = np.random.random([nx, ns])
#Fx = np.random.random([nx, nx, ns])
print(F)
print(Fx)
#
y = arrayinv(F, Fx)
print(y)