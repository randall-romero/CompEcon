__author__ = 'Randall'

from compecon.quad import qnwnorm
import numpy as np
from compecon.tools import example



''' Example in page 3 '''
example(3)
p = 0.25
for i in range(100):
    deltap = (.5 * p **-.2 + .5 * p ** -.5 - 2)/(.1 * p **-1.2 + .25 * p **-1.5)
    p += deltap
    if abs(deltap) < 1.e-8:
        break

print('Price = ', p)


''' Example in page 4 '''
example(4)
y, w = qnwnorm(10, 1, 0.1)
a = 1
for it in range(100):
    aold = a
    p = 3 - 2 * a * y
    f = w.dot(np.maximum(p, 1))
    a = 0.5 + 0.5 * f
    if abs(a - aold) < 1.e-8:
        break

print('Acreage = ', a)
print('Expected market price = ', np.dot(w, p))
print('Expected effective producer price = ', f)