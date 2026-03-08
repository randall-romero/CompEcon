from compecon import Basis, SmolyakGrid
import numpy as np

n = [9, 9, 5]
qna = [3, 2, 1]

nodes, polys = SmolyakGrid(n, qna)

print(nodes.shape)