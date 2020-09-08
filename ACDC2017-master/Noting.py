import numpy as np
pred   = [[0.9,0.2],
           [0.1,0.8],
           [0,0],
           [0,0]]
numpy = np.array(pred)
# data = np.zeros(tuple([1] + [1] + list(new_shp[1:])), dtype=np.float32)
# print(np.array(pred).mean(0))
print(numpy[None])