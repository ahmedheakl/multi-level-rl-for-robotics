import numpy as np
x = np.array([[1,2], [1.5, 2]])
print(np.max(x, axis=0), np.min(x, axis=0))