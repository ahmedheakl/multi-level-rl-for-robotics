import numpy as np

z = np.zeros((1), dtype=np.uint32)
x = np.array([[611., 72., 128., 542.]])
y = np.zeros((3, 4), dtype=np.uint32)
data: np.uint32 = z[0]
print(data.item())