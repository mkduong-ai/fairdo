import time
import numpy as np
import fado

size = 10000
y = np.random.binomial(1, 0.5, size=size)
z = np.random.choice([0, 1, 2, 4], size=size, replace=True)


print(y)
print(z)

