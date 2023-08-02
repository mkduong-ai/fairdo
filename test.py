import time
import numpy as np
import fado
from fado.metrics.nonbinary import *

size = 10000
y = np.random.binomial(1, 0.5, size=size)
z = np.random.choice([0, 1, 2, 4], size=size, replace=True)


print(fado.metrics.statistical_parity_abs_diff_mean(y, z))
print(nb_statistical_parity_sum_abs_difference_normalized(y, z))

print(fado.metrics.statistical_parity_abs_diff_multi(y, z, agg_group=np.mean))
