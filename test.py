import numpy as np
from fairdo.metrics.dependence import nmi_multi, mi, nmi, entropy_estimate_cat, joint_entropy_cat, conditional_entropy_cat, total_correlation, dual_total_correlation, pearsonr_abs, pearsonr
x = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y = 1- x
print(pearsonr(x, y))
