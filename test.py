import numpy as np
from fairdo.metrics.dependence import nmi_multi, mi, nmi, entropy_estimate_cat, joint_entropy_cat
y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
z = np.array([0, 1, 1, 0, 1, 0, 0, 1])
x = np.array([[0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1]])
print(nmi(y, z))
print(joint_entropy_cat(x))
