import numpy as np


def nb_statistical_parity_sum_abs_difference(y: np.array, z: np.array, positive_label=1, **kwargs):
    """
    Difference in statistical parity

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened array of shape y
        protected attribute. It holds integer values.

    positive_label: int

    Returns
    -------

    """
    if z.ndim > 1:
        raise ValueError("z must be a 1D array")

    # invert positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    sum_diff = 0
    groups = list(set(z))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sum_diff += np.abs(np.sum(y & (z == groups[i])) / np.sum(z == groups[i]) -
                               np.sum(y & (z == groups[j])) / np.sum(z == groups[j]))

    return sum_diff


def nb_statistical_parity_sum_abs_difference_normalized(y: np.array, z: np.array, positive_label=1, **kwargs):
    """
    Difference in statistical parity

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened array of shape y
        protected attribute. It holds integer values.

    positive_label: int

    Returns
    -------

    """
    if z.ndim > 1:
        raise ValueError("z must be a 1D array")

    # invert positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    sum_diff = 0
    groups = list(set(z))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sum_diff += np.abs(np.sum(y & (z == groups[i])) / np.sum(z == groups[i]) -
                               np.sum(y & (z == groups[j])) / np.sum(z == groups[j]))

    sum_diff_normalized = 2 * sum_diff / (len(groups)*(len(groups)-1))
    return sum_diff_normalized


def nb_statistical_parity_max_abs_difference(y: np.array, z: np.array, positive_label=1, **kwargs):
    """
    Difference in statistical parity

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened array of shape y
        protected attribute. It holds integer values.

    positive_label: int

    Returns
    -------

    """
    if z.ndim > 1:
        raise ValueError("z must be a 1D array")

    # invert positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    max_diff = 0
    groups = list(set(z))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            diff = np.abs(np.sum(y & (z == groups[i])) / np.sum(z == groups[i]) -
                          np.sum(y & (z == groups[j])) / np.sum(z == groups[j]))
            max_diff = diff if diff > max_diff else max_diff
    return max_diff
