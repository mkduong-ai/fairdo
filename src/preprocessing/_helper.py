import numpy as np
import pandas as pd


def discrepancy(x, method='CD'):
    """Measures the discrepancy of a given sample.
    Assume [0, 1] normalized samples

    Parameters
    ----------
    x: array_like (n, d)
        The sample to compute the discrepancy from.

    method: str, optional
        Type of discrepancy. Default is 'CD' which is Centered L2 discrepancy (Hickernell 1998).

    Returns
    -------
    disc: float
        centered l2 discrepancy
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    n, d = x.shape

    if method=='CD':
        second_sum = 0
        first_sum = np.sum(np.prod(1 + abs(x - 0.5) / 2 - (abs(x - 0.5) ** 2) / 2, axis=1))
        for i in range(n):
            second_sum += np.sum(np.prod(1 + abs(x[i, :] - 0.5) / 2 + abs(x - 0.5) / 2 - abs(x[i, :] - x) / 2, axis=1))
        CL2 = (13/12) ** d - (2 * first_sum - second_sum / n) / n

        return CL2