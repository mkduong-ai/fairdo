import numpy as np


def penalty(x, n):
    """
    Penalty function that penalizes the fitness of a solution if it does not satisfy the constraint.
    The number of 1s in the binary vector should be equal to n. If it is not, the penalty is the absolute
    difference between the number of 1s and n.

    Parameters
    ----------
    x: numpy array
        vector
    n: int
        constraint

    Returns
    -------
    penalty: float
    """
    return np.abs(np.sum(x) - n)


def penalty_normalized(x, n):
    """
    Percentage of the sum of the entries of the vector x that is greater than n

    Parameters
    ----------
    x: numpy array
        vector
    n: int
        constraint

    Returns
    -------
    penalty: float
    """
    return np.abs(np.sum(x) - n) / n