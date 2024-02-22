"""
Penalty Functions for Constrained Optimization
==============================================

This module provides penalty functions specifically designed for fairness optimization with constraints.
The constraint in this context is that the number of data points after pre-processing should match a specified value.
A practical penalty function is ``relative_shortfall_penalty``, which is designed to handle situations
where the number of data points is less than this specified value, and in such cases,
penalties are applied to the solutions.
"""
import numpy as np


def absolute_difference_penalty(x: np.array, n: int):
    """
    Penalty function that penalizes the fitness of a solution if it does not satisfy the constraint.
    The number of 1s in the binary vector should be equal to ``n``. If it is not,
    the absolute_difference_penalty is the absolute
    difference between the number of 1s and ``n``.

    Parameters
    ----------
    x: numpy array
        binary vector
    n: int
        constraint

    Returns
    -------
    float
        The absolute difference between the number of 1s and ``n``.
    """
    if n != 0:
        return np.abs(np.sum(x) - n)
    else:
        return 0


def relative_difference_penalty(x: np.array, n: int):
    """
    Percentage of the sum of the entries of the vector ``x`` that is greater than ``n``.

    Parameters
    ----------
    x: numpy array
        binary vector
    n: int
        constraint

    Returns
    -------
    float
        The percentage of the sum of the entries of the vector ``x`` that is greater than ``n``.
    """
    if n != 0:
        return np.abs(np.sum(x) - n) / n
    else:
        return 0


def relative_shortfall_penalty(x: np.array, n: int):
    """
    Calculates the relative shortfall penalty.

    If the sum of `x` is greater than or equal to `n`, or `n` is zero, the function returns 0.
    Otherwise, it returns the difference between 1 and the ratio of the sum of `x` to `n`,
    which represents the proportion by which the sum of `x` falls short of `n`.

    Parameters
    ----------
    x: numpy array
        binary vector
    n: int
        A constraint that the sum of `x` should ideally meet or exceed.

    Returns
    -------
    float
        The relative shortfall penalty.
    """
    if n != 0:
        if np.sum(x) >= n:
            return 0
        else:
            return 1 - (np.sum(x) / n)
    else:
        return 0


def group_missing_penalty(z: np.array, n_groups: int,
                          agg_group='max', **kwargs) -> float:
    """
    Calculate the penalty for missing groups in a protected attribute.
    The number of groups `n_groups` is used to calculate the penalty.

    Parameters
    ----------
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    n_groups: int
        Number of groups for the protected attribute.
    agg_group: str, optional
        Aggregation function for the group. Default is 'sum'.

    Returns
    -------
    float
        The penalty for missing groups.
    """
    if agg_group == 'sum':
        # Return the penalty for each comparison for each missing group
        n_avail_groups = len(np.unique(z))
        n_missing_groups = n_groups - n_avail_groups
        return n_missing_groups * (2 * n_groups - n_missing_groups - 1) / 2
    elif agg_group == 'max':
        # Return 1 if there is at least one group missing
        return int(len(np.unique(z)) < n_groups)
    else:
        raise NotImplementedError("Only sum and max are implemented for agg_group")


def data_integrity_penalty():
    """
    Placeholder for data integrity penalty.
    """
    raise NotImplementedError("Data Integrity Penalty is not implemented yet.")