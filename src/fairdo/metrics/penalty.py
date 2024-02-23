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
from fairdo.utils.helper import nunique


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


def group_missing_penalty(z: np.array, n_groups: np.array,
                          agg_attribute='max',
                          agg_group='max', **kwargs) -> float:
    """
    Calculate the penalty for missing groups in a protected attribute.
    The number of groups `n_groups` is used to calculate the penalty.

    If `agg_group` is 'max', the penalty is 1 if any group is missing, otherwise 0.
    If `agg_group` is 'sum', the penalty is the sum of the penalties for each group.
    `agg_attribute` is used to aggregate the penalties for each protected attribute.

    Parameters
    ----------
    z: np.array
        Array of shape (n_samples, n_protected_attributes) representing multiple protected attributes.
        or
        (n_samples,) represents one protected attribute.
        
        Each protected attribute can consists of >2 groups.
    n_groups: np.array
        Number of groups for each protected attribute.
    agg_group: str, optional
        Aggregation function for the group. Default is 'sum'.

    Returns
    -------
    float
        The penalty for missing groups.
    """
    n_avail_groups = nunique(z, axis=0)

    if agg_group == 'max':
        if agg_attribute == 'max':
            return int(np.any(n_avail_groups < n_groups))
        elif agg_attribute == 'sum':
            return np.sum(n_avail_groups < n_groups)
    elif agg_group == 'sum':
        n_missing_groups = n_groups - n_avail_groups
        group_penalties = n_missing_groups * (2 * n_groups - n_missing_groups - 1) / 2
        
        if agg_attribute == 'sum':
            return np.sum(group_penalties)
        elif agg_attribute == 'max':
            return np.max(group_penalties)
    else:
        raise NotImplementedError("Only sum and max are implemented for agg_group and agg_attribute.")


def data_integrity_penalty():
    """
    Placeholder for data integrity penalty.
    """
    raise NotImplementedError("Data Integrity Penalty is not implemented yet.")


def data_size_measure(y: np.array, dims: int, **kwargs) -> float:
    """
    Test function to measure the size of the data.
    The function returns the negative of the length of the vector `y` divided by `dims`.
    This is used to test whether genetic algorithms are able to select all the data points.

    Parameters
    ----------
    y: np.array
        Vector to measure the size of.
    dims: int
    """
    return - len(y) / dims
