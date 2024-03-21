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


def group_missing_penalty(z: np.array, n_groups: np.array,
                          agg_attribute='max',
                          agg_group='max',
                          eps=0.1,
                          **kwargs) -> float:
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
    n_groups: np.array or int
        Number of groups for each protected attribute.
    agg_group: str, optional
        Aggregation function for the group. Default is 'sum'.
    agg_attribute: str, optional
        Aggregation function for the attribute. Default is 'max'.
    eps: float, optional
        Small value to add to the penalty. Default is 0.1.
        Acts as an upper bound for the maximum discrimination possible
        that is not a supremum. This is to ensure that missing a group
        is always worse than having a group with a large discrimination.

    Returns
    -------
    float
        The penalty for missing groups.
    """
    n_avail_groups = nunique(z, axis=0)

    if agg_group == 'max':
        if agg_attribute == 'max':
            return int(np.any(n_avail_groups < n_groups)) * (1 + eps)
        elif agg_attribute == 'sum':
            return np.sum((n_avail_groups < n_groups) * (1 + eps))
    elif agg_group == 'sum':
        n_missing_groups = n_groups - n_avail_groups
        group_penalties = n_missing_groups * (2 * n_groups - n_missing_groups - 1) / 2
        
        if agg_attribute == 'max':
            return np.max(group_penalties)
        elif agg_attribute == 'sum':
            return np.sum(group_penalties)
    else:
        raise NotImplementedError("Only sum and max are implemented for agg_group and agg_attribute.")


def data_loss(y: np.array, dims: int, **kwargs) -> float:
    """
    Calculate the relative amount of data lost after pre-processing.

    Parameters
    ----------
    y: np.array
        Labels of the data.
        The size of it depicts the current size of the data.
    dims: int
        The size of the original data.
    """
    return 1 - len(y) / dims
