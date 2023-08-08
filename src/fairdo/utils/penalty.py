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


def absolute_difference_penalty(x, n):
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


def relative_difference_penalty(x, n):
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


def relative_shortfall_penalty(x, n):
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
