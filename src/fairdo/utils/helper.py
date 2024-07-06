"""
Helper functions for the fairdo package.
"""

from itertools import combinations
import numpy as np


def nunique(a, axis=0):
    """
    Count the number of unique elements in an array along a given axis.
    
    Parameters
    ----------
    a: np.array
        The array to count the number of unique elements.
    axis: int, optional
        The axis along which to count the number of unique elements.
        Default is 0.
    
    Returns
    -------
    np.array
        The number of unique elements along the given axis.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.utils.helper import nunique
    >>> nunique(np.array([1, 2, 3, 1, 2, 3]))
    array([3])

    >>> nunique(np.array([[1, 2, 3], [1, 2, 3]]), axis=1)
    array([3, 3])

    >>> nunique(np.array([[1, 2, 3], [1, 2, 3]]), axis=0)
    array([1, 1, 1])
    """
    if a.ndim == 1:
        a = a.reshape(-1,1)
    elif a.ndim > 2:
        raise ValueError('a must be 1D or 2D')
    
    if axis == 0:
        a_s = np.sort(a,axis=0)
        out = a.shape[0]-(a_s[:-1,:] == a_s[1:,:]).sum(axis=0)
    elif axis == 1:
        a_s = np.sort(a,axis=1)
        out = a.shape[1]-(a_s[:,:-1] == a_s[:,1:]).sum(axis=1)
    else:
        raise ValueError('axis must be 0 or 1')
    
    return out


def generate_pairs(lst):
    """
    Generate all possible pairs of elements in a list without repetitions

    Parameters
    ----------
    lst: array_like
        list of elements

    Returns
    -------
    list
        list of pairs of elements

    Examples
    --------
    >>> from fairdo.utils.helper import generate_pairs
    >>> generate_pairs([1, 2, 3])
    [(1, 2), (1, 3), (2, 3)]
    """
    return list(combinations(lst, 2))
