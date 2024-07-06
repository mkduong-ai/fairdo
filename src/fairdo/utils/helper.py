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
    """
    return list(combinations(lst, 2))
