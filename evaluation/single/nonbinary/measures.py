import numpy as np


def sanity_check(**kwargs):
    return 123456789


def count_size(y: np.array, z: np.array, positive_label=1, **kwargs):
    """
    Count the size of this dataset

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened array of shape y
        protected attribute. It holds integer values.
    positive_label: int
    kwargs: dict
        not used

    Returns
    -------

    """
    return len(y)


def count_groups(y: np.array, z: np.array, positive_label=1, **kwargs):
    """
    Count the size of this dataset

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened array of shape y
        protected attribute. It holds integer values.
    positive_label: int
    kwargs: dict
        not used

    Returns
    -------

    """
    values = np.unique(z)
    return len(values)
