import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def nb_mutual_information(y: np.array, z: np.array, **kwargs) -> float:
    """
    Measures whether two variables are independent by calculating their mutual information

    Parameters
    ----------
    y: np.array
    z: np.array
    bins: int
    kwargs: keyworded arguments

    Returns
    -------
    float
    """
    # discrimination measurement
    mi = mutual_info_score(y, z)
    return mi


def nb_normalized_mutual_information(y: np.array, z: np.array, **kwargs) -> float:
    """
    Returns the normalized mutual information between two variables.

    Parameters
    ----------
    y: np.array
    z: np.array
    kwargs: keyworded arguments

    Returns
    -------
    float
    """
    # Normalizes mutual information to 0 (independence) and 1 (perfect correlation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalized_mutual_info_score(y, z)
