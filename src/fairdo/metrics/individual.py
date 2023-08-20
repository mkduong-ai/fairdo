import numpy as np
from sklearn.neighbors import NearestNeighbors


def consistency_score(x: np.array, y: np.array, n_neighbors=5, **kwargs) -> float:
    """
    Compute the Consistency Score as defined in Learning Fair Representations (Zemel et al. 2013).

    This score measures the consistency of the output `y` with respect to nearest neighbors in `x`.
    A higher score indicates more fairness.

    Parameters
    ----------
    x: np.array
        Array representing the input data.
    y: np.array
        Array of the same length as x, representing the output data.
    n_neighbors: int, optional
        Number of neighbors to consider. Default is 5.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The consistency score. Higher values indicate more fairness.
    """

    # fit the KNN model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(x)

    # get the distances and indices of the k nearest neighbors for each instance
    distances, indices = nbrs.kneighbors(x)

    # drop the first column (self-loops)
    indices = indices[:, 1:]

    # compute the absolute differences between the output values of each instance and its neighbors
    differences = np.abs(y[:, None] - y[indices])

    return 1 - np.mean(differences)


def consistency_score_objective(x: np.array, y: np.array, n_neighbors=5, **kwargs) -> float:
    """
    Compute the inverse of the Consistency Score to use as an objective function.

    This function is intended to be minimized. Lower values indicate more individual fairness.

    Parameters
    ----------
    x: np.array
        Array representing the input data.
    y: np.array
        Array of the same length as x, representing the output data.
    n_neighbors: int, optional
        Number of neighbors to consider. Default is 5.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The inverse of the consistency score. Lower values indicate more fairness.
    """
    return 1 - consistency_score(x, y, n_neighbors, **kwargs)
