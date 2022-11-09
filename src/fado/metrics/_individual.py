import numpy as np
from sklearn.neighbors import NearestNeighbors


def consistency_score(x: np.array, y: np.array, n_neighbors=5, **kwargs) -> float:
    """
    Consistency Score in Learning Fair Representations (Zemel et al. 2013)

    Parameters
    ----------
    x: array-like
    y: array-like
    n_neighbors: int
    kwargs

    Returns
    -------
    numeric
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(x)
    indices = nbrs.kneighbors(x, return_distance=False)

    return 1 - np.mean(np.abs(y - np.mean(y[indices], axis=1)))


def consistency_score_objective(x: np.array, y: np.array, n_neighbors=5, **kwargs) -> float:
    """
    Lower score implies more individual fairness

    Parameters
    ----------
    x: array-like
    y: array-like
    n_neighbors: int

    Returns
    -------
    numeric
    """
    return 1 - consistency_score(x, y, n_neighbors)
