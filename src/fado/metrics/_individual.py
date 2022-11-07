import numpy as np
from sklearn.neighbors import NearestNeighbors


def consistency_score(x: np.array, y: np.array, n_neighbors=5, **kwargs) -> float:
    """

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

    return 1 - abs(y - y[indices].mean(axis=1)).mean()


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
