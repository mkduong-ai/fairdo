import numpy as np
from aif360.sklearn.metrics import consistency_score


def consistency_score_objective(X, y, n_neighbors=5):
    """
    Lower score implies more individual fairness

    Parameters
    ----------
    X
    y
    n_neighbors

    Returns
    -------

    """
    return 1 - consistency_score(X, y, n_neighbors)