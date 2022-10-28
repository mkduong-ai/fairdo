import numpy as np
from aif360.sklearn.metrics import consistency_score


def consistency_score_objective(x, y, n_neighbors=5, **kwargs):
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
    return 1 - consistency_score(x, y, n_neighbors)
