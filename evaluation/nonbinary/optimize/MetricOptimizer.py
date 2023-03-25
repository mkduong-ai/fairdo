import numpy as np


def metric_optimizer_constraint(f, d, n_constraint, m_cands):
    """
    Parameters
    ----------
    f: function
        function to optimize
    d: int
        number of dimensions
    n_constraint: int
        constraint of number of 1s
    m_cands: int
        number of candidates to select
    Returns
    -------

    """
    pass


def metric_optimizer(f, d, m_cands):
    """
    Parameters
    ----------
    f: function
        function to optimize
    d: int
        number of dimensions
    m_cands: int
        number of candidates to select
    Returns
    -------

    """
    return metric_optimizer_constraint(f=f, d=d, n_constraint=0, m_cands=m_cands)
