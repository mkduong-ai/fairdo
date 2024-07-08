"""
Mixed math functions used throughout the package.

References
----------
.. [1] Shannon, C. E. (1948). A mathematical theory of communication. Bell system technical journal, 27(3), 379-423.
"""

import numpy as np


def entropy_estimate_cat(x: np.array, **kwargs) -> float:
    """Calculate the entropy [1]_ of a categorical variable.
    It is caclulated as:

    .. math::
        H(X) = - \\sum_{i=1}^{n} p(X_i) \\log_2 p(X_i)

    where :math:`p(X_i)` is the probability of the i-th category. The entropy is a measure of the information/uncertainty
    of a random variable. Higher values indicate more information/uncertainty.

    Parameters
    ----------
    x : np.array (n_samples,)
        Array of shape (n_samples,) containing the categorical labels as numerical values.

    Returns
    -------
    float
        The entropy of the label distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.utils.math import entropy_estimate_cat
    >>> x = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> entropy_estimate_cat(x)
    1.0
    """
    prob_dist = np.bincount(x) / len(x)
    prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
    return -np.sum(prob_dist * np.log2(prob_dist))


def joint_entropy_cat(x: np.array):
    """Calculate the joint entropy [1]_ of multiple categorical variables.
    The joint entropy is a measure of the information/surprise/uncertainty of a set of random variables.
    Let :math:`X = (X_1, X_2, \\ldots, X_m)` be a set of categorical variables, i.e.,
    multivariate random variable, then the joint entropy is calculated as:

    .. math::
        H(X) = -\\sum_{X_1 \\in\\mathcal X_1} \\ldots \\sum_{X_m \\in\\mathcal X_m} P(X_1, ..., X_m)
        \\log_2[P(X_1, ..., X_m)]

    Parameters
    ----------
    x : np.array (n_samples, n_variables)
        Array of shape (n_samples, n_variables) containing the labels as numerical values.

    Returns
    -------
    float
        The joint entropy of the categorical variables in the array ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.utils.math import joint_entropy_cat
    >>> x = np.array([[0, 1, 1, 0, 1, 0, 0, 1],
    ...               [0, 1, 1, 0, 1, 0, 0, 1]])
    >>> joint_entropy_cat(x)
    -0.0
    """
    if x.ndim > 1:
        x_idx = np.ravel_multi_index(x.T, np.max(x, axis=0) + 1)
    else:
        x_idx = x
    prob_dist = np.bincount(x_idx) / len(x_idx)
    prob_dist = prob_dist[prob_dist > 0]
    return -np.sum(prob_dist * np.log2(prob_dist))


def conditional_entropy_cat(x: np.array, y: np.array) -> float:
    """
    Calculate the conditional entropy [1]_ of a categorical variable ``x`` given another categorical variable ``y``,
    i.e.,

    .. math::
        H(X|Y) = H(X, Y) - H(Y)

    where :math:`H(X, Y)` is the joint entropy of the categorical variables ``x`` and ``y``
    and :math:`H(Y)` is the entropy of the variable ``y``.

    Parameters
    ----------
    x : np.array (n_samples,)
        Array of shape (n_samples,) containing the labels.

    y : np.array (n_samples,) or (n_samples, n_variables)
        Array containing the labels. Can represent a single or multiple categorical variables.

    Returns
    -------
    float
        The conditional entropy of the label distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.utils.math import conditional_entropy_cat
    >>> x = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> conditional_entropy_cat(x, y)
    0
    """
    xy = np.column_stack((x, y))

    # Calculate the entropy of the joint distribution H(X, Y)
    H_XY = joint_entropy_cat(xy)

    # Calculate the entropy of Y
    H_Y = joint_entropy_cat(y)

    # Calculate the conditional entropy H(X|Y) = H(X, Y) - H(Y)
    return H_XY - H_Y
