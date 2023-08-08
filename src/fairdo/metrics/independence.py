import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

import warnings


def dependency_multi(y: np.array, z: np.array,
                     dependency_function=normalized_mutual_info_score,
                     agg=np.max,
                     positive_label=1,
                     **kwargs) -> float:
    """
    Compute a measure of dependency for multiple non-binary protected attributes.

    This function calculates the dependency between `y` and each protected attribute in `z`
    using the specified `dependency_function`, and then aggregates these dependency scores
    using the specified `agg` function.

    Parameters
    ----------
    y: np.array
        Flattened binary array of shape (n_samples,), can be a prediction or the truth label.
    z: np.array
        Array of shape (n_samples, n_protected_attributes), represents the protected attributes.
    dependency_function: callable, optional
        Function to compute the dependency between `y` and each protected attribute.
        Default is normalized_mutual_info_score.
    agg: callable, optional
        Aggregation function to combine the dependency scores. Default is np.max.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The aggregated dependency score.
    """
    # check input
    if len(z.shape) != 2:
        raise ValueError('z must be a 2D array')
    # invert privileged and positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    # get normalized mutual information for each attribute
    scores = []
    for i in range(z.shape[1]):
        scores.append(dependency_function(y, z[:, i]))

    return agg(scores)


def normalized_mutual_information_multi(y: np.array, z: np.array,
                                        agg=np.max,
                                        positive_label=1,
                                        **kwargs):
    """
    Compute the normalized mutual information for multiple non-binary protected attributes.

    This function calculates the normalized mutual information between `y` and each
    protected attribute in `z`, and then aggregates these scores using the specified `agg` function.

    Parameters
    ----------
    y: np.array
        Flattened binary array of shape (n_samples,), can be a prediction or the truth label.
    z: np.array
        Array of shape (n_samples, n_protected_attributes), represents the protected attributes.
    agg: callable, optional
        Aggregation function to combine the normalized mutual information scores. Default is np.max.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The aggregated normalized mutual information score.
    """
    dependency_multi(y, z,
                     dependency_function=normalized_mutual_info_score,
                     agg=agg, positive_label=positive_label)


def mutual_information(y: np.array, z: np.array, bins=2, **kwargs) -> float:
    """
    Calculate the mutual information between two arrays.
    The protected attribute `z` can be binary or non-binary.

    Mutual information is a measure of the mutual dependence between two variables.
    It quantifies the "amount of information" (in units such as bits) obtained
    about one random variable, by observing the other random variable.

    Parameters
    ----------
    y: np.array
        Flattened array, can be a prediction or the truth label.
    z: np.array
        Flattened array of the same shape as y.
    bins: int, optional
        Number of bins for discretization. Default is 2.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The mutual information between y and z.
    """
    mi = mutual_info_score(y, z)
    return mi


def normalized_mutual_information(y: np.array, z: np.array, **kwargs) -> float:
    """
    Calculate the normalized mutual information between two arrays.
    The protected attribute `z` can be binary or non-binary.

    Normalized mutual information is a normalization of the Mutual Information (MI) score
    to scale the results between 0 (no mutual information, independent variables) and 1
    (perfect correlation). The function handles any warning by ignoring them.

    Parameters
    ----------
    y: np.array
        Flattened array, can be a prediction or the truth label.
    z: np.array
        Flattened array of the same shape as y.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The normalized mutual information between y and z.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalized_mutual_info_score(y, z)


def pearsonr(y: np.array, z: np.array, **kwargs) -> float:
    """
    Calculate the Pearson correlation coefficient between two arrays.
    The protected attribute `z` can be binary or non-binary.

    The Pearson correlation coefficient measures the linear relationship between two variables.
    The calculation of the Pearson correlation coefficient is not affected by scaling,
    and it ranges from -1 to 1. A value of 1 implies a perfect positive correlation,
    while a value of -1 implies a perfect negative correlation.

    Parameters
    ----------
    y: np.array
        Flattened array, can be a prediction or the truth label.
    z: np.array
        Flattened array of the same shape as y.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The Pearson correlation coefficient between y and z.
    """
    return np.corrcoef(y.reshape(1, -1), z.reshape(1, -1))[0, 1]


def pearsonr_abs(y: np.array, z: np.array, **kwargs) -> float:
    """
    Calculate the absolute value of the Pearson correlation coefficient between two arrays.
    The protected attribute `z` can be binary or non-binary.

    The Pearson correlation coefficient measures the linear relationship between two datasets.
    The calculation of the Pearson correlation coefficient is not affected by scaling,
    and it ranges from -1 to 1. A value of 1 implies a perfect positive correlation,
    while a value of -1 implies a perfect negative correlation. The absolute value is taken
    to disregard the direction of the correlation. Any correlation is considered as dependency
    and therefore unfairness.

    Parameters
    ----------
    y: np.array
        Flattened array, can be a prediction or the truth label.
    z: np.array
        Flattened array of the same shape as y.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The absolute value of the Pearson correlation coefficient between y and z.
    """
    return np.abs(pearsonr(y, z))


def rdc(y: np.array, z: np.array, f=np.sin, k=20, s=1 / 6., n=1, **kwargs):
    """
    Implements the Randomized Dependence Coefficient
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf

    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    Computes the Randomized Dependence Coefficient

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Parameters
    ----------
    y, z: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)

    Returns
    -------
    float
        The RDC between X and Y
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(y, z, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(y.shape) == 1: y = y.reshape((-1, 1))
    if len(z.shape) == 1: z = z.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in y.T]) / float(y.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in z.T]) / float(z.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))
