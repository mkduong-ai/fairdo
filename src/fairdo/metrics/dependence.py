"""
This module contains functions to calculate the dependency, correlation, association or any other relationship
between two variables.

In the fairness context, the dependency between the target variable and the protected attribute(s) is of interest.
Let :math:`y` be the target variable and :math:`z` be the protected attribute(s), then some kind of relationship between
these two variables is calculated using a function :math:`f`: :math:`f(y, z)`.
"""

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
    Calculates the dependency between ``y`` with each ``z[:,i]`` using the specified ``dependency_function``.
    Aggregates the dependency scores using the ``agg`` function.

    Let :math:`f` be the dependency function, :math:`y` be the target variable,
    and :math:`z` be the protected attributes, then the dependency score is calculated as (pythonic notation):

    .. math::
        \\text{dependency}(y, z) = \\text{agg}(f(y, z[:,0]), f(y, z[:,1]), \\ldots, f(y, z[:,-1]))


    Parameters
    ----------
    y : np.array, shape (n_samples,)
        Flattened binary array of shape (n_samples,), can be a prediction or the truth label.
    z : np.array, shape (n_samples, n_protected_attributes)
        Array of shape (n_samples, n_protected_attributes), represents the protected attributes.
    dependency_function : callable, optional
        Function to compute the dependency between `y` and each protected attribute.
        Default is normalized_mutual_info_score.
    agg : callable, optional
        Aggregation function to combine the dependency scores. Default is np.max.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The aggregated dependency score.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import dependency_multi
    >>> y = np.random.randint(0, 2, (10,))
    >>> z = np.random.randint(0, 2, (10, 3))
    >>> dependency_multi(y, z)
    0.12634639359704877
    """
    # check input
    if len(z.shape) == 1:
        z = z.reshape(-1, 1)

    if len(z.shape) > 2:
        raise ValueError('z must be a 2D or 1D array')

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


def nmi_multi(y: np.array, z: np.array,
              agg=np.max,
              positive_label=1,
              **kwargs):
    """
    Compute the normalized mutual information for multiple non-binary protected attributes.

    This function calculates the normalized mutual information between ``y``` and each
    protected attribute in ``z``, and then aggregates these scores using the specified ``agg`` function.
    Let :math:`y` be the target variable and :math:`z` be the protected attributes, then the normalized mutual
    information score for multi. protected attributes is calculated as (pythonic notation):

    .. math::
        \\text{NMI}(y, z) = \\text{agg}(\\text{NMI}(y, z[:,0]), \\text{NMI}(y, z[:,1]), \\ldots, \\text{NMI}(y, z[:,-1]))

    Parameters
    ----------
    y : np.array, shape (n_samples,)
        Flattened binary array, can be a prediction or the truth label.
    z : np.array, shape (n_samples, n_protected_attributes)
        Each ``z[:,i]`` represents a single protected attribute.
    agg : callable, optional
        Aggregation function to combine the normalized mutual information scores. Default is np.max.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The aggregated normalized mutual information score.

    References
    ----------
    [1] Strehl, A., Ghosh, J., & Mooney, R. J. (2002). Impact of similarity measures on web-page clustering.
        In Workshop on The WebKDD (pp. 58-64).

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import nmi_multi
    >>> y = np.random.randint(0, 2, (10,))
    >>> z = np.random.randint(0, 2, (10, 3))
    >>> nmi_multi(y, z)
    0.09855890449799566
    """
    return dependency_multi(y, z,
                            dependency_function=normalized_mutual_info_score,
                            agg=agg, positive_label=positive_label)


def nmi(y: np.array, z: np.array, **kwargs) -> float:
    """
    Calculate the normalized mutual information between two arrays.
    The protected attribute `z` can be binary or non-binary.

    Normalized mutual information is a normalization of the Mutual Information (MI) score
    to scale the results between 0 (no mutual information, independent variables) and 1
    (perfect correlation). The function handles any warning by ignoring them.
    The formula is given by:

    .. math::
        \\text{NMI}(y, z) = \\frac{2 \\cdot I(y, z)}{H(y) + H(z)}

    where :math:`I(y, z)` is the mutual information between `y` and `z`, and :math:`H(y)` and :math:`H(z)`
    are the entropies of `y` and `z`, respectively.

    Parameters
    ----------
    y : np.array, shape (n_samples,)
        Flattened array, can be a prediction or the truth label.
    z : np.array, shape (n_samples,)
        Flattened array of the same shape as y.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The normalized mutual information between y and z.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import nmi
    >>> y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> z = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> nmi(y, z)
    1.0
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalized_mutual_info_score(y, z)


def mi(y: np.array, z: np.array, bins=2, **kwargs) -> float:
    """
    Calculate the mutual information between two arrays.
    The protected attribute ``z`` can be binary or non-binary.

    Mutual information is a measure of the mutual dependence between two variables.
    It quantifies the "amount of information" (in units such as bits) obtained
    about one random variable, by observing the other random variable.
    Higher values indicate a higher dependency between the two variables. It is defined as:

    .. math::
        I(y, z) = \\sum_{y, z} p(y, z) \\log \\left(\\frac{p(y, z)}{p(y) \\cdot p(z)}\\right)

    where :math:`p(y, z)` is the joint probability distribution of `y` and `z`, and :math:`p(y)` and :math:`p(z)`
    are the marginal probability distributions of `y` and `z`, respectively.

    Parameters
    ----------
    y : np.array (n_samples,)
        Flattened array, can be a prediction or the truth label.
    z : np.array (n_samples,)
        Flattened array of the same shape as y.
    bins : int, optional
        Number of bins for discretization. Default is 2.
    **kwargs
        Additional keyword arguments. These are not currently used.

    Returns
    -------
    float
        The mutual information between y and z.

    References
    ----------
    [1] Cover, T. M., & Thomas, J. A. (2006). Elements of information theory. John Wiley & Sons.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import mi
    >>> y = np.random.randint(0, 2, (10,))
    >>> z = np.random.randint(0, 2, (10,))
    >>> mi(y, z)
    0.013844293808390806
    """
    mi_value = mutual_info_score(y, z)
    return mi_value


def entropy_estimate_cat(x: np.array, **kwargs) -> float:
    """Calculate the entropy of a categorical variable.
    It is caclulated as:

    .. math::
        H(x) = - \\sum_{i=1}^{n} p(x_i) \\log_2 p(x_i)

    where :math:`p(x_i)` is the probability of the i-th category. The entropy is a measure of the information/uncertainty
    of a random variable. Higher values indicate more information/uncertainty.
    
    Parameters
    ----------
    x : np.array (n_samples,)
        Array of shape (n_samples,) containing the labels.
    
    Returns
    -------
    float
        The entropy of the label distribution.

    References
    ----------
    [1] Shannon, C. E. (1948). A mathematical theory of communication. Bell system technical journal, 27(3), 379-423.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import entropy_estimate_cat
    >>> x = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> entropy_estimate_cat(x)
    1.0
    """
    prob_dist = np.bincount(x) / len(x)
    prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
    return -np.sum(prob_dist * np.log2(prob_dist))


def joint_entropy_cat(x: np.array):
    """Calculate the joint entropy of multiple categorical variables.
    The joint entropy is a measure of the information/surprise/uncertainty of a set of random variables.

    Parameters
    ----------
    x : np.array (n_samples, n_variables)
        Array of shape (n_samples, n_variables) containing the labels.
    
    Returns
    -------
    float
        The joint entropy of the categorical variables.

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import joint_entropy_cat
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
    Calculate the conditional entropy of a categorical variable given another categorical variable.

    Parameters
    ----------
    x : np.array (n_samples,)
        Array of shape (n_samples,) containing the labels.
    
    y : np.array (n_samples,) or (n_samples, n_variables)
        Array containing the labels.
    
    Returns
    -------
    float
        The conditional entropy of the label distribution.
    """
    xy = np.column_stack((x, y))

    # Calculate the entropy of the joint distribution H(X, Y)
    H_XY = joint_entropy_cat(xy)

    # Calculate the entropy of Y
    H_Y = joint_entropy_cat(y)

    # Calculate the conditional entropy H(X|Y) = H(X, Y) - H(Y)
    return H_XY - H_Y


def total_correlation(*arrays) -> float:
    """Calculate the total correlation (multi-information) of multiple categorical variables.
    
    Parameters
    ----------
    *arrays: np.array
        Arrays of shape (n_samples,) containing the labels.
    
    Returns
    -------
    float
        The total correlation of the categorical variables.
        
    References
    ----------
    [1] Watanabe, S. (1960). Information theoretical analysis of multivariate correlation. IBM Journal of Research and Development, 4(1), 66-82.
    [2] Garner, W. R. (1962). Uncertainty and Structure as Psychological Concepts, JohnWiley & Sons, New York
    """
    # Calculate sum of individual entropies
    try:
        sum_individual_entropies = sum([entropy_estimate_cat(arr) for arr in arrays])
    except:
        warnings.warn("Calculate total correlation along axis=1.")

        sum_individual_entropies = 0
        for i in range(len(arrays)):
            if len(arrays[i].shape) == 1:
                sum_individual_entropies += entropy_estimate_cat(arrays[i])
            elif len(arrays[i].shape) == 2:
                sum_individual_entropies += sum(
                    entropy_estimate_cat(arrays[i][:, j]) for j in range(arrays[i].shape[1]))
            else:
                raise Exception(f"{i}-th Parameter is not 1d or 2d.")

    # Calculate joint entropy
    joint_entropy = joint_entropy_cat(np.column_stack(arrays))

    # Total correlation is the sum of individual entropies minus the joint entropy
    tc = sum_individual_entropies - joint_entropy
    return tc


def dual_total_correlation(*arrays):
    """Calculate the dual total correlation using mutual information for more than two variables.
    
    Parameters
    ----------
    *arrays: np.array
        Arrays of shape (n_samples,) containing the labels.
    
    Returns
    -------
    float
        The dual total correlation of the categorical variables.

    References
    ----------
    [1] Han, Te Sun. (1978). Nonnegative entropy measures of multivariate symmetric correlations.
        Information and Control. 36 (2): 133–156.
    """
    n = len(arrays)
    if n < 2:
        raise ValueError("Need at least two variables to calculate dual total correlation.")

    xs = np.column_stack(arrays)

    # Calculate joint entropy
    joint_entropy = joint_entropy_cat(xs)

    # Calculate conditional entropy
    conditional_entropy = 0
    for i in range(xs.shape[1]):
        H_i = conditional_entropy_cat(xs[:, i], xs[:, np.arange(n) != i])
        conditional_entropy += H_i

    dtc = joint_entropy - conditional_entropy
    return dtc


def o_information(*arrays):
    """
    Calculate the O-information of multiple categorical variables.
    The O-information is the difference between the total correlation and the dual total correlation.
    
    Parameters
    ----------
    *arrays : np.array
        Arrays of shape (n_samples,) containing the labels.
    
    Returns
    -------
    float
        The O-information of the categorical variables."""
    return total_correlation(*arrays) - dual_total_correlation(*arrays)


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
    y : np.array
        Flattened array, can be a prediction or the truth label.
    z : np.array
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
    y : np.array
        Flattened array, can be a prediction or the truth label.
    z : np.array
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
    The Randomized Dependence Coefficient by
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf [1]_

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Parameters
    ----------
    y : np.array (n_samples,) or (n_samples, n_variables)
    z : np.array (n_samples,) or (n_samples, n_variables)
    f : callable
        function to use for random projection
    k : int
        number of random projections to use
    s : numeric
        scale parameter
    n : int
        number of times to compute the RDC and
        return the median (for stability)

    Returns
    -------
    float
        RDC between ``y`` and ``z``.

    Notes
    -----
    Implementation by Gary Doran and taken from: https://github.com/garydoranjr/rdc

    References
    ----------
    .. [1] David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf. (2013). The Randomized Dependence Coefficient.
           In Advances in Neural Information Processing Systems 26 (NIPS 2013).
           http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from fairdo.metrics.dependence import rdc
    >>> y = np.random.rand(100)
    >>> z = np.random.rand(100)
    >>> rdc(y, z)
    0.287647809294975
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
