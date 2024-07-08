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
    y: np.array (n_samples,)
        Flattened array, can be a prediction or the truth label.
    z: np.array (n_samples,)
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
    y: np.array (n_samples,)
        Flattened array, can be a prediction or the truth label.
    z: np.array (n_samples,)
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


def entropy_estimate_cat(x: np.array, **kwargs) -> float:
    """Calculate the entropy of a categorical variable.
    
    Parameters
    ----------
    x: np.array (n_samples,)
        Array of shape (n_samples,) containing the labels.
    
    Returns
    -------
    float
        The entropy of the label distribution."""
    prob_dist = np.bincount(x) / len(x)
    prob_dist = prob_dist[prob_dist > 0]  # Remove zeros
    return -np.sum(prob_dist * np.log2(prob_dist))


def joint_entropy_cat(x: np.array):
    """Calculate the joint entropy of multiple categorical variables.

    Parameters
    ----------
    x: np.array (n_samples, n_variables)
        Array of shape (n_samples, n_variables) containing the labels.
    
    Returns
    -------
    float
        The joint entropy of the categorical variables.
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
                sum_individual_entropies += sum(entropy_estimate_cat(arrays[i][:, j]) for j in range(arrays[i].shape[1]))
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
    [1] Han, Te Sun (1978). Nonnegative entropy measures of multivariate symmetric correlations. Information and Control. 36 (2): 133–156.
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
    *arrays: np.array
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
