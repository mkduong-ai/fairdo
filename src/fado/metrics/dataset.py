import numpy as np
import warnings

from fado.utils.helper import generate_pairs


def statistical_parity_absolute_difference_multi(y: np.array, z: np.array,
                                                 agg_attribute=np.sum,
                                                 agg_group=np.sum,
                                                 positive_label=1,
                                                 **kwargs) -> float:
    """
    Calculate the difference in statistical parity for multiple non-binary protected attributes.

    Parameters
    ----------
    y : np.array
        Flattened binary array of shape (n_samples,), can be the prediction or the truth label.
    z : np.array
        Array of shape (n_samples, n_protected_attributes) representing the protected attribute.
    agg_attribute : callable, optional
        Aggregation function for the attribute. Default is np.sum.
    agg_group : callable, optional
        Aggregation function for the group. Default is np.sum.
    positive_label : int, optional
        Label considered as positive. Default is 1.

    Returns
    -------
    float
        Aggregated attribute disparity.
    """
    # check input
    if len(z.shape) != 2:
        raise ValueError('z must be a 2D array')
    # invert privileged and positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    # get unique values for each attribute
    groups = [np.unique(z[:, i]) for i in range(z.shape[1])]
    # get statistical parity for each attribute
    attributes_disparity = []
    for k, zk in enumerate(groups):
        # generate all possible pairs of values for the attribute
        pairs = generate_pairs(zk)
        group_disparity = []
        for i, j in pairs:
            # get statistical parity for each pair
            parity_i = np.sum(y & z[:, k] == i) / np.sum(z[:, k] == i)
            parity_j = np.sum(y & z[:, k] == j) / np.sum(z[:, k] == j)
            group_disparity.append(np.abs(parity_i - parity_j))
        attributes_disparity.append(agg_group(group_disparity))

    return agg_attribute(attributes_disparity)


def statistical_parity_difference(y: np.array, z: np.array,
                                  positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the difference in statistical parity according to Lee et al. (2022).

    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in statistical parity between unprivileged and privileged groups.
    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)
    priv = np.sum(y & z) / np.sum(z)
    unpriv = np.sum(y & (1-z)) / np.sum(1 - z)

    return unpriv-priv


def mean_difference(*args, **kwargs) -> float:
    """
    Alias for the statistical_parity_difference function.

    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in statistical parity between unprivileged and privileged groups.
    """
    return statistical_parity_difference(*args, **kwargs)


def statistical_parity_absolute_difference(*args, **kwargs) -> float:
    """
    Calculate the absolute value of the statistical parity difference.

    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The absolute value of the statistical parity difference.
    """
    return np.abs(statistical_parity_difference(*args, **kwargs))


def disparate_impact_ratio(y: np.array, z: np.array,
                           positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the Disparate Impact ratio.

    This function computes the ratio of probabilities of positive outcomes for
    the unprivileged group to the privileged group. A value of 1 indicates
    fairness, while a value < 1 indicates discrimination towards the
    unprivileged group. A value of > 1 would indicate discrimination towards
    the privileged group.

    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The Disparate Impact ratio.
    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)
    priv = np.sum(y & z) / np.sum(z)
    unpriv = np.sum(y & (1 - z)) / np.sum(1 - z)

    if priv == 0:
        warnings.warn("Disparate impact cannot be calculated. y=1 and z=1 are not apparent in the dataset.")
        warnings.warn("Return 1 (fair).")

        return 1

    return unpriv/priv


def disparate_impact_ratio_objective(y: np.array, z: np.array,
                                     positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the objective Disparate Impact ratio.

    This function computes the absolute difference between 1 and the Disparate
    Impact ratio. It can be used as an objective function to minimize
    discrimination towards the unprivileged group (and the privileged group).
    Lower values indicate less discrimination.

    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The objective Disparate Impact ratio.
    """
    return np.abs(1 - disparate_impact_ratio(y, z, positive_label, privileged_group, **kwargs))


def disparate_impact_ratio_deviation(y: np.array, z: np.array,
                                     positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the difference in objective Disparate Impact ratio.

    This function computes the difference between 1 and the Disparate Impact
    ratio. A value of 0 indicates fairness. A positive value indicates
    discrimination towards the unprivileged group. A negative value indicates
    discrimination towards the privileged group.


    Parameters
    ----------
    y : np.array
        Flattened binary array, can be the prediction or the truth label.
    z : np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label : int, optional
        Label considered as positive. Default is 1.
    privileged_group : int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in objective Disparate Impact ratio.
    """
    return 1 - disparate_impact_ratio(y, z, positive_label, privileged_group, **kwargs)