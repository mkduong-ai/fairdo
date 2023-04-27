import numpy as np
import warnings

from ._helper import generate_pairs


def statistical_parity_absolute_difference_multi(y: np.array, z: np.array,
                                                 agg_attribute=np.sum,
                                                 agg_group=np.sum,
                                                 positive_label=1,
                                                 **kwargs) -> float:
    """
    Difference in statistical parity for multiple non-binary protected attributes

    Parameters
    ----------
    y: flattened binary array of shape (n_samples,)
        can be the prediction or the truth label
    z: (n_samples, n_protected_attributes)
        protected attribute
    agg_attribute: callable
        aggregation function for the attribute
    agg_group: callable
        aggregation function for the group
    positive_label: int

    Returns
    -------

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
    Difference in statistical parity (Lee et al. 2022)

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened binary array of shape y
        protected attribute
    positive_label: int
    privileged_group: int

    Returns
    -------

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


def mean_difference(*args, **kwargs):
    return statistical_parity_difference(*args, **kwargs)


def statistical_parity_absolute_difference(*args, **kwargs):
    return np.abs(statistical_parity_difference(*args, **kwargs))


def disparate_impact_ratio(y: np.array, z: np.array,
                           positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Disparate Impact: Higher score -> greater discrimination towards unprivileged group

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened binary array of shape y
        protected attribute
    positive_label: int
    privileged_group: int

    Returns
    -------

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
    Disparate Impact: Higher score -> greater discrimination towards unprivileged group

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened binary array of shape y
        protected attribute
    positive_label: int
    privileged_group: int

    Returns
    -------

    """
    return np.abs(1 - disparate_impact_ratio(y, z, positive_label, privileged_group, **kwargs))


def disparate_impact_ratio_objective_difference(y: np.array, z: np.array,
                                                positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Disparate Impact: Higher score -> greater discrimination towards unprivileged group

    Parameters
    ----------
    y: flattened binary array
        can be the prediction or the truth label
    z: flattened binary array of shape y
        protected attribute
    positive_label: int
    privileged_group: int

    Returns
    -------

    """
    return 1 - disparate_impact_ratio(y, z, positive_label, privileged_group, **kwargs)