import numpy as np
import warnings

from fairdo.utils.helper import generate_pairs


def group_missing_penalty(z: np.array, n_groups: int,
                          agg_group='max', **kwargs) -> float:
    """
    Calculate the penalty for missing groups in a protected attribute.
    The number of groups `n_groups` is used to calculate the penalty.

    Parameters
    ----------
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    n_groups: int
        Number of groups for the protected attribute.
    agg_group: str, optional
        Aggregation function for the group. Default is 'sum'.
    
    Returns
    -------
    float
        The penalty for missing groups.
    """
    if agg_group == 'sum':
        # Return the penalty for each comparison for each missing group
        n_avail_groups = len(np.unique(z))
        n_missing_groups = n_groups - n_avail_groups
        return n_missing_groups * (2 * n_groups - n_missing_groups - 1) / 2
    elif agg_group == 'max':
        # Return 1 if there is at least one group missing
        return int(len(np.unique(z)) < n_groups)
    else:
        raise NotImplementedError("Only sum and max are implemented for agg_group")


def statistical_parity_abs_diff_multi(y: np.array, z: np.array,
                                      agg_attribute=np.sum,
                                      agg_group=np.sum,
                                      positive_label=1,
                                      **kwargs) -> float:
    """
    Calculate the absolute difference in statistical parity for multiple non-binary protected attributes.
    Protected attributes `z[i]` can be binary or non-binary.

    Parameters
    ----------
    y: np.array
        Flattened binary array of shape (n_samples,), can be the prediction or the truth label.
    z: np.array
        Array of shape (n_samples, n_protected_attributes) representing the protected attribute.
    agg_attribute: callable, optional
        Aggregation function for the attribute. Default is np.sum.
    agg_group: callable, optional
        Aggregation function for the group. Default is np.sum.
    positive_label: int, optional
        Label considered as positive. Default is 1.

    Returns
    -------
    float
        Aggregated attribute disparity.
    """
    # check input
    if len(z.shape) < 2:
        z = z.reshape(-1, 1)
    # invert privileged and positive label if required
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)

    # get unique values for each attribute
    groups = [np.unique(z[:, i]) for i in range(z.shape[1])]
    # print(groups)
    # get statistical parity for each attribute
    attributes_disparity = []
    for k, zk in enumerate(groups):
        # calculate statistical parities for all groups in one pass
        parities = {i: np.sum(y & (z[:, k] == i)) / np.sum(z[:, k] == i) for i in zk}
        # generate all possible pairs of values for the attribute
        pairs = generate_pairs(zk)
        group_disparity = [np.abs(parities[i] - parities[j]) for i, j in pairs]
        # print(group_disparity)
        try:
            attributes_disparity.append(agg_group(group_disparity))
        except ValueError:
            warnings.warn(f"Could not aggregate disparity for attribute {k} with aggregation function {agg_group}. "
                          f"Returning disparity of 0.")
            attributes_disparity.append(0)
    return agg_attribute(attributes_disparity)


def statistical_parity_abs_diff(y: np.array, z: np.array, agg_group=np.sum, **kwargs) -> float:
    """
    Calculate the absolute value of the statistical parity difference between all groups inside a protected attribute.
    The protected attribute `z` can be binary or non-binary.
    Returned value is aggregated with `agg_group`.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    agg_group: callable, optional
        Aggregation function for the group. Default is np.sum.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The absolute value of the statistical parity difference.
    """
    if z.ndim > 1:
        raise ValueError("z must be a 1D array")
    return statistical_parity_abs_diff_multi(y=y, z=z, agg_group=agg_group, **kwargs)


def statistical_parity_abs_diff_sum(y: np.array, z: np.array,
                                    **kwargs) -> float:
    """
    Calculate the maximum of statistical parity absolute differences between all groups in a protected attribute.
    The protected attribute `z` can be binary or non-binary.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        Average of the absolute value of the statistical parity differences between all groups.
    """
    return statistical_parity_abs_diff(y=y, z=z, agg_group=np.sum, **kwargs)


def statistical_parity_abs_diff_mean(y: np.array, z: np.array,
                                     **kwargs) -> float:
    """
    Calculate the sum of statistical parity absolute differences between all groups and return the average score.
    The protected attribute `z` can be binary or non-binary.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        Average of the absolute value of the statistical parity differences between all groups.
    """
    return statistical_parity_abs_diff(y=y, z=z, agg_group=np.mean, **kwargs)


def statistical_parity_abs_diff_max(y: np.array, z: np.array,
                                    **kwargs) -> float:
    """
    Calculate the maximum of statistical parity absolute differences between all groups in a protected attribute.
    The protected attribute `z` can be binary or non-binary.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened array of shape y, represents the protected attribute.
        Can represent non-binary protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        Average of the absolute value of the statistical parity differences between all groups.
    """
    return statistical_parity_abs_diff(y=y, z=z, agg_group=np.max, **kwargs)


def statistical_parity_difference(y: np.array, z: np.array,
                                  positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the difference in statistical parity according to [1].
    The protected attribute `z` must be binary. Returned value can be negative.

    [1] A Maximal Correlation Framework for Fair Machine Learning (Lee et al. 2022) (https://arxiv.org/abs/2106.00051)

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in statistical parity between unprivileged and privileged groups.
    """
    if z.ndim > 1:
        raise ValueError("z must be a 1D array")

    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y = 1 - y

    y = y.astype(int)
    z = z.astype(int)
    priv = np.sum(y & z) / np.sum(z)
    unpriv = np.sum(y & (1 - z)) / np.sum(1 - z)

    return unpriv - priv


def mean_difference(*args, **kwargs) -> float:
    """
    Alias for the `statistical_parity_difference` function.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in statistical parity between unprivileged and privileged groups.
    """
    return statistical_parity_difference(*args, **kwargs)


def disparate_impact_ratio(y: np.array, z: np.array,
                           positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the Disparate Impact ratio.
    The protected attribute `z` must be binary.

    This function computes the ratio of probabilities of positive outcomes for
    the unprivileged group to the privileged group. A value of 1 indicates
    fairness, while a value < 1 indicates discrimination towards the
    unprivileged group. A value of > 1 would indicate discrimination towards
    the privileged group.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
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

    return unpriv / priv


def disparate_impact_ratio_objective(y: np.array, z: np.array,
                                     positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the objective Disparate Impact ratio.
    The protected attribute `z` must be binary.

    This function computes the absolute difference between 1 and the Disparate
    Impact ratio. It can be used as an objective function to minimize
    discrimination towards the unprivileged group (and the privileged group).
    Lower values indicate less discrimination.

    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
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
    The protected attribute `z` must be binary.

    This function computes the difference between 1 and the Disparate Impact
    ratio. A value of 0 indicates fairness. A positive value indicates
    discrimination towards the unprivileged group. A negative value indicates
    discrimination towards the privileged group.


    Parameters
    ----------
    y: np.array
        Flattened binary array, can be the prediction or the truth label.
    z: np.array
        Flattened binary array of shape y, represents the protected attribute.
    positive_label: int, optional
        Label considered as positive. Default is 1.
    privileged_group: int, optional
        Label considered as privileged. Default is 1.

    Returns
    -------
    float
        The difference in objective Disparate Impact ratio.
    """
    return 1 - disparate_impact_ratio(y, z, positive_label, privileged_group, **kwargs)
