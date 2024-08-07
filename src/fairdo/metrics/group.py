import numpy as np
import warnings
from itertools import product

from fairdo.utils.helper import generate_pairs


def statistical_parity_abs_diff_multi(y: np.array, z: np.array,
                                      agg_attribute=np.max,
                                      agg_group=np.max,
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
    # z = z.astype(int)

    # get unique values for each attribute
    groups = [np.unique(z[:, i]) for i in range(z.shape[1])]
    # print(groups)
    # get statistical parity for each attribute
    attributes_disparity = []
    for k, zk in enumerate(groups):
        # calculate statistical parities for all groups in one pass
        parities = {i: np.sum(y & (z[:, k] == i)) / np.sum(z[:, k] == i) for i in zk}
        # parities = {i: np.mean(y[z[:, k] == i]) for i in zk} # slower than the above method
        # generate all possible pairs of values for the attribute
        if agg_group == np.max:
            group_disparity = np.max(list(parities.values())) - np.min(list(parities.values()))
            attributes_disparity.append(group_disparity)
        else:
            pairs = generate_pairs(zk)
            group_disparity = [np.abs(parities[i] - parities[j]) for i, j in pairs]

            try:
                attributes_disparity.append(agg_group(group_disparity))
            except ValueError:
                warnings.warn(f"Could not aggregate disparity for attribute {k} with aggregation function {agg_group}. "
                            f"The disparity for this attribute is {group_disparity}. "
                            f"Returning disparity of 0.")
                attributes_disparity.append(0)
    return agg_attribute(attributes_disparity)


def statistical_parity_abs_diff_intersectionality(y: np.array, z: np.array,
                                                  agg_group=np.max,
                                                  **kwargs) -> float:
    """
    Calculate the absolute difference in statistical parity for multiple non-binary protected attributes.
    Intersections from all protected attributes are considered.
    Protected attributes `z[i]` can be binary or non-binary.

    Parameters
    ----------
    y: np.array
        Flattened binary array of shape (n_samples,), can be the prediction or the truth label.
    z: np.array
        Array of shape (n_samples, n_protected_attributes) representing the protected attribute.
    agg_group: callable, optional
        Aggregation function for the group. Default is np.sum.
    **kwargs: dict
        Additional keyword arguments.
    """
    z_subgroups = np.apply_along_axis(lambda x: ''.join(map(str, x)), axis=1, arr=z)
    all_subgroups = list(set(z_subgroups))
    parities = {i: np.sum(y & (z_subgroups == i)) / np.sum(z_subgroups == i) for i in all_subgroups}
    
    if agg_group == np.max:
        group_disparity = np.max(list(parities.values())) - np.min(list(parities.values()))
    else:
        pairs = generate_pairs(list(all_subgroups))
        group_disparity = [np.abs(parities[i] - parities[j]) for i, j in pairs]

    return agg_group(group_disparity)


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


def equal_opportunity_difference(y_true: np.array, y_pred: np.array, z: np.array,
                                 positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Compute the difference in Equality of Opportunity [1] between
    the privileged group and the unprivileged group.

    Equality of Opportunity [1] is a fairness metric
    that measures the difference in true positive rates between the privileged and unprivileged groups.
    This function returns a float representing that difference.
    A value of 0 indicates perfect fairness, positive values indicate bias
    against the unprivileged group, while negative values indicate
    bias against the privileged group.

    [1] Equality of Opportunity (Hardt, Price, Srebro, 2016)](https://arxiv.org/abs/1610.02413)

    Parameters
    ----------
    y_true: numpy.array
        The true binary labels as a flattened array.
    y_pred: numpy.array
        The predicted binary labels from the model.
        Should be of the same shape as y_true.
    z: numpy.array
        The protected attribute as a binary array.
        This array indicates the group (privileged or unprivileged) for each instance in the data.
        Should be of the same shape as y_true.
    positive_label: int, optional (default=1)
        The label considered as positive in the dataset.
    privileged_group: int, optional (default=1)
        The label that denotes the privileged group.
        If 0, the function will treat the unprivileged group as the privileged group.

    Returns
    -------
    float
        The difference in Equality of Opportunity between the privileged and unprivileged groups.
    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z = z.astype(int)
    priv_eo = np.sum(y_pred & y_true & z) / np.sum(y_true & z)
    unpriv_eo = np.sum(y_pred & y_true & (1 - z)) / np.sum(y_true & (1 - z))

    return priv_eo - unpriv_eo


def equal_opportunity_abs_diff(*args, **kwargs):
    """
    Compute the absolute difference in Equality of Opportunity [1].

    [1] Equality of Opportunity (Hardt, Price, Srebro, 2016) (https://arxiv.org/abs/1610.02413)

    Parameters
    ----------
    *args: arguments
        Variable length argument list to be passed to `equal_opportunity_difference` function.
    **kwargs: keyword arguments
        Arbitrary keyword arguments to be passed to `equal_opportunity_difference` function.

    Returns
    -------
    float
        The absolute difference in Equality of Opportunity between privileged and unprivileged groups.
    """
    return np.abs(equal_opportunity_difference(*args, **kwargs))


def predictive_equality_difference(y_true: np.array, y_pred: np.array, z: np.array,
                                   positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the difference in Predictive Equality.

    Parameters
    ----------
    y_true: numpy.array
        True binary labels as a flattened array.
    y_pred: numpy.array
        Predicted binary labels as a flattened array. Must have same shape as y_true.
    z: numpy.array
        Binary array denoting privileged (1) or unprivileged (0) group. Same shape as y_true.
    positive_label: int, optional
        Label considered as positive, default is 1.
    privileged_group: int, optional
        Label representing the privileged group, default is 1.

    Returns
    -------
    float
        The difference in Predictive Equality between privileged and unprivileged groups.

    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z = z.astype(int)
    priv_eo = np.sum(y_pred & (1-y_true) & z) / np.sum((1-y_true) & z)
    unpriv_eo = np.sum(y_pred & (1-y_true) & (1-z)) / np.sum((1-y_true) & (1-z))

    return priv_eo - unpriv_eo


def predictive_equality_abs_diff(*args, **kwargs):
    """
    Compute the absolute difference in Predictive Equality.

    Parameters
    ----------
    *args: arguments
        Variable length argument list to be passed to `predictive_equality_difference` function.
    **kwargs: keyword arguments
        Arbitrary keyword arguments to be passed to `predictive_equality_difference` function.

    Returns
    -------
    float
        The absolute difference in Predictive Equality between privileged and unprivileged groups.
    """
    return np.abs(predictive_equality_difference(*args, **kwargs))


def average_odds_difference(y_true: np.array, y_pred: np.array, z: np.array,
                            positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Calculate the difference in Average Odds between privileged and unprivileged groups.

    [1] Equality of Opportunity in Supervised Learning (Hardt, Price, Srebro, 2016) (https://arxiv.org/abs/1610.02413)

    Parameters
    ----------
    y_true: numpy.array
        Flattened array of true binary labels.
    y_pred: numpy.array
        Flattened array of predicted binary labels. Must have same shape as y_true.
    z: numpy.array
        Binary array indicating privileged (1) or unprivileged (0) group. Same shape as y_true.
    positive_label: int, optional
        Label considered as positive, default is 1.
    privileged_group: int, optional
        Label denoting the privileged group, default is 1.

    Returns
    -------
    float
        The difference in Average Odds between privileged and unprivileged groups.
    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    eod = equal_opportunity_difference(y_true, y_pred, z, positive_label, privileged_group)
    ped = predictive_equality_difference(y_true, y_pred, z, positive_label, privileged_group)

    return (eod + ped)/2


def average_odds_error(y_true: np.array, y_pred: np.array, z: np.array,
                       positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Compute the Average Odds Error.
    Can be used as an objective function to minimize.

    Parameters
    ----------
    y_true: numpy.array
        Flattened array of true binary labels.
    y_pred: numpy.array
        Flattened array of predicted binary labels. Must have same shape as y_true.
    z: numpy.array
        Binary array indicating privileged (1) or unprivileged (0) group. Same shape as y_true.
    positive_label: int, optional
        Label considered as positive, default is 1.
    privileged_group: int, optional
        Label denoting the privileged group, default is 1.

    Returns
    -------
    float
        The Average Odds Error between privileged and unprivileged groups.
    """
    # invert privileged and positive label if required
    if privileged_group == 0:
        z = 1 - z
    if positive_label == 0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    eod = equal_opportunity_abs_diff(y_true, y_pred, z, positive_label, privileged_group)
    ped = predictive_equality_abs_diff(y_true, y_pred, z, positive_label, privileged_group)

    return (eod + ped)/2
