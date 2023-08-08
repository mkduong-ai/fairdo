import numpy as np


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


