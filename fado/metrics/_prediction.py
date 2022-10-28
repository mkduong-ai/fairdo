import numpy as np


def equal_opportunity_difference(y_true: np.array, y_pred: np.array, z: np.array,
                                 positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Difference in Equality of Opportunity (Lee et al. 2022)

    Parameters
    ----------

    y_true: flattened binary array
    y_pred: flattened binary array of shape y_true
    z: flattened binary array of shape y_true
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
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z = z.astype(int)
    priv_eo = np.sum(y_true & z & y_pred) / np.sum(y_true & z)
    unpriv_eo = np.sum(y_true & (1 - z) & y_pred) / np.sum(y_true & (1 - z))

    return priv_eo - unpriv_eo


def equal_opportunity_absolute_difference(*args, **kwargs):
    return np.abs(equal_opportunity_difference(*args, **kwargs))


def predictive_equality_difference(y_true: np.array, y_pred: np.array, z: np.array,
                                   positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Difference in Predictive Equality

    Parameters
    ----------

    y_true: flattened binary array
    y_pred: flattened binary array of shape y_true
    z: flattened binary array of shape y_true
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
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    z = z.astype(int)
    priv_eo = np.sum(y_pred & (1-y_true) & z) / np.sum((1-y_true) & z)
    unpriv_eo = np.sum(y_pred & (1-y_true) & (1-z)) / np.sum((1-y_true) & (1-z))

    return priv_eo - unpriv_eo


def predictive_equality_absolute_difference(*args, **kwargs):
    return np.abs(predictive_equality_difference(*args, **kwargs))


def average_odds_difference(y_true: np.array, y_pred: np.array, z: np.array,
                            positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Difference in average odds (aif360)

    Parameters
    ----------

    y_true: flattened binary array
    y_pred: flattened binary array of shape y_true
    z: flattened binary array of shape y_true
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
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    eod = equal_opportunity_difference(y_true, y_pred, z, positive_label, privileged_group)
    ped = predictive_equality_difference(y_true, y_pred, z, positive_label, privileged_group)

    return (eod + ped)/2


def average_odds_error(y_true: np.array, y_pred: np.array, z: np.array,
                       positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Average odds error

    Parameters
    ----------

    y_true: flattened binary array
    y_pred: flattened binary array of shape y_true
    z: flattened binary array of shape y_true
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
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    eod = equal_opportunity_absolute_difference(y_true, y_pred, z, positive_label, privileged_group)
    ped = predictive_equality_absolute_difference(y_true, y_pred, z, positive_label, privileged_group)

    return (eod + ped)/2


