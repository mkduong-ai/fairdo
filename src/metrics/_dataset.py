import numpy as np
import warnings


def statistical_parity_difference(y: np.array, z: np.array,
                                  positive_label=1, privileged_group=1, **kwargs) -> float:
    """
    Difference in equalized opportunities (Lee et al. 2022)

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