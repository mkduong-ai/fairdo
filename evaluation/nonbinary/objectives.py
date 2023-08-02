import numpy as np
import pandas as pd

# generate synthetic datapoints
from sdv.tabular import GaussianCopula

from fado.metrics import statistical_parity_abs_diff


def f_remove(binary_vector, dataframe, label, protected_attributes,
             disc_measure=statistical_parity_abs_diff,
             **kwargs):
    """
    Determine which data points can be removed from the training set to prevent discrimination.
    This can be easily applied for any dataset where discrimination prevention happens before
    training any machine learning model.

    Parameters
    ----------
    binary_vector: np.array
    dataframe: pandas DataFrame
    label: str
    protected_attributes: list of strings
    disc_measure: callable
        takes in x, y, z and return a numeric value

    Returns
    -------
    numeric
    """
    y = dataframe[label]
    z = dataframe[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataframe.drop(columns=cols_to_drop)

    # only keep the columns that are selected by the heuristic
    mask = np.array(binary_vector) == 1
    x, y, z = x[mask], y[mask], z[mask]

    # We handle multiple protected attributes by not flattening the z array
    y = y.to_numpy().flatten()
    z = z.to_numpy()
    if len(protected_attributes) == 1:
        z = z.flatten()
    return disc_measure(x=x, y=y, z=z)


def f_add(binary_vector, dataframe, synthetic_dataframe, label, protected_attributes,
          disc_measure=statistical_parity_abs_diff):
    """
    Generated (synthetic) data points are added to the original data to promote fairness.
    The question here is: Which of the generated data points should be added to the
    original data to prevent discrimination?

    Parameters
    ----------
    binary_vector: np.array
    dataframe: pandas DataFrame
    synthetic_dataframe: pandas DataFrame
    label: str
    protected_attributes: list of strings
    disc_measure: callable
        takes in x, y, z and return a numeric value

    Returns
    -------
    numeric
    """
    # mask on synthetic data
    mask = np.array(binary_vector) == 1
    synthetic_dataframe = synthetic_dataframe[mask]

    # concatenate synthetic data with original data
    dataframe = pd.concat([dataframe, synthetic_dataframe], axis=0)

    # evaluate on whole dataset
    y = dataframe[label]
    z = dataframe[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataframe.drop(columns=cols_to_drop)

    # We handle multiple protected attributes by not flattening the z array
    y = y.to_numpy().flatten()
    z = z.to_numpy()
    if len(protected_attributes) == 1:
        z = z.flatten()
    return disc_measure(x=x, y=y, z=z)
