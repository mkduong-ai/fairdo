# Standard library imports
from functools import partial

# Related third-party imports
import numpy as np
import pandas as pd

# fairdo imports
from fairdo.preprocessing import Preprocessing
from fairdo.metrics import statistical_parity_abs_diff_max


class HeuristicWrapper(Preprocessing):
    """
    A preprocessing wrapper class that applies a given heuristic method to optimize a given
    discrimination measure and outputs a pre-processed dataset.
    The pre-processed dataset is a subset of the original dataset, where the columns are
    selected based on the heuristic method.

    Attributes
    ----------
    heuristic: callable
        The method that optimizes the discrimination measure. It takes a function and the
        number of dimensions, and returns a binary numpy array of shape (d, ) indicating
        selected columns and the optimized discrimination measure.
    func: callable
        The discrimination measure function to be optimized. It is defined within the `fit`
        method.
    dims: int
        The number of dimensions or columns in the dataset. It is defined within the `fit`
        method.
    disc_measure: callable
        The discrimination measure to be optimized. It takes the feature matrix (x), labels
        (y), and protected attributes (z) and returns a numeric value.
    dataset: pd.DataFrame
        The dataset to be preprocessed. It is defined within the `fit` method.
    """

    def __init__(self,
                 heuristic,
                 protected_attribute,
                 label,
                 disc_measure=statistical_parity_abs_diff_max,
                 **kwargs):
        """
        Constructs all the necessary attributes for the HeuristicWrapper object.

        Parameters
        ----------
        heuristic: callable
            The method that optimizes the discrimination measure.
        protected_attribute: str or List[str]
            The protected attribute in the dataset.
        label: str
            The target variable in the dataset.
        disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
            The discrimination measure to be optimized.
            Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and
            minimum statistical parity values.
        kwargs: dict
            Additional arguments for the heuristic method.
        """
        self.heuristic = heuristic
        self.func = None
        self.dims = None
        self.disc_measure = disc_measure

        # required by Preprocessing
        self.dataset = None
        super().__init__(protected_attribute=protected_attribute, label=label)

    def fit(self, dataset, sample_dataset=None, approach='remove'):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to be preprocessed.
        sample_dataset: pd.DataFrame, optional
            The sample dataset to be used for the 'add' approach. It is required only if the
            'add' approach is used.
        approach: str
            The approach to be used for the heuristic method. It can be either 'remove' or 'add'.

        Returns
        -------
        self
        """
        self.dataset = dataset.copy()
        if approach == 'remove':
            self.func = partial(f_remove,
                                dataframe=self.dataset,
                                label=self.label,
                                protected_attributes=self.protected_attribute,
                                disc_measure=self.disc_measure)
        elif approach == 'add':
            if sample_dataset is None:
                raise ValueError('Sample dataset is required for the \'add\' approach.')

            self.func = partial(f_add,
                                dataframe=self.dataset,
                                sample_dataframe=sample_dataset,
                                label=self.label,
                                protected_attributes=self.protected_attribute,
                                disc_measure=self.disc_measure)
        self.dims = len(self.dataset)
        return self

    def transform(self):
        """
        Applies the heuristic method to the dataset and returns a preprocessed version of it.

        Returns
        -------
        pd.DataFrame
            The preprocessed (fair) dataset.
        """
        mask = self.heuristic(f=self.func, d=self.dims)[0] == 1
        return self.dataset[mask]


def f_remove(binary_vector, dataframe, label, protected_attributes,
             disc_measure=statistical_parity_abs_diff_max):
    """
    Calculates a given discrimination measure on a dataframe for a set of selected columns.
    In other words, determine which data points can be removed from the training set to prevent discrimination.
    This can be easily applied for any dataset where discrimination prevention happens before
    training any machine learning model.

    Parameters
    ----------
    binary_vector: np.array
        Binary vector indicating which columns to include in the discrimination measure calculation.
    dataframe: pd.DataFrame
        The data to calculate the discrimination measure on.
    label: str
        The column in the dataframe to use as the target variable.
    protected_attributes: Union[str, List[str]]
        The column or columns in the dataframe to consider as protected attributes.
    disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.
        Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and minimum
        statistical parity values.

    Returns
    -------
    float
        The calculated discrimination measure.
    """
    if isinstance(protected_attributes, str):
        protected_attributes = [protected_attributes]

    y = dataframe[label]
    z = dataframe[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataframe.drop(columns=cols_to_drop)

    # only keep the columns that are selected by the heuristic
    mask = np.array(binary_vector) == 1
    x, y, z = x[mask], y[mask], z[mask]

    # Note: This does not handle multiple protected attributes
    y = y.to_numpy().flatten()
    z = z.to_numpy().flatten()
    return disc_measure(x=x, y=y, z=z)


def f_add(binary_vector, dataframe, sample_dataframe, label, protected_attributes,
          disc_measure=statistical_parity_abs_diff_max):
    """
    Additional sample data points are added to the original data to promote fairness.
    The sample data can be synthetic data.
    The question here is: Which of the data points from the sample set should be added to the
    original data to prevent discrimination?

    Parameters
    ----------
    binary_vector: np.array
        Binary vector indicating which columns to include in the discrimination measure calculation.
    dataframe: pd.DataFrame
        The data to calculate the discrimination measure on.
    sample_dataframe: pd.DataFrame
        Extra samples to be added to the original data. Samples can be synthetic data.
    label: str
        The column in the dataframe to use as the target variable.
    protected_attributes: Union[str, List[str]]
        The column or columns in the dataframe to consider as protected attributes.
    disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.
        Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and minimum
        statistical parity values.

    Returns
    -------
    float
        The calculated discrimination measure.
    """
    if isinstance(protected_attributes, str):
        protected_attributes = [protected_attributes]

    # mask on sample data
    mask = np.array(binary_vector) == 1
    sample_dataframe = sample_dataframe[mask]

    # concatenate synthetic data with original data
    dataframe = pd.concat([dataframe, sample_dataframe], axis=0)

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
