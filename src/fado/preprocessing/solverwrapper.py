from fado.preprocessing import Preprocessing

import numpy as np
import pandas as pd


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

    Methods
    -------
    fit(dataset: pd.DataFrame):
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.
    transform() -> pd.DataFrame:
        Applies the heuristic method to the dataset and returns a preprocessed version of it.
    """

    def __init__(self,
                 heuristic,
                 disc_measure,
                 protected_attribute,
                 label,
                 **kwargs):
        """
        Constructs all the necessary attributes for the HeuristicWrapper object.

        Parameters
        ----------
        heuristic: callable
            The method that optimizes the discrimination measure.
        disc_measure: callable
            The discrimination measure to be optimized.
        protected_attribute: str
            The protected attribute in the dataset.
        label: str
            The target variable in the dataset.
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

    def fit(self, dataset):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to be preprocessed.
        """
        self.dataset = dataset.copy()
        self.func = lambda binary_vector: f_remove(binary_vector,
                                                   self.dataset,
                                                   self.label,
                                                   self.protected_attribute,
                                                   disc_measure=self.disc_measure)
        self.dims = len(self.dataset)

    def transform(self):
        """
        Applies the heuristic method to the dataset and returns a preprocessed version of it.

        Returns
        -------
        pd.DataFrame
            The preprocessed (fair) dataset.
        """
        mask = self.heuristic(self.func, self.dims)[0] == 1
        return self.dataset[mask]


def f_remove(binary_vector, dataframe, label, protected_attributes,
             disc_measure):
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
    disc_measure: callable
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.

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
          disc_measure):
    """
    Additional sample data points are added to the original data to promote fairness.
    The sample data can be synthetic data.
    The question here is: Which of the data points from the sample should be added to the
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
    disc_measure: callable
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.

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
