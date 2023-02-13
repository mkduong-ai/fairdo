from fado.preprocessing import Preprocessing

import numpy as np
import pandas as pd


def f(binary_vector, dataframe, label, protected_attributes, disc_measure):
    """

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

    # Note: This does not handle multiple protected attributes
    y = y.to_numpy().flatten()
    z = z.to_numpy().flatten()
    return disc_measure(x=x, y=y, z=z)


class PreprocessingHeuristicWrapper(Preprocessing):

    def __init__(self,
                 heuristic,
                 disc_measure,
                 protected_attribute,
                 label,
                 **kwargs):
        """
        Preprocessing wrapper for heuristic methods

        Parameters
        ----------
        heuristic: callable
            method that takes func, dims and returns a binary np.array of shape (dims, ) that optimizes func and the
            value of func at the solution.
        func: callable
            function to be optimized by heuristic
        protected_attribute: str
        label: str
            predicting label
        kwargs: dict
            additional arguments for heuristic
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

        Parameters
        ----------
        dataset: pandas DataFrame

        Returns
        -------

        """
        self.dataset = dataset.copy()
        self.func = lambda binary_vector: f(binary_vector,
                                            self.dataset,
                                            self.label,
                                            self.protected_attribute,
                                            disc_measure=self.disc_measure)
        self.dims = len(self.dataset)

    def transform(self):
        """

        Returns
        -------
        pandas DataFrame
            preprocessed (fair) dataset
        """
        mask = self.heuristic(self.func, self.dims)[0] == 1
        return self.dataset[mask]
