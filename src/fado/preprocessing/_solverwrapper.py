from fado.preprocessing import Preprocessing

import numpy as np


class HeuristicWrapper(Preprocessing):

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
            method that takes f, d and returns a binary np.array of shape (d, ) that optimizes disc_measure
            and the value of f at the solution.
        disc_measure: callable
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
    if type(protected_attributes) == str:
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
