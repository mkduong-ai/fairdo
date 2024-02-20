import abc

import numpy as np
import pandas as pd


class Preprocessing(metaclass=abc.ABCMeta):
    """
    Base class for all preprocessing methods.

    Parameters
    ----------
    protected_attribute: str
    label: str
        predicting label
    dataset: pd.DataFrame
        original dataset
    transformed_data: pd.DataFrame
        dataset after transformation/pre-processing
    """
    def __init__(self, protected_attribute, label):
        """
        Base class for all preprocessing methods.

        Parameters
        ----------
        protected_attribute: str
        label: str
            predicting label
        """
        self.protected_attribute = protected_attribute
        self.label = label
        self.dataset = None
        self.transformed_data = None

    @abc.abstractmethod
    def transform(self):
        pass

    def fit(self, dataset):
        """
        Copies the dataset to the class and checks if the dataset is valid, i.e., all columns are numeric.

        Parameters
        ----------
        dataset: pd.DataFrame

        Returns
        -------
        self
        """
        self.dataset = dataset.copy()
        self._check_valid_datatype()
        return self

    def fit_transform(self, *args, **kwargs):
        """
        Fit the model to the dataset and transform the dataset.

        Parameters
        ----------
        args: list
            Positional arguments for the `fit` method.
        kwargs: dict
            Keyword arguments for the `fit` method.

        Returns
        -------
        self
        """
        return self.fit(*args, **kwargs).transform()

    def _check_valid_datatype(self):
        if not isinstance(self.dataset, pd.DataFrame):
            try:
                self.dataset = self.dataset.convert_to_dataframe()[0]
            except:
                raise Exception('Type of dataset is unknown.')

        # check if all columns are numeric (including boolean)
        is_number = np.vectorize(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, bool))
        if not np.all(is_number(self.dataset.dtypes)):
            raise Exception(f"All columns must be numeric. The datatypes of the columns are:\n{self.dataset.dtypes}")


class OriginalData(Preprocessing):
    """
    This class is used to return the original dataset.
    """
    def __init__(self, **kwargs):
        self.dataset = None
        self.transformed_data = None

    def transform(self):
        """
        Returns the original dataset.

        Returns
        -------
        pd.DataFrame
            The original dataset.
        """
        self.transformed_data = self.dataset
        return self.dataset


class Random(Preprocessing):
    """
    This class is used to return a random subset of the dataset.
    The size of the subset is determined by the `frac` parameter.
    """
    def __init__(self, frac=0.8,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.random_state = random_state
        np.random.seed(self.random_state)

    def transform(self):
        """
        Returns a random subset of the dataset.

        Returns
        -------
        pd.DataFrame
            The random subset of the dataset.
        """
        if self.dataset is None:
            raise Exception('Model not fitted.')

        self.transformed_data = self.dataset.sample(frac=self.frac, axis=0,
                                                    random_state=self.random_state)
        return self.transformed_data
