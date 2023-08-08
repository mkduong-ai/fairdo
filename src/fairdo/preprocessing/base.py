import abc

import numpy as np
import pandas as pd


class Preprocessing(metaclass=abc.ABCMeta):

    def __init__(self, protected_attribute, label,
                 frac=0.8):
        """

        Parameters
        ----------
        protected_attribute: str
        label: str
            predicting label
        """
        if frac in (0, 1):
            raise ValueError('frac can not be set to 0 or 1.')
        self.frac = frac
        self.protected_attribute = protected_attribute
        self.label = label
        self.dataset = None
        self.transformed_data = None

    @abc.abstractmethod
    def transform(self):
        pass

    def fit(self, dataset):
        """

        Parameters
        ----------
        dataset: DataFrame

        Returns
        -------
        self
        """
        self.dataset = dataset.copy()
        self.check_valid_datatype()
        return self

    def fit_transform(self, dataset):
        return self.fit(dataset).transform()

    def check_valid_datatype(self):
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

    def __init__(self, **kwargs):
        self.dataset = None
        self.transformed_data = None

    def transform(self):
        self.transformed_data = self.dataset
        return self.dataset


class Random(Preprocessing):

    def __init__(self, frac=0.8,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.random_state = random_state
        np.random.seed(self.random_state)

    def transform(self):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        self.transformed_data = self.dataset.sample(frac=self.frac, axis=0,
                                                    random_state=self.random_state)
        return self.transformed_data
