import abc

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.decomposition import PCA


class Preprocessing(metaclass=abc.ABCMeta):

    def __init__(self, protected_attribute, label,
                 frac=0.8,
                 drop_protected_attribute=False,
                 drop_label=False,
                 drop_features=False,
                 dim_reduction=False, n_components=2):
        """

        Parameters
        ----------
        protected_attribute: str
        label: str
            predicting label
        drop_protected_attribute: boolean
        drop_label: boolean
        drop_features: boolean
        dim_reduction: boolean
        n_components: int
            Only used if dim_reduction is true.
        """
        if frac in (0, 1):
            raise ValueError('frac can not be set to 0 or 1.')
        self.frac = frac
        self.protected_attribute = protected_attribute
        self.label = label
        self.drop_protected_attribute = drop_protected_attribute
        self.drop_label = drop_label
        self.drop_features = drop_features
        self.dim_reduction = dim_reduction
        self.n_components = n_components
        self.dataset = None
        self.transformed_data = None
        self.samples = None

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
        self.dataset = dataset
        if not isinstance(self.dataset, pd.DataFrame):
            try:
                self.dataset = self.dataset.convert_to_dataframe()[0]
            except:
                print('Type of dataset is unknown.')

        if self.drop_protected_attribute and self.protected_attribute is not None:
            self.dataset = self.dataset.drop(
                columns=self.protected_attribute, axis=1, inplace=False)
        if self.drop_label and self.label is not None:
            self.dataset = self.dataset.drop(
                columns=self.label, axis=1, inplace=False)
        if self.drop_features:
            if None in (self.protected_attribute, self.label):
                raise Exception('Protected attribute or label not given.'
                                'Not possible to determine which columns are features to drop.')
            else:
                self.dataset = self.dataset[[self.protected_attribute, self.label]]

        self.transform_data()
        if self.dim_reduction:
            self.reduce_dimensionality(self.n_components)
        return self

    def fit_transform(self, dataset):
        return self.fit(dataset).transform()

    def reduce_dimensionality(self, n_components=2):
        pca = PCA(n_components)
        self.transformed_data = pd.DataFrame(
            pca.fit_transform(self.transformed_data))

    def transform_data(self):
        # One Hot Encoding
        is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
        if np.all(is_number(self.dataset.dtypes)):
            self.transformed_data = self.dataset.copy()
        else:
            raise Exception(f"All columns must be numeric. The datatypes of the columns are:\n{self.dataset.dtypes}")
            #self.transformed_data = pd.get_dummies(self.dataset)

        # drop columns with same values in all rows
        # nunique = self.transformed_data.nunique()
        # cols_to_drop = nunique[nunique == 1].index
        # self.transformed_data = self.transformed_data.drop(
        #     columns=cols_to_drop, axis=1, inplace=False)

        # Min-Max Normalization
        # self.transformed_data = \
        #     (self.transformed_data - self.transformed_data.min()) / \
        #     (self.transformed_data.max() - self.transformed_data.min())


class PreprocessingWrapper:
    """
    Wrapper Class for pre-processing methods that take a
    pandas DataFrame and return a pandas DataFrame instead
    of a BinaryLabelDataset.
    """
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        self.__class__.__name__ = type(preprocessing).__name__

    def fit_transform(self, dataset_train: BinaryLabelDataset) -> BinaryLabelDataset:
        # z_train already included in x_train
        transformed_df_dataset = self.preprocessing.fit_transform(dataset_train.convert_to_dataframe()[0])

        transformed_dataset = BinaryLabelDataset(df=transformed_df_dataset,
                                                 protected_attribute_names=dataset_train.protected_attribute_names,
                                                 label_names=dataset_train.label_names)

        return transformed_dataset


class OriginalData:
    def __init__(self, **kwargs):
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        return self

    def transform(self):
        return self.dataset

    def fit_transform(self, dataset):
        return dataset


class Random(Preprocessing):

    def __init__(self, frac=0.8,
                 protected_attribute=None, label=None,
                 drop_protected_attribute=False,
                 drop_label=False,
                 drop_features=False,
                 dim_reduction=False, n_components=2, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label,
                         drop_protected_attribute=drop_protected_attribute,
                         drop_label=drop_label,
                         drop_features=drop_features,
                         dim_reduction=dim_reduction, n_components=n_components)
        self.random_state = random_state

    def transform(self):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        return self.dataset.sample(frac=self.frac, axis=0,
                                   random_state=self.random_state)


