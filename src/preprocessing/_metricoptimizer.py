import numpy as np
import pandas as pd

# generate synthetic datapoints
from sdv.tabular import GaussianCopula, CTGAN, TVAE
from sdv.lite import TabularPreset

from src.preprocessing._base import Preprocessing
from src.metrics import statistical_parity_absolute_difference


class MetricOptimizer(Preprocessing):
    """
    Deletes samples which worsen the discrimination in the dataset
    """

    def __init__(self, frac=0.8, m=5, eps=0,
                 additions=None,
                 deletions=None,
                 fairness_metric=statistical_parity_absolute_difference,
                 protected_attribute=None, label=None,
                 data_generator='GaussianCopula',
                 data_generator_params=None,
                 drop_protected_attribute=False,
                 drop_label=False,
                 drop_features=False,
                 dim_reduction=False, n_components=2, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label,
                         drop_protected_attribute=drop_protected_attribute,
                         drop_label=drop_label,
                         drop_features=drop_features,
                         dim_reduction=dim_reduction, n_components=n_components)
        self.preproc = None
        self.fairness_metric = fairness_metric
        self.m = m
        self.eps = eps
        # generating data
        self.additions = additions
        self.data_generator = data_generator
        self.data_generator_params = data_generator_params
        # removing data
        self.deletions = deletions
        self.random_state = random_state
        np.random.seed(random_state)

        self.init_preproc()

    def init_preproc(self):
        if self.frac < 1:
            self.preproc = MetricOptRemover(frac=self.frac, m=self.m, eps=self.eps,
                                            deletions=None,
                                            fairness_metric=self.fairness_metric,
                                            protected_attribute=self.protected_attribute, label=self.label,
                                            drop_protected_attribute=self.drop_protected_attribute,
                                            drop_label=self.drop_label,
                                            drop_features=self.drop_features,
                                            dim_reduction=self.dim_reduction, n_components=self.n_components,
                                            random_state=self.random_state)
        else:
            self.preproc = MetricOptGenerator(frac=self.frac, m=self.m, eps=self.eps,
                                              additions=self.additions,
                                              fairness_metric=self.fairness_metric,
                                              protected_attribute=self.protected_attribute, label=self.label,
                                              data_generator_str=self.data_generator,
                                              data_generator_params=self.data_generator_params,
                                              drop_protected_attribute=self.drop_protected_attribute,
                                              drop_label=self.drop_label,
                                              drop_features=self.drop_features,
                                              dim_reduction=self.dim_reduction, n_components=self.n_components,
                                              random_state=self.random_state)

    def fit(self, dataset):
        self.preproc.fit(dataset)
        return self

    def transform(self):
        return self.preproc.transform()


class MetricOptRemover(Preprocessing):
    """
    Deletes samples which worsen the discrimination in the dataset
    """

    def __init__(self, frac=0.8, m=5, eps=0,
                 deletions=None,
                 fairness_metric=statistical_parity_absolute_difference,
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
        self.fairness_metric = fairness_metric
        self.m = m
        self.eps = eps
        self.deletions = deletions
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self):
        """

        Parameters
        ----------
        fairness_metric: callable
            metric from fairness notion for datasets
            Smaller values indicate smaller discrimination

        Returns
        -------

        """
        if self.dataset is None:
            raise Exception('Model not fitted.')
        if self.protected_attribute is None:
            raise Exception('Protected attribute is None.')
        if self.label is None:
            raise Exception('Label is None.')

        z = self.transformed_data[self.protected_attribute].to_numpy()
        y = self.transformed_data[self.label].to_numpy()

        # preparation
        samples = self.transformed_data.copy()

        if self.deletions is None:
            n = (len(self.transformed_data) -
                 int(len(self.transformed_data) * self.frac))
        else:
            n = self.deletions
        for i in range(1, n):
            # create candidates
            cands = samples.sample(n=min(self.m, len(samples)), replace=False)

            # list consists of single candidates removed from samples
            samples_wo_cands_list = [samples.drop(index=i, inplace=False) for i in cands.index]  # slowest operation

            discrimination_values = []
            for j in range(self.m):
                x = samples_wo_cands_list[j].drop(columns=[self.label, self.protected_attribute])
                y = samples_wo_cands_list[j][self.label]
                z = samples_wo_cands_list[j][self.protected_attribute]
                discrimination_values.append(self.fairness_metric(x=x, y=y, z=z))

            opt_cand_index = np.argmin(discrimination_values)

            # update samples
            samples = samples_wo_cands_list[opt_cand_index]

            # stop criterion if fairness is fulfilled
            if self.eps > 0:
                if discrimination_values[opt_cand_index] <= self.eps:
                    break

        self.samples = self.dataset.loc[samples.index]
        return self.samples


class MetricOptGenerator(Preprocessing):
    """
    Deletes samples which worsen the discrimination in the dataset
    """

    def __init__(self, frac=1.2, m=5, eps=0,
                 additions=None,
                 fairness_metric=statistical_parity_absolute_difference,
                 protected_attribute=None, label=None,
                 data_generator_str='GaussianCopula',
                 data_generator_params=None,
                 drop_protected_attribute=False,
                 drop_label=False,
                 drop_features=False,
                 dim_reduction=False, n_components=2, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label,
                         drop_protected_attribute=drop_protected_attribute,
                         drop_label=drop_label,
                         drop_features=drop_features,
                         dim_reduction=dim_reduction, n_components=n_components)
        self.fairness_metric = fairness_metric
        self.m = m
        self.eps = eps
        self.additions = additions
        if data_generator_str == 'TabularPreset':
            self.data_generator = TabularPreset(name='FAST_ML')
        else:
            # init data generator
            data_generators = {'GaussianCopula': GaussianCopula}
            if data_generator_params is not None:
                    self.data_generator = data_generators[data_generator_str](**data_generator_params)
            else:
                self.data_generator = data_generators[data_generator_str]

        self.data_generator_params = data_generator_params
        self.random_state = random_state
        np.random.seed(random_state)

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

        # fit data generator to data
        self.data_generator.fit(self.dataset)

        return self

    def transform(self):
        """

        Parameters
        ----------
        fairness_metric: callable
            metric from fairness notion for datasets
            Smaller values indicate smaller discrimination

        Returns
        -------

        """
        if self.dataset is None:
            raise Exception('Model not fitted.')
        if None in (self.protected_attribute, self.label):
            raise Exception('Protected attribute or label not given.')

        z = self.transformed_data[self.protected_attribute].to_numpy()
        y = self.transformed_data[self.label].to_numpy()

        # preparation
        samples = self.transformed_data.copy()

        if self.additions is None:
            n = (int(len(self.transformed_data) * self.frac) -
                 len(self.transformed_data))
        else:
            n = self.additions
        for i in range(1, n):
            # create candidates
            cands = self.data_generator.sample(self.m)

            # list consists of single candidates added to dataset
            samples_concat_list = [pd.concat([samples, cands.iloc[[i]]], ignore_index=True) for i in range(len(cands))]

            discrimination_values = []
            for j in range(self.m):
                x = samples_concat_list[j].drop(columns=[self.label, self.protected_attribute])
                y = samples_concat_list[j][self.label]
                z = samples_concat_list[j][self.protected_attribute]
                discrimination_values.append(self.fairness_metric(x=x, y=y, z=z))

            opt_cand_index = np.argmin(discrimination_values)

            # update samples
            samples = samples_concat_list[opt_cand_index]

            # stop criterion if fairness is fulfilled
            if self.eps > 0:
                if discrimination_values[opt_cand_index] <= self.eps:
                    break

        self.samples = self.dataset.loc[samples.index]
        return self.samples
