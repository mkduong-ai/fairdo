from fairdo.preprocessing import Preprocessing
from fairdo.metrics import statistical_parity_abs_diff

# third party
import numpy as np
import pandas as pd

# generate synthetic datapoints
from sdv.tabular import GaussianCopula


class MetricOptimizer(Preprocessing):
    """
    Deletes samples or adds generated samples to decrease the discrimination/bias in the given dataset.
    """
    def __init__(self, protected_attribute, label,
                 frac=0.75, m=5, eps=0,
                 additions=None,
                 deletions=None,
                 fairness_metric=statistical_parity_abs_diff,
                 data_generator='GaussianCopula',
                 data_generator_params=None, random_state=None):
        super().__init__(protected_attribute=protected_attribute,
                         label=label)
        if frac in (0, 1):
            raise ValueError('frac can not be set to 0 or 1.')
        self.frac = frac
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
            self.preproc = MetricOptRemover(protected_attribute=self.protected_attribute, label=self.label,
                                            frac=self.frac, m=self.m, eps=self.eps,
                                            deletions=self.deletions,
                                            fairness_metric=self.fairness_metric,
                                            random_state=self.random_state)
        else:
            self.preproc = MetricOptGenerator(protected_attribute=self.protected_attribute, label=self.label,
                                              frac=self.frac, m=self.m, eps=self.eps,
                                              additions=self.additions,
                                              fairness_metric=self.fairness_metric,
                                              data_generator=self.data_generator,
                                              random_state=self.random_state)
        self.__class__.__name__ = type(self.preproc).__name__

    def fit(self, dataset):
        self.preproc.fit(dataset)
        return self

    def transform(self):
        return self.preproc.transform()


class MetricOptRemover(Preprocessing):
    """
    Deletes samples which worsen the discrimination in the dataset
    """

    def __init__(self, protected_attribute, label,
                 frac=0.75, m=5, eps=0,
                 deletions=None,
                 fairness_metric=statistical_parity_abs_diff, random_state=None):
        """

        Parameters
        ----------
        protected_attribute: str
            String of column that is discriminated
        label: str
            String of column that represents the prediction label
        frac: float
            A fraction/factor of the given dataset's size
        m: int
            Number of candidates
        eps: float
        deletions: int
            If deletions is given, then frac is ignored. Number of samples to be removed from the dataset.
        fairness_metric:
            A fairness metric which can take x, y, or z as array parameters and calculates a fairness score.
        random_state: int
        """
        super().__init__(protected_attribute=protected_attribute,
                         label=label)
        if self.frac > 1:
            raise Exception('Fraction frac can not be greater than 1.')
        self.frac = frac
        self.fairness_metric = fairness_metric
        self.m = m
        self.eps = eps
        self.deletions = deletions
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self, index=False):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        # preparation
        samples = self.dataset.copy()

        if self.deletions is None:
            n = (len(samples) -
                 int(len(samples) * self.frac))
        else:
            n = self.deletions

        # check whether dataset is already fair
        x = np.array(samples.drop(columns=[self.label, self.protected_attribute]))
        y = np.array(samples[self.label])
        z = np.array(samples[self.protected_attribute])
        if self.fairness_metric(x=x, y=y, z=z) <= self.eps:
            return samples

        for i in range(1, n):
            # create candidates
            cands = samples.sample(n=min(self.m, len(samples)), replace=False)

            # list consists of single candidates removed from samples
            samples_wo_cands_list = [samples.drop(index=j, inplace=False) for j in cands.index]  # slowest operation

            discrimination_values = []
            for j in range(self.m):
                x = np.array(samples_wo_cands_list[j].drop(columns=[self.label, self.protected_attribute]))
                y = np.array(samples_wo_cands_list[j][self.label])
                z = np.array(samples_wo_cands_list[j][self.protected_attribute])
                discrimination_values.append(self.fairness_metric(x=x, y=y, z=z))

            opt_cand_index = np.argmin(discrimination_values)

            # update samples
            samples = samples_wo_cands_list[opt_cand_index]

            # stop criterion if fairness is fulfilled
            if discrimination_values[opt_cand_index] <= self.eps:
                break

        self.transformed_data = self.dataset.loc[samples.index]
        if index:
            return samples.index
        else:
            return self.transformed_data


class MetricOptGenerator(Preprocessing):
    """
    Deletes samples which worsen the discrimination in the dataset
    """

    def __init__(self, protected_attribute, label,
                 frac=1.25, m=5, eps=0,
                 additions=None,
                 fairness_metric=statistical_parity_abs_diff,
                 data_generator='GaussianCopula', random_state=None):
        """

        Parameters
        ----------
        protected_attribute: str
            String of column that is discriminated
        label: str
            String of column that represents the prediction label
        frac: float
            A multiplicative of the given dataset's size
        m: int
            Number of candidates
        eps: float
        additions: int
            If additions is given, then frac is ignored. Number of samples to be added to the dataset.
        fairness_metric:
            A fairness metric which can take x, y, or z as array parameters and calculates a fairness score.
        data_generator:
            A string indicating the generative model or the generative model itself with
            .fit() and .sample() methods.
        random_state: int
        """
        super().__init__(protected_attribute=protected_attribute,
                         label=label)
        if self.frac < 1:
            raise Exception('Fraction frac can not be less than 1.')
        self.frac = frac
        self.fairness_metric = fairness_metric
        self.m = m
        self.eps = eps
        self.additions = additions
        self.data_fitted = False
        if isinstance(data_generator, str):
            data_generators = {'GaussianCopula': GaussianCopula}
            # init data generator
            if data_generator in data_generators.keys():
                self.data_generator = data_generators[data_generator]()
            else:
                raise Exception('Unknown data generator.')
        else:
            self.data_generator = data_generator
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, dataset):
        """

        Parameters
        ----------
        dataset: pd.DataFrame

        Returns
        -------
        self
        """
        super().fit(dataset)

        # fit data generator to data if not fitted
        if not self.data_fitted:
            self.data_generator.fit(dataset)
            self.data_fitted = True

        return self

    def transform(self, index=False):
        if self.dataset is None:
            raise Exception('Model not fitted.')
        if None in (self.protected_attribute, self.label):
            raise Exception('Protected attribute or label not given.')

        # preparation
        samples = self.dataset.copy()

        if self.additions is None:
            n = (int(len(samples) * self.frac) -
                 len(samples))
        else:
            n = self.additions

        # check whether dataset is already fair
        x = np.array(samples.drop(columns=[self.label, self.protected_attribute]))
        y = np.array(samples[self.label])
        z = np.array(samples[self.protected_attribute])
        if self.fairness_metric(x=x, y=y, z=z) <= self.eps:
            return samples

        for i in range(0, n):
            # create candidates
            cands = self.data_generator.sample(self.m)

            # list consists of single candidates added to dataset
            samples_concat_list = [pd.concat([samples, cands.iloc[[j]]], ignore_index=True) for j in range(len(cands))]

            discrimination_values = []
            for j in range(self.m):
                x = np.array(samples_concat_list[j].drop(columns=[self.label, self.protected_attribute]))
                y = np.array(samples_concat_list[j][self.label])
                z = np.array(samples_concat_list[j][self.protected_attribute])
                discrimination_values.append(self.fairness_metric(x=x, y=y, z=z))

            opt_cand_index = np.argmin(discrimination_values)

            # update samples
            samples = samples_concat_list[opt_cand_index]

            # stop criterion if fairness is fulfilled
            if discrimination_values[opt_cand_index] <= self.eps:
                break

        self.transformed_data = samples
        return self.transformed_data
