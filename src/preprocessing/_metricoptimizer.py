import numpy as np

from src.preprocessing._base import Preprocessing
from src.metrics import statistical_parity_absolute_difference


class MetricOptimizer(Preprocessing):
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
        if None in (self.protected_attribute, self.label):
            raise Exception('Protected attribute or label not given.')

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
            samples_wo_cands_list = [samples.drop(index=i, inplace=False) for i in cands.index] # slowest operation

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
