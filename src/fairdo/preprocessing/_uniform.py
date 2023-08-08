from fairdo.preprocessing.base import Preprocessing

# third party
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


class LowDiscMinimization(Preprocessing):

    def __init__(self, frac=0.8, m=5,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.m = m
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        sample_set = self.dataset.copy()
        samples = self.dataset.sample(n=1, replace=False)
        sample_set.drop(index=samples.index, inplace=True)

        n = int(len(self.dataset) * self.frac)
        for i in range(1, n):
            # create candidates
            cands = sample_set.sample(n=min(self.m, len(sample_set)), replace=False)

            # creating multiple lists with different samples + candidates
            multi_samples = [
                pd.concat((samples, cands.iloc[j:j + 1]), axis=0) for j in range(min(self.m, len(self.dataset - i)))]

            # calculate the discrepancies
            discrepancies = list(map(discrepancy, multi_samples))
            min_disc_idx = np.argmin(discrepancies)

            # candidate with lowest discrepancy is the next sample
            samples = multi_samples[min_disc_idx]

            # delete sampled sample
            sample_set.drop(index=cands.iloc[min_disc_idx].name, inplace=True)

        self.transformed_data = self.dataset.loc[samples.index]
        return self.transformed_data


class MaximalMinDistance(Preprocessing):

    def __init__(self, frac=0.8, m=5, window_size=100,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.m = m
        self.window_size = window_size
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        # first sample
        sample_set = self.dataset.copy()
        samples = self.dataset.sample(n=1, replace=False)
        sample_set.drop(index=samples.index, inplace=True)

        n = int(self.frac * len(self.dataset))
        for i in range(1, n):
            # create candidates
            cands = sample_set.sample(n=min(self.m, len(sample_set)),
                                      replace=False)

            dists = cdist(cands,
                          samples
                          .sample(min(self.window_size, len(samples)), replace=False))
            min_dists = np.min(dists, axis=1)
            argmax_min_dist = np.argmax(min_dists)

            # update indices
            samples = pd.concat((samples, cands.iloc[argmax_min_dist:argmax_min_dist+1]), axis=0)
            sample_set.drop(index=cands.iloc[argmax_min_dist].name, inplace=True)

        self.transformed_data = self.dataset.loc[samples.index]
        return self.transformed_data


class MitchellsSampling(MaximalMinDistance):

    def __init__(self, frac=0.8, m=5,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, m=m, window_size=0,
                         protected_attribute=protected_attribute, label=label, random_state=random_state)


class MinimalMinDistance(Preprocessing):
    """
    Deletes samples with the smallest distance to its nearest neighbor
    """

    def __init__(self, frac=0.8, m=5, window_size=100,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.m = m
        self.window_size = window_size
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self):
        if self.dataset is None:
            raise Exception('Model not fitted.')

        # preparation
        samples = self.dataset.copy()

        n = (len(self.dataset) -
             int(len(self.dataset) * self.frac))
        for i in range(1, n):
            # create candidates
            cands = samples.sample(n=min(self.m, len(samples)), replace=False)

            # calculate distances between candidates and samples
            dists = cdist(cands, samples.sample(n=min(self.window_size, len(samples)), replace=False)
                          .drop(index=cands.index, errors='ignore', inplace=False))
            min_dists = np.min(dists, axis=1)
            argmin_min_dist = np.argmin(min_dists)

            # update samples
            samples.drop(index=cands.index[argmin_min_dist], inplace=True)

        self.transformed_data = self.dataset.loc[samples.index]
        return self.transformed_data


class LeastDiscriminatingSampling(MaximalMinDistance):
    """
    Wrapper class for MaximalMinDistanceWindow
    """

    def __init__(self, frac=0.8, m=5,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, m=m, window_size=100,
                         protected_attribute=protected_attribute, label=label, random_state=random_state)


class MostDiscriminatingSamplesRemover(MinimalMinDistance):
    """
    Wrapper class for MinimalMinDistance
    """

    def __init__(self, frac=0.8, m=5, window_size=100,
                 protected_attribute=None, label=None, random_state=None):
        super().__init__(frac=frac, protected_attribute=protected_attribute, label=label)
        self.m = m
        self.window_size = window_size
        self.random_state = random_state
        np.random.seed(random_state)


def discrepancy(x, method='CD'):
    """Measures the discrepancy of a given sample.
    Assume [0, 1] normalized samples

    Parameters
    ----------
    x: array_like (n, d)
        The sample to compute the discrepancy from.

    method: str, optional
        Type of discrepancy. Default is 'CD' which is Centered L2 discrepancy (Hickernell 1998).

    Returns
    -------
    disc: float
        centered l2 discrepancy
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    n, d = x.shape

    if method=='CD':
        second_sum = 0
        first_sum = np.sum(np.prod(1 + abs(x - 0.5) / 2 - (abs(x - 0.5) ** 2) / 2, axis=1))
        for i in range(n):
            second_sum += np.sum(np.prod(1 + abs(x[i, :] - 0.5) / 2 + abs(x - 0.5) / 2 - abs(x[i, :] - x) / 2, axis=1))
        CL2 = (13/12) ** d - (2 * first_sum - second_sum / n) / n

        return CL2
