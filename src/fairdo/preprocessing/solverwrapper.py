# Standard library imports
from functools import partial

# Related third-party imports
import numpy as np
import pandas as pd

# fairdo imports
from fairdo.preprocessing import Preprocessing
from fairdo.optimize import genetic_algorithm

# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max
from fairdo.metrics.penalty import group_missing_penalty


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
    """

    def __init__(self,
                 heuristic,
                 protected_attribute,
                 label,
                 disc_measure=statistical_parity_abs_diff_max,
                 **kwargs):
        """
        Constructs all the necessary attributes for the HeuristicWrapper object.

        Parameters
        ----------
        heuristic: callable
            The method that optimizes the discrimination measure.
        protected_attribute: str or List[str]
            The protected attribute in the dataset.
        label: str
            The target variable in the dataset.
        disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
            The discrimination measure to be optimized.
            Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and
            minimum statistical parity values.
        kwargs: dict
            Additional arguments for the heuristic method.
        """
        self.heuristic = heuristic
        self.func = None
        self.dims = None
        self.disc_measure = disc_measure

        # required by Preprocessing
        self.dataset = None
        self.synthetic_dataset = None
        self.approach = None
        super().__init__(protected_attribute=protected_attribute, label=label)

    def fit(self, dataset, synthetic_dataset=None, approach='remove'):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to be preprocessed.
        synthetic_dataset: pd.DataFrame, optional
            The synthetic dataset to be used for the 'add' approach.
            It is required only if the 'add' approach is used.
        approach: str
            The approach to be used for the heuristic method.
            It can be either 'remove' or 'add'.

        Returns
        -------
        self
        """
        self.dataset = dataset.copy()
        if synthetic_dataset is not None:
            self.synthetic_dataset = synthetic_dataset.copy()

        self.approach = approach
        # Number of dimensions
        if approach == 'add':
            self.dims = len(self.synthetic_dataset)
        elif approach == 'remove':
            self.dims = len(self.dataset)
        
        # get unique values for each protected attribute
        if isinstance(self.protected_attribute, list):
            n_groups = np.array([self.dataset[attr].nunique() for attr in self.protected_attribute])
        else:
            n_groups = np.array([self.dataset[self.protected_attribute].nunique()])

        # define penalty function
        penalty = partial(group_missing_penalty,
                          n_groups=n_groups)

        self.func = partial(f,
                            dataset=self.dataset,
                            label=self.label,
                            protected_attributes=self.protected_attribute,
                            approach=approach,
                            synthetic_dataset=self.synthetic_dataset,
                            disc_measure=self.disc_measure,
                            penalty=penalty)

        return self

    def transform(self):
        """
        Applies the heuristic method to the dataset and returns a preprocessed version of it.

        Returns
        -------
        pd.DataFrame
            The preprocessed (fair) dataset.
        """
        mask = self.heuristic(f=self.func, d=self.dims)[0] == 1

        # apply the mask to the dataset
        if self.approach == 'add':
            self.transformed_data = pd.concat([self.dataset, self.synthetic_dataset[mask]], axis=0)
        elif self.approach == 'remove':
            self.transformed_data = self.dataset[mask]

        return self.transformed_data


class DefaultPreprocessing(HeuristicWrapper):
    """
    DefaultPreprocessing is a processing method that can be used on-the-go.
    It uses a Genetic Algorithm to select a subset of the given dataset to optimize for fairness.
    It also includes a penalty for missing groups in the protected attribute.

    The default parameters are:
        pop_size=100, num_generations=500.
        Selection: Elitist
        Crossover: Uniform
        Mutation: Fractional Bit Flip

    Attributes
    ----------
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
    """

    def __init__(self,
                 protected_attribute,
                 label,
                 disc_measure=statistical_parity_abs_diff_max,
                 pop_size=100,
                 num_generations=500,
                 **kwargs):
        """
        Constructs all the necessary attributes for the HeuristicWrapper object.

        Parameters
        ----------
        protected_attribute: str or List[str]
            The protected attribute in the dataset.
        label: str
            The target variable in the dataset.
        disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
            The discrimination measure to be optimized.
            Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and
            minimum statistical parity values.
        pop_size: int, optional (default=100)
            The population size for the genetic algorithm.
        num_generations: int, optional (default=500)
            The number of generations for the genetic algorithm.
        kwargs: dict
            Additional arguments for the heuristic method.
        """
        # set default heuristic method
        heuristic = partial(genetic_algorithm,
                            pop_size=pop_size,
                            num_generations=num_generations)
        super().__init__(heuristic=heuristic,
                         protected_attribute=protected_attribute,
                         label=label,
                         disc_measure=disc_measure)


def f(binary_vector, dataset, label, protected_attributes,
      approach='remove',
      synthetic_dataset=None,
      disc_measure=statistical_parity_abs_diff_max,
      penalty=None):
    """
    Two different approaches can be used for the heuristic method:
    1. 'remove': The data points from the given `dataset` are removed to promote fairness.
    2. 'add': Additional samples are added to the original data to promote fairness.
    The sample data can be synthetic data.
    Approach addresses this question: Which of the data points from the `synthetic_dataframe` should be added to the
    original data to prevent discrimination?

    Parameters
    ----------
    binary_vector: np.array
        Binary vector indicating which columns to include in the discrimination measure calculation.
    dataset: pd.DataFrame
        The data to calculate the discrimination measure on.
    label: str
        The column in the dataset to use as the target variable.
    protected_attributes: Union[str, List[str]]
        The column or columns in the dataset to consider as protected attributes.
    approach: str
        The approach to be used for the heuristic method.
        It can be either 'remove' or 'add'.
    synthetic_dataset: pd.DataFrame, optional
        Extra samples to be added to the original data. Samples can be synthetic data.
        It is required only if the 'add' approach is used.
    disc_measure: callable, optional (default=statistical_parity_abs_diff_max)
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.
        Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and minimum
        statistical parity values.
    penalty: callable, optional (default=None)
        A function that takes a dictionary of keyword arguments and returns a numeric value.
        This function is used to penalize the discrimination loss.
        Default is None which means no penalty is applied.

    Returns
    -------
    float
        The calculated discrimination measure.
    """
    if isinstance(protected_attributes, str):
        protected_attributes = [protected_attributes]

    # Create mask
    mask = np.array(binary_vector) == 1

    if approach == 'add' and synthetic_dataset is not None:
        # mask on sample data
        synthetic_dataset = synthetic_dataset[mask]

        # concatenate synthetic data with original data
        dataset = pd.concat([dataset, synthetic_dataset], axis=0)
    elif approach == 'remove':
        # only keep the columns that are selected by the heuristic
        dataset = dataset[mask]
    else:
        raise ValueError('Invalid approach. It can be either \'remove\' or \'add\'.')

    # evaluate on masked dataset
    y = dataset[label]
    z = dataset[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataset.drop(columns=cols_to_drop)

    # We handle multiple protected attributes by not flattening the z array
    y = y.to_numpy().flatten()
    z = z.to_numpy()
    if len(protected_attributes) == 1:
        z = z.flatten()

    if penalty is not None:
        return disc_measure(x=x, y=y, z=z, dims=len(mask)) + penalty(x=x, y=y, z=z)
    else:
        return disc_measure(x=x, y=y, z=z)