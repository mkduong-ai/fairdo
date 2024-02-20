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
from fairdo.metrics import group_missing_penalty


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
        super().__init__(protected_attribute=protected_attribute, label=label)

    def fit(self, dataset, sample_dataset=None, approach='remove'):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to be preprocessed.
        sample_dataset: pd.DataFrame, optional
            The sample dataset to be used for the 'add' approach.
            It is required only if the 'add' approach is used.
        approach: str
            The approach to be used for the heuristic method.
            It can be either 'remove' or 'add'.

        Returns
        -------
        self
        """
        self.dataset = dataset.copy()
        self.dims = len(self.dataset)
        # define penalty function
        penalty = partial(group_missing_penalty, n_groups=len(dataset[self.protected_attribute].unique()))

        self.func = partial(f,
                            dataframe=self.dataset,
                            label=self.label,
                            protected_attributes=self.protected_attribute,
                            approach=approach,
                            sample_dataframe=sample_dataset,
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


def f(binary_vector, dataframe, label, protected_attributes,
      approach='remove',
      sample_dataframe=None,
      disc_measure=statistical_parity_abs_diff_max,
      penalty=None):
    """
    Additional sample data points are added to the original data to promote fairness.
    The sample data can be synthetic data.
    The question here is: Which of the data points from the sample set should be added to the
    original data to prevent discrimination?

    Parameters
    ----------
    binary_vector: np.array
        Binary vector indicating which columns to include in the discrimination measure calculation.
    dataframe: pd.DataFrame
        The data to calculate the discrimination measure on.
    label: str
        The column in the dataframe to use as the target variable.
    protected_attributes: Union[str, List[str]]
        The column or columns in the dataframe to consider as protected attributes.
    approach: str
        The approach to be used for the heuristic method.
        It can be either 'remove' or 'add'.
    sample_dataframe: pd.DataFrame, optional
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
    penalty_kwargs: dict, optional (default=None)
        A dictionary of keyword arguments to be passed to the penalty function.

    Returns
    -------
    float
        The calculated discrimination measure.
    """
    if isinstance(protected_attributes, str):
        protected_attributes = [protected_attributes]

    # Create mask
    mask = np.array(binary_vector) == 1

    if approach=='add' and sample_dataframe is not None:
        # mask on sample data
        sample_dataframe = sample_dataframe[mask]

        # concatenate synthetic data with original data
        dataframe = pd.concat([dataframe, sample_dataframe], axis=0)
    elif approach=='remove':
        # only keep the columns that are selected by the heuristic
        dataframe = dataframe[mask]
    else:
        raise ValueError('Invalid approach. It can be either \'remove\' or \'add\'.')

    # evaluate on masked dataset
    y = dataframe[label]
    z = dataframe[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataframe.drop(columns=cols_to_drop)

    # We handle multiple protected attributes by not flattening the z array
    y = y.to_numpy().flatten()
    z = z.to_numpy()
    if len(protected_attributes) == 1:
        z = z.flatten()

    return disc_measure(x=x, y=y, z=z) + penalty(x=x, y=y, z=z) if penalty is not None else disc_measure(x=x, y=y, z=z)
