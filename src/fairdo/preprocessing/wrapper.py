# Standard library imports
from functools import partial
import warnings

# Related third-party imports
import numpy as np
import pandas as pd

# fairdo imports
from fairdo.preprocessing import Preprocessing
from fairdo.optimize.single import genetic_algorithm

# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max, data_loss
from fairdo.metrics.penalty import group_missing_penalty


class MultiWrapper(Preprocessing):
    """
    A preprocessing wrapper class that applies a given multi-objective optimization method to optimize multiple
    given objective functions and outputs the Pareto front of the solutions.
    The solutions are returned as a binary numpy array of shape `(n, d)` where n is the number of solutions and d is the
    number of dimensions.
    The objective functions are defined as a list of functions to be optimized.
    They evaluate properties of the dataset such as the fairness and data quality/data loss.
    The pre-processed dataset is a subset of the original dataset, where the columns are
    selected based on the multi-objective optimization method.

    Attributes
    ----------
    heuristic: callable
        The method that optimizes multiple fitness functions. It takes multiple fitness functions, the
        number of dimensions, and some other parameters.
        It returns solutions in the Pareto front and their corresponding fitness values.
        All fronts can be returned if requested.
        The solution has a shape of `(n, dims)` where `n` is the number of solutions and `dims` is the number of dimensions.
    funcs: callable
        List of objective function to be minimized. Wrapper for user-given `fitness_functions`.
        It is defined within the `fit`
        method.
    dims: int
        The number of dimensions or columns in the dataset. It is defined within the `fit`
        method.
    fitness_functions: list of callable
        The list of objective functions to be minimized. They evaluate properties of the dataset
        such as the fairness and data quality/data loss.
    dataset: pandas DataFrame
        The dataset to be preprocessed. It is defined within the `fit` method.
    """

    def __init__(self,
                 heuristic,
                 protected_attribute,
                 label,
                 fitness_functions=[statistical_parity_abs_diff_max, data_loss],
                 **kwargs):
        """
        Constructs all the necessary attributes for the HeuristicWrapper object.

        Parameters
        ----------
        heuristic: callable
            The method that optimizes multiple fitness functions. It takes multiple fitness functions, the
            number of dimensions, and some other parameters.
            It returns solutions in the Pareto front and their corresponding fitness values.
            All fronts can be returned if requested.
            The solution has a shape of `(n, dims)` where `n` is the number of solutions and `dims` is the number of dimensions.
        protected_attribute: str or List[str]
            The protected attribute in the dataset.
        label: str
            The target variable in the dataset.
        fitness_functions: list of callable
            The list of objective functions to be minimized. They evaluate properties of the dataset
            such as the fairness and data quality/data loss.
        kwargs: dict
            Additional arguments for the heuristic method.
        """
        self.heuristic = heuristic
        self.funcs = None
        self.dims = None
        self.fitness_functions = fitness_functions

        # required by Preprocessing
        self.dataset = None
        self.synthetic_dataset = None
        self.approach = None

        # multi-objective specific
        self.masks = None
        self.fitness_values = None
        self.fitness = None # Best weighted fitness value
        self.index_best = None
        super().__init__(protected_attribute=protected_attribute, label=label)

    def fit(self, dataset, synthetic_dataset=None, approach='remove'):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pandas DataFrame
            The dataset to be preprocessed.
        synthetic_dataset: pandas DataFrame, optional
            The synthetic dataset to be used for the 'add' approach.
            It is required only if the 'add' approach is used.
        approach: str
            The approach to be used for the heuristic method.
            It can be either 'remove' or 'add'.

        Returns
        -------
        self

        Examples
        --------
        >>> from fairdo.utils.dataset import load_data
        >>> from fairdo.preprocessing.wrapper import MultiWrapper
        >>> from fairdo.metrics import statistical_parity_abs_diff_max, data_loss
        >>> from fairdo.optimize.multi import nsga2
        >>> # Load the COMPAS dataset
        >>> data, label, protected_attributes = load_data('compas', print_info=False)
        >>> # Define the heuristic method
        >>> heuristic = nsga2
        >>> fitness_functions = [statistical_parity_abs_diff_max, data_loss]
        >>> wrapper = MultiWrapper(heuristic=heuristic,
        ...                        protected_attribute=protected_attributes,
        ...                        label=label,
        ...                        fitness_functions=fitness_functions)
        >>> wrapper.fit(data) # Fit the model
        """
        super().fit(dataset=dataset)

        if synthetic_dataset is not None:
            self.synthetic_dataset = synthetic_dataset.copy()

        self.approach = approach
        # Number of dimensions
        if approach == 'add':
            self.dims = len(self.synthetic_dataset)
        elif approach == 'remove':
            self.dims = len(self.dataset)

        self.funcs = [partial(f,
                              dataset=self.dataset,
                              label=self.label,
                              protected_attributes=self.protected_attribute,
                              approach=approach,
                              synthetic_dataset=self.synthetic_dataset,
                              fitness_function=fitness_function,
                              penalty=None) for fitness_function in self.fitness_functions]
        
        # Optimization step
        self._apply_heuristic()

        return self
    
    def _apply_heuristic(self):
        """
        Applies the heuristic method to the dataset.

        Returns
        -------
        self.transformed_data: pandas DataFrame
            The dataset to be masked based on the heuristic method.
        masks: np.array of shape (n, dims)
            The binary masks indicating the selected columns.
            Represents the `n` solutions in the Pareto front.
        fitness_values: np.array of shape (n, len(fitness_functions))
            The fitness values of the solutions in the Pareto front.
        """
        masks, fitness_values = self.heuristic(fitness_functions=self.funcs,
                                               d=self.dims)

        # apply the mask to the dataset
        if self.approach == 'add':
            self.transformed_data = pd.concat([self.dataset, self.synthetic_dataset], axis=0)
        elif self.approach == 'remove':
            self.transformed_data = self.dataset
        
        self.masks = masks == 1
        self.fitness_values = fitness_values

        return self.transformed_data, self.masks, self.fitness_values

    def transform(self,
                  ideal_solution=np.array([0, 0]),
                  w=0.5):
        """
        Applies the heuristic method to the dataset and
        returns the best solution in the Pareto front, that is,
        the solution closest to the ideal solution.

        If fitted before, it is possible to return a different dataset
        by changing the `ideal_solution` and `w` parameters.

        Parameters
        ----------
        ideal_solution: np.array, optional (default=[0, 0])
            The ideal solution to be used for the optimization.
            Default is [0, 0].
        w: float or np.array, optional (default=0.5)
            The weight to be used for the weighted fitness value.

        Returns
        -------
        data_best: pandas DataFrame
            The dataset closest to the ideal solution.

        Examples
        --------
        >>> from fairdo.utils.dataset import load_data
        >>> from fairdo.preprocessing.wrapper import MultiWrapper
        >>> from fairdo.metrics import statistical_parity_abs_diff_max, data_loss
        >>> from fairdo.optimize.multi import nsga2
        >>> # Load the COMPAS dataset
        >>> data, label, protected_attributes = load_data('compas', print_info=False)
        >>> # Define the heuristic method
        >>> heuristic = nsga2
        >>> fitness_functions = [statistical_parity_abs_diff_max, data_loss]
        >>> wrapper = MultiWrapper(heuristic=heuristic,
        ...                        protected_attribute=protected_attributes,
        ...                        label=label,
        ...                        fitness_functions=fitness_functions)
        >>> # Fit the model
        >>> wrapper.fit(data)
        >>> # Transform the dataset
        >>> transformed_data = wrapper.transform()
        """
        if self.dataset is None:
            raise Exception('Model not fitted. Run the `fit` method first.')
        
        # Check whether weight is numeric or array
        if isinstance(w, (int, float)):
            w = np.array([w, 1 - w])

        self.index_best = np.argmin(np.linalg.norm(w * self.fitness_values - ideal_solution,
                                                   ord=1,
                                                   axis=1))
        solution_best = self.masks[self.index_best]
        data_best = self.transformed_data[solution_best]
        
        return data_best

    def get_best_fitness(self, return_baseline):
        """
        Get the best fitness value of the solutions according to
        the weights set in the `transform` method.
        
        Parameters
        ----------
        return_baseline: bool
            Whether to return the fitness of the original baseline dataset.

        Returns
        -------
        float
            The best fitness value.
        """
        if self.index_best is None:
            raise ValueError('No results to return. Run the `transform` method first.')
        
        if return_baseline:
            baseline_solution = np.ones(len(self.dataset))
            baseline_fitness = np.array([func(baseline_solution) for func in self.funcs]).reshape(1, -1)
            
            return self.fitness_values[self.index_best], baseline_fitness
        else:
            return self.fitness_values[self.index_best]
    
    def get_pareto_front(self, return_baseline=False):
        """
        Get the Pareto front of the solutions.
        Return the baseline solution if it is requested.

        Parameters
        ----------
        return_baseline: bool, optional (default=False)
            Whether to return the result of the original baseline dataset.
        
        Returns
        -------
        self.fitness_values: np.array of shape (n, len(fitness_functions))
            The Pareto front of the solutions.
        baseline_fitness: np.array of shape (1, len(fitness_functions))
            The fitness values of the baseline solution. Only returned if `return_baseline` is True.
        """
        if self.fitness_values is None:
            raise ValueError('No results to return. Run the `transform` method first.')
        
        if return_baseline:
            baseline_solution = np.ones(len(self.dataset))
            baseline_fitness = np.array([func(baseline_solution) for func in self.funcs]).reshape(1, -1)

            return self.fitness_values, baseline_fitness
        else:
            return self.fitness_values
    
    def plot_results(self,
                     xaxis=0, yaxis=1,
                     xlabel='Fitness 1', ylabel='Fitness 2',
                     title='Multi-Objective Optimization Results',
                     figsize=(7, 7),
                     show=True,
                     savefig=False,
                     filename='pareto_front.png'):
        """
        Plot the results of the multi-objective optimization.

        Parameters
        ----------
        xaxis: int, optional (default=0)
            The index of the fitness function to be plotted on the x-axis.
        yaxis: int, optional (default=1)
            The index of the fitness function to be plotted on the y-axis.
        xlabel: str, optional (default='Fitness 1')
            The label of the x-axis.
        ylabel: str, optional (default='Fitness 2')
            The label of the y-axis.
        title: str, optional (default='Multi-Objective Optimization Results')
            The title of the plot.
        figsize: tuple, optional (default=(7, 7))
            The size of the plot.
        show: bool, optional (default=True)
            Whether to show the plot.
        savefig: bool, optional (default=False)
            Whether to save the plot.
        filename: str, optional (default='pareto_front.png')
        
        Examples
        --------
        >>> from fairdo.utils.dataset import load_data
        >>> from fairdo.preprocessing.wrapper import MultiWrapper
        >>> from fairdo.metrics import statistical_parity_abs_diff_max, data_loss
        >>> from fairdo.optimize.multi import nsga2
        >>> # Load the COMPAS dataset
        >>> data, label, protected_attributes = load_data('compas', print_info=False)
        >>> # Define the heuristic method
        >>> heuristic = nsga2
        >>> fitness_functions = [statistical_parity_abs_diff_max, data_loss]
        >>> wrapper = MultiWrapper(heuristic=heuristic,
        ...                        protected_attribute=protected_attributes,
        ...                        label=label,
        ...                        fitness_functions=fitness_functions)
        >>> # Fit the model
        >>> wrapper.fit(data)
        >>> # Transform the dataset
        >>> transformed_data = wrapper.transform()
        >>> # Plot the results
        >>> wrapper.plot_results()
        """
        if self.fitness_values is None:
            raise ValueError('No results to plot. Run the `transform` method first.')
        
        if self.fitness_values.shape[1] < 2:
            raise ValueError('The number of fitness functions is less than 2.\
                              At least two fitness functions are required to plot the Pareto front.')

        import matplotlib.pyplot as plt

        # Plot the results
        plt.figure(figsize=figsize)
        plt.scatter(self.fitness_values[:, xaxis], self.fitness_values[:, yaxis],
                        label=f'Pareto Front',
                        c='r',
                        s=30)
        base_solution = np.ones(len(self.dataset))
        plt.scatter(self.funcs[xaxis](base_solution), self.funcs[yaxis](base_solution),
                    label=f'Original Dataset',
                    c='b',
                    s=30)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if savefig:
            plt.savefig(filename,
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0)
        
        if show:
            plt.show()


class SingleWrapper(Preprocessing):
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
    dataset: pandas DataFrame
        The dataset to be preprocessed. It is defined within the `fit` method.
    """

    def __init__(self,
                 heuristic,
                 protected_attribute,
                 label,
                 disc_measure=statistical_parity_abs_diff_max,
                 fitness_functions=None,
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
        fitness_functions: list of callable, optional (default=None)
            Can only contain one function. It is a wrapper for the user-given `disc_measure`.
        kwargs: dict
            Additional arguments for the heuristic method.
        """
        self.heuristic = heuristic
        self.func = None
        self.dims = None
        if fitness_functions is not None:
            if len(fitness_functions) > 1:
                warnings.warn('The `fitness_functions` parameter contains more than one function.\
                               Only the first function will be used in this wrapper.')
        self.disc_measure = disc_measure if fitness_functions is None else fitness_functions[0]

        # required by Preprocessing
        self.dataset = None
        self.synthetic_dataset = None
        self.approach = None

        # single-objective specific
        self.mask = None # Mask to select best solution

        super().__init__(protected_attribute=protected_attribute, label=label)

    def fit(self, dataset, synthetic_dataset=None, approach='remove'):
        """
        Defines the discrimination measure function and the number of dimensions based on the
        input dataset.

        Parameters
        ----------
        dataset: pandas DataFrame
            The dataset to be preprocessed.
        synthetic_dataset: pandas DataFrame, optional (default=None)
            The synthetic dataset to be used for the 'add' approach.
            It is required only if the 'add' approach is used.
        approach: str, optional (default='remove')
            The approach to be used for the heuristic method.
            It can be either 'remove' or 'add'.

        Returns
        -------
        self
        """
        super().fit(dataset=dataset)

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
        # penalty = partial(group_missing_penalty,
        #                  n_groups=n_groups)

        self.func = partial(f,
                            dataset=self.dataset,
                            label=self.label,
                            protected_attributes=self.protected_attribute,
                            approach=approach,
                            synthetic_dataset=self.synthetic_dataset,
                            fitness_function=self.disc_measure,
                            penalty=None)
        
        # Optimization step
        self.mask = self.heuristic(f=self.func, d=self.dims)[0] == 1

        return self

    def transform(self):
        """
        Applies the heuristic method to the dataset and returns a preprocessed version of it.

        Returns
        -------
        pandas DataFrame
            The preprocessed (fair) dataset.
        """
        # apply the mask to the dataset
        if self.approach == 'add':
            self.transformed_data = pd.concat([self.dataset, self.synthetic_dataset[self.mask]], axis=0)
        elif self.approach == 'remove':
            self.transformed_data = self.dataset[self.mask]

        return self.transformed_data
    
    def get_best_fitness(self, return_baseline=False):
        """
        Get the best fitness value of the solution.
        
        Parameters
        ----------
        return_baseline: bool, optional (default=False)
            Whether to return the fitness of the original baseline dataset.

        Returns
        -------
        float
            The best fitness value.
        """
        if self.mask is None:
            raise ValueError('No results to return. Run the `transform` method first.')
        
        best_fitness = self.func(self.mask)

        if return_baseline:
            baseline_solution = np.ones(len(self.dataset))
            baseline_fitness = self.func(baseline_solution)
            
            return best_fitness, baseline_fitness
        else:
            return best_fitness


class DefaultPreprocessing(SingleWrapper):
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
    dataset: pandas DataFrame
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
                         disc_measure=disc_measure,
                         **kwargs)


def f(binary_vector, dataset, label, protected_attributes,
      approach='remove',
      synthetic_dataset=None,
      fitness_function=statistical_parity_abs_diff_max,
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
    dataset: pandas DataFrame
        The data to calculate the discrimination measure on.
    label: str
        The column in the dataset to use as the target variable.
    protected_attributes: Union[str, List[str]]
        The column or columns in the dataset to consider as protected attributes.
    approach: str
        The approach to be used for the heuristic method.
        It can be either 'remove' or 'add'.
    synthetic_dataset: pandas DataFrame, optional
        Extra samples to be added to the original data. Samples can be synthetic data.
        It is required only if the 'add' approach is used.
    fitness_function: callable, optional (default=statistical_parity_abs_diff_max)
        A function that takes in x (features), y (labels), and z (protected attributes) and returns a numeric value.
        Default is `statistical_parity_abs_diff_max` which is the absolute difference between the maximum and minimum statistical parity values.
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
        return fitness_function(x=x, y=y, z=z, dims=len(mask)) + penalty(x=x, y=y, z=z)
    else:
        return fitness_function(x=x, y=y, z=z, dims=len(mask))
    