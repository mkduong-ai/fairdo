# standard library
from functools import partial
import itertools
import os

# third party
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pathos.multiprocessing import ProcessPool
# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# fairdo package
from fairdo.utils.dataset import load_data
# everything needed for custom preprocessing
from fairdo.preprocessing import MultiObjectiveWrapper, HeuristicWrapper
from fairdo.optimize.multi import nsga2
from fairdo.optimize.single import genetic_algorithm
from fairdo.optimize.geneticoperators import variable_initialization, random_initialization,\
    elitist_selection, elitist_selection_multi, tournament_selection_multi,\
    uniform_crossover, onepoint_crossover, no_crossover, \
    fractional_flip_mutation, shuffle_mutation,\
    bit_flip_mutation
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max,\
    statistical_parity_abs_diff_sum,\
    data_loss, group_missing_penalty


def penalized_discrimination(y, z, n_groups, agg_group='max', **kwargs):
    """
    Penalized discrimination function that combines the statistical parity and group missing penalty.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    n_groups: int
        The number of groups.
    
    Returns
    -------
    float
        The penalized discrimination."""
    if agg_group=='sum':
        penalized_discrimination = statistical_parity_abs_diff_sum(y=y, z=z) + group_missing_penalty(z=z, n_groups=n_groups, agg_group=agg_group)
    elif agg_group=='max':
        penalized_discrimination = np.max([statistical_parity_abs_diff_max(y=y, z=z), group_missing_penalty(z=z, n_groups=n_groups, agg_group=agg_group)])
    else:
        raise ValueError("Invalid aggregation group. Supported values are 'sum' and 'max'.")
    return penalized_discrimination


def main():
    # number of runs
    n_runs = 10

    # settings
    pop_size = 100
    num_generations = 200

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data_str = 'compas'
    data, label, protected_attributes = load_data(data_str, print_info=False)
    n_groups = len(data[protected_attributes[0]].unique())

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Trial', 'Dataset', 'Label', 'Protected_Attributes', 'N_Groups',
                                       'Initializer', 'Selection', 'Crossover', 'Mutation',
                                       'Hypervolume', 'Pareto_Front', 'Baseline'])

    # Setting up pre-processor
    ga = partial(nsga2,
             pop_size=pop_size,
             num_generations=num_generations,
             initialization=variable_initialization,
             selection=elitist_selection_multi,
             crossover=onepoint_crossover,
             mutation=bit_flip_mutation)

    # Select best data
    # Initialize the wrapper class for custom preprocessors
    preprocessor_multi = MultiObjectiveWrapper(heuristic=ga,
                                     protected_attribute=protected_attributes[0],
                                     label=label,
                                     fitness_functions=[penalized_discrimination, penalized_discrimination])

    data_multi = preprocessor_multi.fit_transform(dataset=data)

    # Initialize classifiers

    # Evaluate classifiers

    # Plot results



if __name__ == '__main__':
    main()
