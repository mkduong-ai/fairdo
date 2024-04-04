from functools import partial
import numpy as np

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
from fairdo.metrics import statistical_parity_abs_diff_max, statistical_parity_abs_diff_sum, data_loss, group_missing_penalty

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)
n_groups = data[protected_attributes[0]].unique()

pop_size=100
num_generations=100

# Multi Objective
ga = partial(nsga2,
             pop_size=pop_size,
             num_generations=num_generations,
             initialization=variable_initialization,
             selection=elitist_selection_multi,
             crossover=onepoint_crossover,
             mutation=bit_flip_mutation)

# Penalized discrimination
def penalized_discrimination(y, z, agg_group='max', eps=0.01,**kwargs):
    """
    Penalized discrimination function that combines the statistical parity and group missing penalty.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    
    Returns
    -------
    float
        The penalized discrimination."""
    if agg_group=='sum':
        penalized_discrimination = statistical_parity_abs_diff_sum(y=y,
                                                                   z=z) + \
                                   group_missing_penalty(z=z,
                                                         n_groups=n_groups,
                                                         agg_group=agg_group)
    elif agg_group=='max':
        penalized_discrimination = np.max([statistical_parity_abs_diff_max(y=y,
                                                                           z=z),
                                           group_missing_penalty(z=z,
                                                                 n_groups=n_groups,
                                                                 agg_group=agg_group,
                                                                 eps=eps)])/(1+eps)
    else:
        raise ValueError("Invalid aggregation group. Supported values are 'sum' and 'max'.")
    return penalized_discrimination


# Initialize the wrapper class for custom preprocessors
preprocessor_multi = MultiObjectiveWrapper(heuristic=ga,
                                     protected_attribute=protected_attributes[0],
                                     label=label,
                                     fitness_functions=[penalized_discrimination, data_loss])

# Fit and transform the data, returns the data closest to the ideal solution
data_multi = preprocessor_multi.fit_transform(dataset=data)

# Single Objective
def weighted_loss(y, z, n_groups, dims, w=0.5, agg_group='max', eps=0.01, **kwargs):
    """
    A single objective function that combines the statistical parity and data loss.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    dims: int
        The number of samples.
    
    Returns
    -------
    float
        The weighted fairness and quality of the data."""    
    return w * penalized_discrimination(y=y, z=z, n_groups=n_groups, agg_group=agg_group, eps=eps) +\
        (1-w) * data_loss(y=y, dims=dims)

ga = partial(genetic_algorithm,
             pop_size=pop_size,
             num_generations=num_generations,
             initialization=variable_initialization,
             crossover=onepoint_crossover,
             mutation=bit_flip_mutation)

# Initialize the wrapper class for custom preprocessors
preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                fitness_functions=[weighted_loss])

# Fit and transform the data
data_single = preprocessor.fit_transform(dataset=data)

# Print no. samples and discrimination before and after
print("Size (higher is better) and discrimination (lower is better).")
print("Original data:", 0,
      statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy()))
print("Multi-objective fair data:", 1-len(data_multi)/len(data),
        statistical_parity_abs_diff_max(data_multi[label], data_multi[protected_attributes[0]].to_numpy()))
print("Single-objective fair data:", 1-len(data_single)/len(data),
        statistical_parity_abs_diff_max(data_single[label], data_single[protected_attributes[0]].to_numpy()))

# Plot the results
preprocessor_multi.plot_results(x_label='Max. Statistical Parity', y_label='% of samples removed', title='NSGA-II Optimization Results')
