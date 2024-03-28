from functools import partial
import numpy as np

# fairdo package
from fairdo.utils.dataset import load_data
# everything needed for custom preprocessing
from fairdo.preprocessing import MultiObjectiveWrapper
from fairdo.optimize.multi import nsga2
from fairdo.optimize.geneticoperators import variable_initialization, random_initialization,\
    onepoint_crossover, fractional_flip_mutation, elitist_selection, shuffle_mutation
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max, data_loss, group_missing_penalty

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)

# Custom settings for the Genetic Algorithm
ga = partial(nsga2,
             pop_size=100,
             num_generations=50,
             mutation=fractional_flip_mutation,)

# Initialize the wrapper class for custom preprocessors
preprocessor = MultiObjectiveWrapper(heuristic=ga,
                                     protected_attribute=protected_attributes[0],
                                     label=label,
                                     fitness_functions=[statistical_parity_abs_diff_max, data_loss])

# Fit and transform the data, returns the data closest to the ideal solution
data_best = preprocessor.fit_transform(dataset=data)

# Plot the results
preprocessor.plot_results(x_label='Max. Statistical Parity', y_label='% of samples removed', title='NSGA-II Optimization Results')
