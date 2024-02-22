from functools import partial

# fairdo package
from fairdo.utils.dataset import load_data
# everything needed for custom preprocessing
from fairdo.preprocessing import HeuristicWrapper
from fairdo.optimize.geneticalgorithm import genetic_algorithm
from fairdo.optimize.geneticoperators import onepoint_crossover, fractional_flip_mutation, elitist_selection
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)

# Custom settings for the Genetic Algorithm
ga = partial(genetic_algorithm,
             selection=elitist_selection,
             crossover=onepoint_crossover,
             mutation=fractional_flip_mutation,
             pop_size=50,
             num_generations=100)

# Initialize the wrapper class for custom preprocessors
preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                disc_measure=statistical_parity_abs_diff_max)
                                
# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data)

# Print no. samples and discrimination before and after
print("Size and discrimination (lower is better).")
print("Merged data (original and synthetic):", len(data),
      statistical_parity_abs_diff_max(data[label],
                                      data[protected_attributes[0]].to_numpy()))
print("Preprocessed merged data:", len(data_fair),
      statistical_parity_abs_diff_max(data_fair[label],
                                      data_fair[protected_attributes[0]].to_numpy()))
