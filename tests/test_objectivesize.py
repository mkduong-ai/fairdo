from functools import partial

# fairdo package
from fairdo.utils.dataset import load_data
# everything needed for custom preprocessing
from fairdo.preprocessing import HeuristicWrapper
from fairdo.optimize.single import genetic_algorithm
from fairdo.optimize.geneticoperators import uniform_crossover, onepoint_crossover, shuffle_mutation, adaptive_mutation, bit_flip_mutation, fractional_flip_mutation, elitist_selection, variable_probability_initialization
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max, data_size_measure

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)

# Custom settings for the First Genetic Algorithm
ga = partial(genetic_algorithm,
             selection=elitist_selection,
             crossover=uniform_crossover,
             mutation=fractional_flip_mutation,
             pop_size=100,
             num_generations=100)

# Initialize the wrapper class for custom preprocessors
preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                disc_measure=data_size_measure)
                                
# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data)

# Custom settings for the First Genetic Algorithm
initialization = partial(variable_probability_initialization,
                         initial_probability=0.95,
                         min_probability=0.5)

mutation = partial(shuffle_mutation,
                   mutation_rate=0.01)
ga = partial(genetic_algorithm,
             initialization=initialization,
             selection=elitist_selection,
             crossover=uniform_crossover,
             mutation=mutation,
             pop_size=100,
             num_generations=100)

# Initialize the wrapper class for custom preprocessors
preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                disc_measure=data_size_measure)
                                
# Fit and transform the data
data_fair2 = preprocessor.fit_transform(dataset=data)


# Print no. samples and discrimination before and after
print("Size and discrimination (lower is better).")
print("Original Data:", len(data)/len(data),
      statistical_parity_abs_diff_max(data[label],
                                      data[protected_attributes[0]].to_numpy()))
print("Fair Data No. 1:", len(data_fair)/len(data),
      statistical_parity_abs_diff_max(data_fair[label],
                                      data_fair[protected_attributes[0]].to_numpy()))
print("Fair Data No. 2:", len(data_fair2)/len(data),
      statistical_parity_abs_diff_max(data_fair2[label],
                                      data_fair2[protected_attributes[0]].to_numpy()))
