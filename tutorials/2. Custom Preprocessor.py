from functools import partial

# Related third-party imports
from sdv.tabular import GaussianCopula
import pandas as pd

# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import HeuristicWrapper
from fairdo.optimize.geneticalgorithm import genetic_algorithm
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data_orig, label, protected_attributes = load_data('compas', print_info=False)

# Create synthetic data (optional step: useful to enlarge dataset)
gc = GaussianCopula()
gc.fit(data_orig)
data_syn = gc.sample(data_orig.shape[0])

# Merge/concat original and synthetic data
data = pd.concat([data_orig, data_syn.copy()], axis=0)

# Custom settings for the Genetic Algorithm (Selection, Crossover, Mutation are also customizable)
ga = partial(genetic_algorithm,
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
print("Original data:", len(data_orig), statistical_parity_abs_diff_max(data_orig[label], data_orig[protected_attributes[0]].to_numpy()))
print("Merged data (original and synthetic):", len(data), statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy()))
print("Preprocessed merged data:", len(data_fair), statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy()))
