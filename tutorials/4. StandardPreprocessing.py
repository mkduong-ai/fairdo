from functools import partial

# Related third-party imports
# from sdv.tabular import GaussianCopula
import pandas as pd

# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
#from fairdo.optimize.geneticalgorithm import genetic_algorithm
from fairdo.metrics import statistical_parity_abs_diff_max
from fairdo.metrics import group_missing_penalty

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data_orig, label, protected_attributes = load_data('compas', print_info=False)

# Create synthetic data
# gc = GaussianCopula()
# gc.fit(data_orig)
# data_syn = gc.sample(data_orig.shape[0])

# Merge/concat original and synthetic data
# data = pd.concat([data_orig, data_syn.copy()], axis=0)
data = data_orig.copy()

# Optimization step
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label,
                                    disc=statistical_parity_abs_diff_max)
                                
# Add penalty when removing groups completely
penalty_kwargs = {'n_groups': len(data[protected_attributes[0]].unique())}
data_fair = preprocessor.fit_transform(dataset=data,
                                       approach='remove',
                                       penalty=group_missing_penalty,
                                       penalty_kwargs=penalty_kwargs)

# Print no. samples
print(data_orig.shape, statistical_parity_abs_diff_max(data_orig[label], data_orig[protected_attributes[0]].to_numpy()))
print(data.shape, statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy()))
print(data_fair.shape, statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy()))
