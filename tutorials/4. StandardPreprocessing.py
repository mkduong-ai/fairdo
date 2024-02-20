# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max
from fairdo.metrics import group_missing_penalty

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)

# Optimization step
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label,
                                    disc_measure=statistical_parity_abs_diff_max)
                                
# Add penalty when removing groups completely
penalty_kwargs = {'n_groups': len(data[protected_attributes[0]].unique())}
data_fair = preprocessor.fit_transform(dataset=data,
                                       approach='remove',
                                       penalty=group_missing_penalty,
                                       penalty_kwargs=penalty_kwargs)

# Print no. samples
print(data.shape, statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy()))
print(data_fair.shape, statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy()))
