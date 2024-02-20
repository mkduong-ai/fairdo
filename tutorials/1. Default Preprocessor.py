# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample dataset with all required information
# data is a pandas.DataFrame
data, label, protected_attributes = load_data('compas', print_info=False)

# Initialize DefaultPreprocessing object
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label)

# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data)

# Print no. samples and discrimination
disc_before = statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy())
disc_after = statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy())
print(len(data), disc_before)
print(len(data_fair), disc_after)
print("By removing {0:.2f}% of the samples, discrimination was reduced by {1:.2f}%.".format((1 - (data_fair.shape[0] / data.shape[0])) * 100,
    (disc_before - disc_after) / disc_before * 100))
print("By default, there is a mechanism to prevent removing a group completely.")