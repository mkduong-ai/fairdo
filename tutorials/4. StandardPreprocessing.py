# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas', print_info=False)

# Initialize DefaultPreprocessing object
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label,
                                    disc_measure=statistical_parity_abs_diff_max)

# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data,
                                       approach='remove')

# Print no. samples and discrimination
disc_before = statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy())
disc_after = statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy())
print(data.shape, disc_before)
print(data_fair.shape, disc_after)
print("By removing {0:.2f}% of the samples, discrimination was reduced by {1:.2f}%.".format((1 - (data_fair.shape[0] / data.shape[0])) * 100,
    (disc_before - disc_after) / disc_before * 100))
