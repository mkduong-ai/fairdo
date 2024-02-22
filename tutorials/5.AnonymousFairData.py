# Related third-party imports
from fairdo.utils.helper import generate_data

# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataset
data_orig, label, protected_attributes = load_data('compas', print_info=False)

# Create synthetic data (optional step: useful to enlarge dataset)
data_syn = generate_data(data_orig, num_rows=len(data_orig))

# Initialize DefaultPreprocessing object
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label)

# Fit and transform the synthetic data
data_fair = preprocessor.fit_transform(dataset=data_syn)

# Print no. samples and discrimination of the original data and pre-processed synthetic data
print("Size and discrimination (lower is better).")
print("Original data:", len(data_orig),
      statistical_parity_abs_diff_max(data_orig[label], data_orig[protected_attributes[0]].to_numpy()))
print("Anonymous, fair synthetic data:", len(data_fair),
      statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy()))
