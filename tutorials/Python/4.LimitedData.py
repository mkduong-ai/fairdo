# Related third-party imports
import pandas as pd

# fairdo package
from fairdo.utils.dataset import load_data, generate_data
from fairdo.preprocessing import DefaultPreprocessing
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataset
data_orig, label, protected_attributes = load_data('compas', print_info=False)

# Create synthetic data (optional step: useful to enlarge dataset)
data_syn = generate_data(data_orig, num_rows=1000)

# Merge/concat original and synthetic data
data = pd.concat([data_orig, data_syn.copy()], axis=0)

# Initialize DefaultPreprocessing object
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label)

# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data)

# Print no. samples and discrimination before and after
print("Size and discrimination (lower is better).")
print("Original data:", len(data_orig),
      statistical_parity_abs_diff_max(data_orig[label], data_orig[protected_attributes[0]].to_numpy()))
print("Merged data (original and synthetic):", len(data),
      statistical_parity_abs_diff_max(data[label], data[protected_attributes[0]].to_numpy()))
print("Preprocessed merged data:", len(data_fair),
      statistical_parity_abs_diff_max(data_fair[label], data_fair[protected_attributes[0]].to_numpy()))
