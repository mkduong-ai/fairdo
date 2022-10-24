from main import selecting_dataset

import numpy as np
import time

# generate synthetic datapoints
from sdv.tabular import CTGAN
from sdv.lite import TabularPreset

def fix_german_dataset():
    pass


def main():
    dataset, privileged_groups, unprivileged_groups = selecting_dataset('adult', 'sex')
    dataset_df, dataset_dict = dataset.convert_to_dataframe()

    # start = time.perf_counter()
    # ctgan = CTGAN(epochs=10)
    # ctgan.fit(dataset_df)
    # elapsed = time.perf_counter() - start
    # print(elapsed)

    generator = TabularPreset(name='FAST_ML')

    # Create synthetic data
    synthetic_data = generator.sample(100)
    print(type(synthetic_data))


if __name__ == "__main__":
    main()
