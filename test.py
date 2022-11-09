from evaluation.pipeline.dataset import selecting_dataset

# generate synthetic datapoints
from sdv.tabular import GaussianCopula


def fix_german_dataset():
    pass


def main():
    dataset, privileged_groups, unprivileged_groups = selecting_dataset('german', 'foreign_worker')
    dataset_df, dataset_dict = dataset.convert_to_dataframe()

    #generator = GaussianCopula()
    #generator.fit(dataset_df)

    # Create synthetic data
    #synthetic_data = generator.sample(100)
    #print(synthetic_data.head())
    print(dataset_df.sample(20))
    # print(dataset_dict)


if __name__ == "__main__":
    main()
