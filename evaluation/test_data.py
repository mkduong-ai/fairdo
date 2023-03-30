from pipeline.dataset import selecting_dataset


def main():
    dataset_orig, privileged_groups, unprivileged_groups = selecting_dataset('compas', 'race')
    print(dataset_orig.convert_to_dataframe()[0])
    print(dataset_orig.convert_to_dataframe()[0].columns)
    print(len(dataset_orig.convert_to_dataframe()[0].columns))
    print(privileged_groups)
    print(unprivileged_groups)


if __name__ == '__main__':
    main()
