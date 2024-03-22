from pipeline.dataset import selecting_dataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing


def main():
    dataset_orig, privileged_groups, unprivileged_groups = selecting_dataset('compas', 'race')
    preproc = LFR(unprivileged_groups=unprivileged_groups,
                  privileged_groups=privileged_groups,
                  k=5, Ax=0.01, Ay=1.0, Az=1.0)
    print('preprocessing done')
    fair_data = preproc.fit_transform(dataset_orig)
    print(fair_data.convert_to_dataframe()[0])
    print(fair_data.convert_to_dataframe()[0]['two_year_recid'].value_counts())


if __name__ == '__main__':
    main()
