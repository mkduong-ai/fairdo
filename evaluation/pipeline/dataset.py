# datasets
from aif360.datasets import AdultDataset, BankDataset, CompasDataset, GermanDataset, BinaryLabelDataset
# import folktables


def selecting_dataset(dataset_used: str, protected_attribute_used: str):
    """
    Adult Dataset (Calmon et al. 2017 setting):
        X: Age, Education
        Y: Income
        Z: Gender/Race

    COMPAS (Calmon et al. 2017 setting):
        X: severity of charges, number of prior crimes, age category
        Y: binary indicator whether the individual recidivated
        Z: race

    Parameters
    ----------
    dataset_used: str
    protected_attribute_used: str

    Returns
    -------
    dataset_orig: pandas DataFrame
    privileged_groups: list of dict
    unprivileged_groups: list of dict

    """
    dataset_orig = None
    if protected_attribute_used is None:
        if dataset_used == "adult":
            # Calmon et al. 2017
            protected_attribute = "sex"
        if dataset_used == "compas":
            # Calmon et al. 2017
            protected_attribute = "race"
        if dataset_used == "german":
            # Mary et al. 2019
            # protected_attribute = "age"
            protected_attribute = "foreign_worker"
        if dataset_used == "bank":
            protected_attribute = "age"

    if dataset_used == "adult":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]

            dataset_orig = AdultDataset(protected_attribute_names=['sex'], privileged_classes=[['Male']],
                                        categorical_features=[],
                                        features_to_keep=['age', 'education-num'])
        elif protected_attribute_used == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]

            dataset_orig = AdultDataset(protected_attribute_names=['race'], privileged_classes=[['White']],
                                        categorical_features=[],
                                        features_to_keep=['age', 'education-num'])
    elif dataset_used == "bank":
        if protected_attribute_used == "age":
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]

            dataset_orig = BankDataset(protected_attribute_names=['age'])#, privileged_classes=privileged_groups)
    elif dataset_used == "compas":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]

            label_map = [{1.0: 'Did recid.', 0.0: 'No recid.'}]
            protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            dataset_orig = CompasDataset(protected_attribute_names=['sex'], privileged_classes=[['Male']],
                                         categorical_features=['c_charge_degree', 'age_cat'],
                                         features_to_keep=['c_charge_degree', 'priors_count', 'age_cat'],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
        elif protected_attribute_used == "race":
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]

            label_map = [{1.0: 'Did recid.', 0.0: 'No recid.'}]
            protected_attribute_maps = [{1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
            dataset_orig = CompasDataset(protected_attribute_names=['race'], privileged_classes=[['Caucasian']],
                                         categorical_features=['c_charge_degree', 'age_cat'],
                                         features_to_keep=['c_charge_degree', 'priors_count', 'age_cat'],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
    elif dataset_used == "german":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]

            label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            dataset_orig = GermanDataset(protected_attribute_names=['sex'],
                                         features_to_drop=['personal_status'],
                                         privileged_classes=[['Male']],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
        elif protected_attribute_used == "age":
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]

            label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            protected_attribute_maps = [{1.0: 'Old', 0.0: 'Young'}]
            dataset_orig = GermanDataset(protected_attribute_names=['age'],
                                         categorical_features=['status', 'credit_history', 'purpose',
                                                               'savings', 'employment', 'other_debtors', 'property',
                                                               'installment_plans', 'housing', 'skill_level',
                                                               'telephone', 'sex', 'foreign_worker'],
                                         features_to_drop=['personal_status'],
                                         privileged_classes=[lambda x: x > 25],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
        elif protected_attribute_used == "foreign_worker":
            privileged_groups = [{'foreign_worker': 1}]
            unprivileged_groups = [{'foreign_worker': 0}]

            label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            protected_attribute_maps = [{1.0: 'native', 0.0: 'foreigner'}]
            dataset_orig = GermanDataset(protected_attribute_names=['foreign_worker'],
                                         categorical_features=['status', 'credit_history', 'purpose',
                                                               'savings', 'employment', 'other_debtors', 'property',
                                                               'installment_plans', 'housing', 'skill_level',
                                                               'telephone', 'sex'],
                                         features_to_drop=['personal_status'],
                                         privileged_classes=[lambda x: x == 'A202'],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
    else:
        raise Exception('dataset_used or protected_attribute_used not available.')

    if dataset_orig is None:
        raise Exception('dataset_used or protected_attribute_used not available.')

    # print(privileged_groups)
    return dataset_orig, privileged_groups, unprivileged_groups
