import os
import warnings

import numpy as np
import pandas as pd

# datasets
from aif360.datasets import AdultDataset, BankDataset, CompasDataset, GermanDataset, BinaryLabelDataset
# import folktables

# machine learning
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# fair preprocessors
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing
from src.preprocessing import MetricOptimizer, OriginalData, PreprocessingWrapper

# evaluation
## dataset evaluation
from aif360.metrics import BinaryLabelDatasetMetric
## classification metrics
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, \
    recall_score
## fairness metrics
from src.metrics import mutual_information, normalized_mutual_information,\
    rdc, statistical_parity_absolute_difference, \
    equal_opportunity_absolute_difference, disparate_impact_ratio, disparate_impact_ratio_objective,\
    predictive_equality_absolute_difference, average_odds_error, average_odds_difference,\
    consistency_score, consistency_score_objective


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
            dataset_orig = BankDataset(protected_attribute_names=['age'])
    elif dataset_used == "compas":
        if protected_attribute_used == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]

            label_map = [{1.0: 'Did recid.', 0.0: 'No recid.'}]
            protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            dataset_orig = CompasDataset(protected_attribute_names=['sex'], privileged_classes=[['Male']],
                                         categorical_features=['c_charge_degree', 'age_cat'],
                                         features_to_keep=['c_charge_degree', 'priors_count', 'age_cat'],
                                         metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
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
                                         features_to_drop=['personal_status'],
                                         privileged_classes=[['Old']],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
        elif protected_attribute_used == "foreign_worker":
            # TODO: this part is broken
            privileged_groups = [{'foreign_worker': 0}]
            unprivileged_groups = [{'foreign_worker': 1}]

            label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            protected_attribute_maps = [{1.0: 'foreigner', 0.0: 'native'}]
            dataset_orig = GermanDataset(protected_attribute_names=['foreign_worker'],
                                         features_to_drop=['personal_status'],
                                         privileged_classes=[['foreigner']],
                                         metadata={'label_maps': label_map,
                                                   'protected_attribute_maps': protected_attribute_maps})
    else:
        raise Exception('dataset_used or protected_attribute_used not available.')

    if dataset_orig is None:
        raise Exception('dataset_used or protected_attribute_used not available.')

    return dataset_orig, privileged_groups, unprivileged_groups


def preprocess_dataset(preprocessors, dataset_train):
    """

    Parameters
    ----------
    preprocessors: list of preprocessors
    dataset_train: pandas DataFrame

    Returns
    -------

    """
    return {type(preprocessor).__name__: preprocessor.fit_transform(dataset_train) for preprocessor in
            preprocessors}


def get_BinaryLabelDatasetMetric(dataset_train: pd.DataFrame, preproc_datasets: list,
                                 privileged_groups: list,
                                 unprivileged_groups: list):
    dataset_metric = BinaryLabelDatasetMetric(dataset_train,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)

    preproc_dataset_metrics = {key: BinaryLabelDatasetMetric(preproc_datasets[key],
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups) for key in
                               preproc_datasets}

    return dataset_metric, preproc_dataset_metrics


def evaluate_dataset_metrics(key: str, dataset_metric: BinaryLabelDatasetMetric):
    evaluate_dataset_metric = lambda keyName, blDatasetMetric: pd.DataFrame({
        'Name': keyName,
        'Base Rate': blDatasetMetric.base_rate(),
        'Consistency': blDatasetMetric.consistency(),
        'Statistical Parity Difference': blDatasetMetric.statistical_parity_difference(),
        'Disparate Impact': blDatasetMetric.disparate_impact(),
        'Smoothed Empirical Differential Fairness': blDatasetMetric.smoothed_empirical_differential_fairness()
    })

    return evaluate_dataset_metric(key, dataset_metric)


def train_models(models_trained: dict, Xs_preproc: dict, ys_preproc: dict, ws_preproc: dict):
    """
    The keys of Xs, ys, ws represent preprocessing methods

    Parameters
    ----------
    models_trained: dict
    Xs_preproc: dict of features
    ys_preproc: dict of labels
    ws_preproc: dict of sample weights

    Returns
    -------
    models_trained: dict
    """
    # pipeline = lambda name, model: Pipeline([('scaler', StandardScaler()), (name, model)])

    for key_model, value_model in models_trained.items():
        for key_preproc in value_model:
            model = models_trained[key_model][key_preproc]

            try:
                models_trained[key_model][key_preproc] = \
                    models_trained[key_model][key_preproc].fit(Xs_preproc[key_preproc], ys_preproc[key_preproc],
                                                               sample_weight=ws_preproc[key_preproc])
            except:
                # fix logistic regression for 1 class
                uniqueness = np.unique(ys_preproc[key_preproc])
                if key_model == 'LogisticRegression' and len(uniqueness) == 1:
                    warnings.warn("Fix for LogisticRegression for one available class.")
                    models_trained[key_model][key_preproc] = \
                        DummyClassifier(strategy='constant',
                                        constant=uniqueness[0]).\
                            fit(Xs_preproc[key_preproc], ys_preproc[key_preproc])
                else:
                    models_trained[key_model][key_preproc] = \
                        models_trained[key_model][key_preproc].fit(Xs_preproc[key_preproc], ys_preproc[key_preproc])

    return models_trained


def evaluate_ml_models(results: dict, models_trained: dict, X_test, y_test, z_test):
    """

    Parameters
    ----------
    results: dict
        e.g. stores results in results['LogisticRegression']['DisparateImpactRemover']['Accuracy]
    models_trained: dict
        e.g. models are in models_trained['LogisticRegression']['DisparateImpactRemover']
    X_test: ndarray
    y_test: flattened numpy array
    z_test: flattened numpy array

    Returns
    -------

    """

    # evaluate the models
    for key_model in models_trained:
        for key_preproc in models_trained[key_model]:
            # Classification Metric
            y_pred = \
                models_trained[key_model][key_preproc].predict_proba(X_test)
            y_pred_argmax = np.argmax(y_pred, axis=1)

            results[key_model][key_preproc]['Accuracy'] = \
                accuracy_score(y_test, y_pred_argmax)

            results[key_model][key_preproc]['Precision'] = \
                precision_score(y_test, y_pred_argmax)

            results[key_model][key_preproc]['Recall'] = \
                recall_score(y_test, y_pred_argmax)

            results[key_model][key_preproc]['F1 Score'] = \
                f1_score(y_test, y_pred_argmax)

            results[key_model][key_preproc]['Balanced Accuracy'] = \
                balanced_accuracy_score(y_test, y_pred_argmax)

            if len(y_pred.shape) < 2:
                results[key_model][key_preproc]['AUC'] = \
                    roc_auc_score(y_test, y_pred) # sklearn documentation: greater label
            else:
                if y_pred.shape[1] == 2:
                    results[key_model][key_preproc]['AUC'] = \
                        roc_auc_score(y_test, y_pred[:, 1])  # sklearn documentation: greater label
                else:
                    results[key_model][key_preproc]['AUC'] = \
                        roc_auc_score(y_test, y_pred)

            # Fairness Notion
            # independence
            results[key_model][key_preproc]['Mutual Information'] = \
                mutual_information(y_pred_argmax, z_test)

            results[key_model][key_preproc]['Normalized MI'] = \
                normalized_mutual_information(y_pred_argmax, z_test)

            results[key_model][key_preproc]['Randomized Dependence Coefficient'] = \
                rdc(y_pred_argmax, z_test)

            results[key_model][key_preproc]['Pearson Correlation'] = \
                np.abs(np.corrcoef(y_pred_argmax, z_test)[0, 1])

            # parity based measures
            results[key_model][key_preproc]['Statistical Parity Abs Diff'] = \
                statistical_parity_absolute_difference(y_pred_argmax, z_test)

            results[key_model][key_preproc]['Disparate Impact'] = \
                disparate_impact_ratio(y_pred_argmax, z_test)

            results[key_model][key_preproc]['Disparate Impact Obj'] = \
                disparate_impact_ratio_objective(y_pred_argmax, z_test)

            # prediction
            results[key_model][key_preproc]['Equal Opportunity Abs Diff'] = \
                equal_opportunity_absolute_difference(y_test, y_pred_argmax, z_test)

            results[key_model][key_preproc]['Predictive Equality Abs Diff'] = \
                predictive_equality_absolute_difference(y_test, y_pred_argmax, z_test)

            results[key_model][key_preproc]['Average Odds Diff'] = \
                average_odds_difference(y_test, y_pred_argmax, z_test)

            results[key_model][key_preproc]['Average Odds Error'] = \
                average_odds_error(y_test, y_pred_argmax, z_test)

            # individual fairness
            results[key_model][key_preproc]['Consistency'] = \
                consistency_score(X_test, y_pred_argmax, n_neighbors=5)

            results[key_model][key_preproc]['Consistency Obj'] = \
                consistency_score_objective(X_test, y_pred_argmax, n_neighbors=5)

    return results


def results_to_df(results):
    df_list = []
    for key_model in results:
        df = pd.DataFrame(results[key_model]).transpose()
        df['Model'] = key_model
        df['Preprocessor'] = df.index

        df_list.append(df)

    df_concat = pd.concat(df_list, keys=list(results.keys()))
    df_concat['ModelPreprocessor'] = df_concat.index
    df_concat = df_concat.reset_index()
    df_concat = df_concat.drop(columns=['level_0', 'level_1'])

    return df_concat


def preprocess_pipeline(dataset_train: BinaryLabelDataset, dataset_test: BinaryLabelDataset,
                        privileged_groups: list, unprivileged_groups: list,
                        models: list, preprocessors: list):
    preproc_datasets = {type(preprocessor).__name__: preprocessor.fit_transform(dataset_train) for preprocessor in
                        preprocessors}

    # evaluate fairness in datasets
    dataset_metric, preproc_dataset_metrics = \
        get_BinaryLabelDatasetMetric(dataset_train,
                                     preproc_datasets,
                                     privileged_groups,
                                     unprivileged_groups)

    dataset_evaluation = evaluate_dataset_metrics('Dataset (Original)', dataset_metric)
    all_dataset_evaluation = pd.concat(
        [evaluate_dataset_metrics(key, value) for key, value in preproc_dataset_metrics.items()])
    all_dataset_evaluation = pd.concat([dataset_evaluation, all_dataset_evaluation])

    # evaluate classification fairness
    # prepare training dataset
    Xs_preproc = {key: value.features for key, value in preproc_datasets.items()}
    ys_preproc = {key: value.labels.ravel() for key, value in preproc_datasets.items()}
    ws_preproc = {key: value.instance_weights.ravel() for key, value in preproc_datasets.items()}

    # initialize ML models
    models_trained = {type(model).__name__:
                          {type(preprocessor).__name__: clone(model) for preprocessor in preprocessors} for model in
                      models}

    # train models
    models_trained = train_models(models_trained, Xs_preproc, ys_preproc, ws_preproc)

    # evaluate classification
    X_test, y_test = dataset_test.features, dataset_test.labels.ravel()
    z_test = dataset_test.protected_attributes.ravel()  # subset of X_test

    # create results dictionary
    results = {type(model).__name__:
                   {type(preprocessor).__name__: {} for preprocessor in preprocessors} for model in models}

    results = evaluate_ml_models(results, models_trained, X_test, y_test, z_test)
    df_results = results_to_df(results)

    return all_dataset_evaluation, df_results


def run_experiments(models, dataset="compas", protected_attribute="race", preprocessors=None,
                    n_runs=5, seed=1,
                    filepath_clfresults='classification_results.csv'):
    """
    models:
    preprocessors:
    n_run: number of runs
        Calmon et al. 2017 proposed 5-fold cross validation

    Returns
    -------
    results: pandas DataFrame
    """
    np.random.seed(seed)

    # monte carlo cross validation
    dataset_orig, privileged_groups, unprivileged_groups = selecting_dataset(dataset, protected_attribute)
    splits = [dataset_orig.split([0.8], shuffle=True) for _ in range(n_runs)]

    # declare and init preprocs
    for i in range(len(preprocessors)):
        preprocessors[i] = eval(preprocessors[i])

    # pass splits into preprocessing method
    dataset_results_list = [None for _ in range(n_runs)]
    results_list = [None for _ in range(n_runs)]
    for i in range(n_runs):
        print(f"Run: {i}")
        dataset_results_list[i], results_list[i] = preprocess_pipeline(splits[i][0], splits[i][1],
                                                                       privileged_groups,
                                                                       unprivileged_groups,
                                                                       models, preprocessors)

    # merge results
    if not os.path.exists(f"results/{dataset}"):
        os.makedirs(f"results/{dataset}")

    all_results = pd.concat(results_list, axis=0)
    path = f"results/{dataset}/{protected_attribute}_{filepath_clfresults}"
    all_results.to_csv(path)
    print(f"{path} saved")

    all_dataset_results = pd.concat(dataset_results_list, axis=0)
    path = f"results/{dataset}/{protected_attribute}_dataset_{filepath_clfresults}"
    all_dataset_results.to_csv(path)
    print(f"{path} saved")


def run_all_experimental_settings():
    """
    dataset_pro_attributes = [('adult', 'sex'),
                              ('compas', 'race'),
                              ('german', 'foreign_worker'),
                              ('german', 'sex'),
                              ('bank', 'age')]

    models = [LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              SVC(probability=True),
              MLPClassifier(), # MLP does not support sample weight
              KNeighborsClassifier()] # KNN does not support sample weight
    """
    seed = 1
    n_runs = 10

    dataset_pro_attributes = [('adult', 'sex'),
                              ('compas', 'race'),
                              #('german', 'foreign_worker'),
                              ('bank', 'age')]
    dataset_pro_attributes = [('bank', 'age')]

    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]


    # Optimized Preproc. requires distortion functions
    preprocessors = ["OriginalData()",
                     "DisparateImpactRemover(sensitive_attribute=protected_attribute)",
                     "LFR(unprivileged_groups=unprivileged_groups,"
                         "privileged_groups=privileged_groups,"
                         "k=5, Ax=0.01, Ay=1.0, Az=10.0)",
                     "PreprocessingWrapper(MetricOptimizer(frac=0.75,"
                                                          "m=5,"
                                                          "fairness_metric=statistical_parity_absolute_difference,"
                                                          "protected_attribute=protected_attribute,"
                                                          "label=dataset_orig.label_names[0]))",
                     "Reweighing(unprivileged_groups=unprivileged_groups,"
                                "privileged_groups=privileged_groups)"]

    for dataset, protected_attribute in dataset_pro_attributes:
        print(f"{dataset} ({protected_attribute})")
        run_experiments(models=models,
                        dataset=dataset,
                        protected_attribute=protected_attribute,
                        preprocessors=preprocessors,
                        n_runs=n_runs,
                        seed=seed)


def run_fast():
    # settings
    dataset = "compas"  # "adult", "german", "compas", "bank"
    protected_attribute = "race"  # sex, age, race
    n_runs = 2
    seed = 1

    # declare machine learning models
    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    preprocessors = ["OriginalData()",
                     "DisparateImpactRemover(sensitive_attribute=protected_attribute)",
                     "LFR(unprivileged_groups=unprivileged_groups,"
                         "privileged_groups=privileged_groups,"
                         "k=5, Ax=0.01, Ay=1.0, Az=10.0)",
                     "PreprocessingWrapper(MetricOptimizer(frac=0.75,"
                                                          "m=5,"
                                                          "fairness_metric=statistical_parity_absolute_difference,"
                                                          "protected_attribute=protected_attribute,"
                                                          "label=dataset_orig.label_names[0]))",
                     "Reweighing(unprivileged_groups=unprivileged_groups,"
                     "privileged_groups=privileged_groups)"]

    run_experiments(models=models,
                    dataset=dataset,
                    protected_attribute=protected_attribute,
                    preprocessors=preprocessors,
                    n_runs=n_runs,
                    seed=seed)


def main():
    fast_run = True

    if fast_run:
        run_fast()
    else:
        run_all_experimental_settings()


if __name__ == '__main__':
    main()
