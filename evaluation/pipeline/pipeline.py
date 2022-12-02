from .dataset import selecting_dataset
from .helper import PreprocessingWrapper
from .metric import evaluate_ml_models, evaluate_dataset_metrics

# fado
from fado.preprocessing import MetricOptimizer, OriginalData
from fado.metrics import statistical_parity_absolute_difference, normalized_mutual_information, \
    consistency_score_objective, disparate_impact_ratio_objective

# intern
import os
import warnings

# third party
import numpy as np
import pandas as pd

# datasets
from aif360.datasets import BinaryLabelDataset
# import folktables

# machine learning
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

# aif360
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing


def preprocess_dataset(preprocessors, dataset_train):
    """

    Parameters
    ----------
    preprocessors: list of preprocessors
    dataset_train: pd.DataFrame

    Returns
    -------

    """
    return {type(preprocessor).__name__: preprocessor.fit_transform(dataset_train) for preprocessor in
            preprocessors}


def get_BinaryLabelDatasetMetric(dataset_train: BinaryLabelDataset, preproc_datasets: dict,
                                 privileged_groups: dict,
                                 unprivileged_groups: dict):
    """
    Returns BinaryLabelDatasetMetric for both the unprocessed training data
    and the pre-processed training data.

    Parameters
    ----------
    dataset_train: BinaryLabelDataset
    preproc_datasets: dict
    privileged_groups: dict
    unprivileged_groups: dict

    Returns
    -------
    BinaryLabelDatasetMetric, BinaryLabelDatasetMetric

    """
    dataset_metric = BinaryLabelDatasetMetric(dataset_train,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)

    preproc_dataset_metrics = {key: BinaryLabelDatasetMetric(preproc_datasets[key],
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups) for key in
                               preproc_datasets}

    return dataset_metric, preproc_dataset_metrics


def train_models(models_trained: dict, Xs_preproc: dict, ys_preproc: dict, ws_preproc: dict):
    """
    models_train is a dict of dicts. Machine learning models -> Pre-processors
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
                                        constant=uniqueness[0]). \
                            fit(Xs_preproc[key_preproc], ys_preproc[key_preproc])
                else:
                    models_trained[key_model][key_preproc] = \
                        models_trained[key_model][key_preproc].fit(Xs_preproc[key_preproc], ys_preproc[key_preproc])

    return models_trained


def results_to_df(results):
    """
    Returns a pandas dataframe from results

    Parameters
    ----------
    results: dict
        results['model']['pre-processor']['accuracy']

    Returns
    -------

    """
    df_list = []
    for key_model in results:
        df = pd.DataFrame(results[key_model]).transpose()
        df['Model'] = key_model
        df['Preprocessor'] = df.index

        df_list.append(df)

    df_concat = pd.concat(df_list, keys=list(results.keys()))
    df_concat['ModelPreprocessor'] = df_concat.index.to_numpy()
    df_concat = df_concat.reset_index()
    df_concat = df_concat.drop(columns=['level_0', 'level_1'])

    return df_concat


def preprocess_pipeline(dataset_train: BinaryLabelDataset, dataset_test: BinaryLabelDataset,
                        privileged_groups: dict, unprivileged_groups: dict,
                        models: list, preprocessors: list):
    """
    Pipeline for evaluating pre-processors:
    1. Pre-process dataset_train with all pre-processors
    2. Evaluate pre-processed dataset_train
    3. Train machine learning models on each pre-processed dataset_train
    4. Evaluate on multiple metrics (performance, fairness)

    Parameters
    ----------
    dataset_train: BinaryLabelDataset
    dataset_test: BinaryLabelDataset
    privileged_groups: dict
    unprivileged_groups: dict
    models: list
    preprocessors: list of preprocessors from aif360
        pre-processors that work on BinaryLabelDataset

    Returns
    -------

    """
    # dictionary with (key, value) = (pre-processor name, BinaryLabelDataset)
    preproc_datasets = {type(preprocessor).__name__: preprocessor.fit_transform(dataset_train) for preprocessor in
                        preprocessors}

    # evaluate fairness in datasets
    dataset_metric, preproc_dataset_metrics = \
        get_BinaryLabelDatasetMetric(dataset_train,
                                     preproc_datasets,
                                     privileged_groups,
                                     unprivileged_groups)

    dataset_evaluation = evaluate_dataset_metrics('Dataset (Original)', dataset_metric)
    df_all_dataset_evaluation = pd.concat(
        [evaluate_dataset_metrics(key, value) for key, value in preproc_dataset_metrics.items()])
    df_all_dataset_evaluation = pd.concat([dataset_evaluation, df_all_dataset_evaluation])

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

    results = evaluate_ml_models(results, models_trained, X_test, y_test, z_test,
                                 privileged_groups)
    df_results = results_to_df(results)

    return df_all_dataset_evaluation, df_results


def run_experiments(models, dataset="compas", protected_attribute="race", preprocessors_str=None,
                    n_runs=5, seed=1,
                    filepath='results'):
    """
    Runs the experiment for n_runs times with given machine learning models
    and pre-processing methods for a given dataset that discriminated on protected_attribute.

    models: list
        list of estimators with .fit()
    preprocessors: list
        list of strings
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
    preprocessors = []
    for i in range(len(preprocessors_str)):
        preprocessors.append(eval(preprocessors_str[i]))

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
    if not os.path.exists(f"{filepath}/{dataset}"):
        os.makedirs(f"{filepath}/{dataset}")

    all_results = pd.concat(results_list, axis=0)
    path = f"{filepath}/{dataset}/{protected_attribute}_classification_results.csv"
    all_results.to_csv(path)
    print(f"{path} saved")

    all_dataset_results = pd.concat(dataset_results_list, axis=0)
    path = f"{filepath}/{dataset}/{protected_attribute}_dataset.csv"
    all_dataset_results.to_csv(path)
    print(f"{path} saved")
