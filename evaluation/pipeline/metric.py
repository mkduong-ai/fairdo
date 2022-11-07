import numpy as np
import pandas as pd

## aif360 dataset metrics
from aif360.metrics import BinaryLabelDatasetMetric

## classification metrics
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, \
    recall_score
## fairness metrics
from src.fado.metrics import mutual_information, normalized_mutual_information,\
    rdc, statistical_parity_absolute_difference, \
    equal_opportunity_absolute_difference, disparate_impact_ratio, disparate_impact_ratio_objective,\
    predictive_equality_absolute_difference, average_odds_error, average_odds_difference,\
    consistency_score, consistency_score_objective


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


def evaluate_ml_models(results: dict, models_trained: dict, X_test, y_test, z_test) -> dict:
    """

    Parameters
    ----------
    results: dict
        e.g. stores results in results['LogisticRegression']['DisparateImpactRemover']['Accuracy']
    models_trained: dict
        e.g. models are in models_trained['LogisticRegression']['DisparateImpactRemover']
    X_test: ndarray
    y_test: ndarray
        flattened (n,) shape
    z_test: ndarray
        flattened (n,) shape

    Returns
    -------
    dict
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
