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

from evaluation.pipeline import run_experiments
from src.preprocessing import MetricOptimizer, OriginalData, PreprocessingWrapper


def run_comparison_preprocessors():
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

    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    # Optimized Preproc. requires distortion functions
    preprocessors_str = ["OriginalData()",
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
                        preprocessors_str=preprocessors_str,
                        n_runs=n_runs,
                        seed=seed)


def run_fairness_agnostic():
    seed = 1
    n_runs = 10

    dataset_pro_attributes = [('compas', 'race')]

    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    metrics = ['statistical_parity_absolute_difference',
               'normalized_mutual_information',
               'consistency_score_objective']

    metrics = ['disparate_impact_ratio_objective']

    for metric in metrics:
        preprocessing_metric_str = f"PreprocessingWrapper(MetricOptimizer(frac=0.75," \
                               f"m=5," \
                               f"fairness_metric={metric}," \
                               f"protected_attribute=protected_attribute," \
                               f"label=dataset_orig.label_names[0]))"
        preprocessors_str = ["OriginalData()",
                             preprocessing_metric_str]

        for dataset, protected_attribute in dataset_pro_attributes:
            print(f"{dataset} ({protected_attribute})")
            run_experiments(models=models,
                            dataset=dataset,
                            protected_attribute=protected_attribute,
                            preprocessors_str=preprocessors_str,
                            n_runs=n_runs,
                            seed=seed,
                            filepath=f"results/{metric}")


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

    preprocessors_str = ["OriginalData()",
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
                    preprocessors_str=preprocessors_str,
                    n_runs=n_runs,
                    seed=seed)


def main():
    experiments = {'run_fast': run_fast,
                   'run_comparison_preprocessors': run_comparison_preprocessors,
                   'run_fairness_agnostic': run_fairness_agnostic}

    pick = 'run_fairness_agnostic'

    experiments[pick]()


if __name__ == '__main__':
    main()
