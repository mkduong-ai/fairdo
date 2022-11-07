from pipeline.pipeline import run_experiments
from pipeline.helper import PreprocessingWrapper
from settings import get_evaluation_config

from fado.preprocessing import MetricOptimizer, OriginalData

# fair preprocessors
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing


def run_comparison_preprocessors():
    seed = 1
    n_runs = 10
    dataset_pro_attributes, models = get_evaluation_config(config='comparison_preprocessors')

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

    dataset_pro_attributes, models, metrics = get_evaluation_config(config='fairness_agnostic')

    for metric in metrics:
        preprocessing_metric_str = f"PreprocessingWrapper(MetricOptimizer(frac=1.75," \
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


def run_quick():
    # settings
    n_runs = 2
    seed = 1

    dataset_pro_attributes, models = get_evaluation_config(config='quick')

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


def main():
    experiments = {'run_quick': run_quick,
                   'run_comparison_preprocessors': run_comparison_preprocessors,
                   'run_fairness_agnostic': run_fairness_agnostic}

    pick = 'run_quick'

    experiments[pick]()


if __name__ == '__main__':
    main()
