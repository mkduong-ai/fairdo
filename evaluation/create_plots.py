from plot.helper import save_plots_over_models_datasets, save_plots_over_xy_axes
from settings import get_evaluation_config


x_axis_mapper = {'statistical_parity_absolute_difference': 'Statistical Parity Abs Diff',
                 'normalized_mutual_information': 'Normalized MI',
                 'consistency_score_objective': 'Consistency Obj',
                 'disparate_impact_ratio_objective': 'Disparate Impact Obj'}


def plot_all_datasets(frac):
    # settings
    x_axes = ['Statistical Parity Abs Diff']
    y_axes = ['AUC']

    # iteration
    dataset_pro_attributes, models, filepath = get_evaluation_config(config='comparison_preprocessors',
                                                                     frac=frac,
                                                                     plot=True)
    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes, filepath_prefix=filepath)


def plot_all_datasets_metrics(frac):
    # settings
    x_axes = ['Statistical Parity Abs Diff',
              'Normalized MI',
              'Average Odds Error']
    y_axes = ['AUC']

    # iteration
    dataset_pro_attributes, models, filepath = get_evaluation_config(config='comparison_preprocessors',
                                                                     frac=frac,
                                                                     plot=True)
    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes, filepath_prefix=filepath)


def plot_fairness_agnostic(frac):
    # settings
    y_axis = 'AUC'

    # preprocess on multiple metrics
    dataset_pro_attributes, models, metrics, filepath = get_evaluation_config(config='fairness_agnostic',
                                                                              frac=frac,
                                                                              plot=True)
    x_axes = {k: x_axis_mapper[k] for k in metrics}

    for metric_path, metric_name in x_axes.items():
        save_plots_over_models_datasets(metric_name, y_axis, models, dataset_pro_attributes,
                                        filepath_prefix=f"{filepath}/{metric_path}")


def quick_plot(frac):
    # settings
    x_axes = ['Disparate Impact Obj']
    y_axes = ['AUC']

    dataset_pro_attributes, models, filepath = get_evaluation_config(config='quick',
                                                                     frac=frac,
                                                                     plot=True)

    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes, show=False,
                            filepath_prefix=f"{filepath}/quick")


def main():
    experiments = {'quick_plot': quick_plot,
                   'plot_all_datasets': plot_all_datasets,
                   'plot_all_dataset_metrics': plot_all_datasets_metrics,
                   'plot_fairness_agnostic': plot_fairness_agnostic}

    pick = 'plot_fairness_agnostic'
    frac = 1.25

    experiments[pick](frac=frac)


if __name__ == '__main__':
    main()
