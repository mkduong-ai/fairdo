from plot_helper import save_plots_over_models_datasets, save_plots_over_xy_axes

x_axes_template = ['Mutual Information',
                   'Normalized MI',
                   'Randomized Dependence Coefficient',
                   'Pearson Correlation',
                   'Statistical Parity Abs Diff',
                   'Disparate Impact', 'Disparate Impact Obj',
                   'Equal Opportunity Abs Diff',
                   'Predictive Equality Abs Diff',
                   'Average Odds Diff'
                   'Average Odds Error',
                   'Consistency',
                   'Consistency Obj']
y_axes_template = ['Accuracy', 'F1 Score', 'Balanced Accuracy', 'AUC']


def plot_all_datasets():
    # settings
    x_axes = ['Statistical Parity Abs Diff']
    y_axes = ['AUC']

    # iteration
    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('adult', 'sex'),
                              ('bank', 'age'),
                              ('compas', 'race')]
    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes)


def plot_all_datasets_metrics():
    # settings
    x_axes = ['Statistical Parity Abs Diff',
              'Normalized MI',
              'Average Odds Error']
    y_axes = ['AUC']

    # iteration
    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('adult', 'sex'),
                              ('bank', 'age'),
                              ('compas', 'race')]
    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes)


def plot_fairness_agnostic():
    # settings
    x_axes = {'Statistical Parity Abs Diff': 'statistical_parity_absolute_difference',
              'Normalized MI': 'normalized_mutual_information',
              'Consistency Obj': 'consistency_score_objective'}
    x_axes = {'Disparate Impact Obj': 'disparate_impact_ratio_objective'}
    y_axis = 'AUC'

    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('compas', 'race')]

    for metric_name, metric_path in x_axes.items():
        save_plots_over_models_datasets(metric_name, y_axis, models, dataset_pro_attributes,
                                        filepath_prefix=f"results/{metric_path}")


def quick_plot():
    # settings
    x_axes = ['Disparate Impact Obj']
    y_axes = ['AUC']
    models = ['KNeighborsClassifier']

    dataset_pro_attributes = [('adult', 'sex')]

    save_plots_over_xy_axes(x_axes, y_axes, models, dataset_pro_attributes, show=True)


def main():
    experiments = {'quick_plot': quick_plot,
                   'plot_all_datasets': plot_all_datasets,
                   'plot_all_dataset_metrics': plot_all_datasets_metrics,
                   'plot_fairness_agnostic': plot_fairness_agnostic}

    pick = 'plot_fairness_agnostic'

    experiments[pick]()


if __name__ == '__main__':
    main()
