import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')


def create_plot_from_clf_results(results_df: pd.DataFrame, x_axis='Mutual Information', y_axis='F1 Score',
                                 dataset='compas', protected_attribute='race',
                                 groups=None,
                                 model=None,
                                 filepath=None,
                                 show=False):
    """

    Parameters
    ----------
    results_df: DataFrame
        columns: Metrics, ..., Preprocessor, Model
    x_axis: str
        column of DataFrame for x-axis
    y_axis: str
        column of DataFrame for y-axis
    dataset: str
    protected_attribute: str
    groups: list
        list of strings to group by
    model: str
        Estimator name
    show: boolean
    filepath: str
        prefix of directory to save
    Returns
    -------

    """
    # init
    if groups is None:
        groups = ['ModelPreprocessor']
    if model is not None:
        results_df = results_df[results_df['Model'] == model]

    # means and standard deviation
    x_mean = results_df.groupby(groups).mean()[x_axis]
    y_mean = results_df.groupby(groups).mean()[y_axis]
    xerr_std = results_df.groupby(groups).std()[x_axis]
    yerr_std = results_df.groupby(groups).std()[y_axis]
    # confidence interval
    # xerr = 1.96 * results_df.groupby(groups).std()[x_axis]
    # yerr = 1.96 * results_df.groupby(groups).std()[y_axis]
    # std error
    # xerr_std = results_df.groupby(['Model']).std()[x_axis]/np.sqrt(results_df.groupby(['Model'])[x_axis].count())
    # yerr_std = results_df.groupby(['Model']).std()[y_axis]/np.sqrt(results_df.groupby(['Model'])[y_axis].count())

    xerr = xerr_std
    yerr = yerr_std

    # figsize
    plt.figure(figsize=(5, 3.5), dpi=80)
    for i in range(len(x_mean)):
        if model is not None:
            label = eval(x_mean.index[i])[1]
        else:
            label = x_mean.index[i]
        plt.errorbar(x=x_mean.iloc[i], y=y_mean.iloc[i],
                     xerr=xerr.iloc[i], yerr=yerr.iloc[i],
                     fmt='.', label=label, alpha=0.7)

    plt.legend(loc='lower right', prop={'size': 11})
    plt.xlabel(f"{protected_attribute.capitalize()} Discrimination ({x_axis})")
    plt.ylabel(y_axis)
    # plt.title(f"{dataset.upper()} Dataset")
    ax = plt.gca()
    if all(x_mean < 1):
        ax.set_xlim([0, 1])
        if all(x_mean < 0.5):
            ax.set_xlim([0, 0.5])
            if all(x_mean < 0.1):
                ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 1])

    # save plot
    # create plot folder
    disc_name = x_axis.replace(" ", "")
    print(filepath)
    plt.savefig(f"{filepath.split('.')[0]}_{model}_{y_axis}_{disc_name}.pdf", bbox_inches='tight')
    print(f"Figure saved under {filepath.split('.')[0]}_{model}_{y_axis}_{disc_name}.pdf")
    if show:
        plt.show()


def export_plots_over_models_datasets(x_axis: str, y_axis: str, models: list, dataset_pro_attributes: list,
                                      rename_columns: dict, show=False,
                                      filepath_prefix='results'):
    for model in models:
        for dataset, protected_attribute in dataset_pro_attributes:
            # read results.csv file
            filepath = f"{filepath_prefix}/{dataset}/{protected_attribute}_classification_results.csv"
            clf_results = pd.read_csv(filepath)
            # rename columns
            clf_results.rename(columns=rename_columns, inplace=True)
            # rename values

            # plot
            create_plot_from_clf_results(clf_results, x_axis=x_axis, y_axis=y_axis,
                                        dataset=dataset,
                                        protected_attribute=protected_attribute,
                                        model=model,
                                        filepath=filepath,
                                        show=show)


def plot_all_datasets():
    # templates
    x_axes = ['Mutual Information',
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
    y_axes = ['Accuracy', 'F1 Score', 'Balanced Accuracy', 'AUC']

    # settings
    x_axis = 'Statistical Parity Abs Diff'
    y_axis = 'AUC'
    rename_columns = {'DisparateImpactRemover': 'DIR'}

    # iteration
    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('adult', 'sex'),
                              ('bank', 'age'),
                              ('compas', 'race')]

    export_plots_over_models_datasets(x_axis, y_axis, models, dataset_pro_attributes, rename_columns)


def plot_all_datasets_all_metrics():
    # settings
    x_axes = ['Statistical Parity Abs Diff',
              'Normalized MI',
              'Average Odds Error']
    y_axes = ['AUC']
    rename_columns = {'DisparateImpactRemover': 'DIR'}

    # iteration
    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('adult', 'sex'),
                              ('bank', 'age'),
                              ('compas', 'race')]
    for x_axis in x_axes:
        for y_axis in y_axes:
            export_plots_over_models_datasets(x_axis, y_axis, models, dataset_pro_attributes, rename_columns)


def plot_fairness_agnostic():
    # settings
    x_axes = {'Statistical Parity Abs Diff': 'statistical_parity_absolute_difference',
              'Normalized MI': 'normalized_mutual_information',
              'Consistency Obj': 'consistency_score_objective'}
    x_axes = {'Disparate Impact Obj': 'disparate_impact_ratio_objective'}
    y_axis = 'AUC'

    models = ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes = [('compas', 'race')]
    rename_columns = {'DisparateImpactRemover': 'DIR'}

    for metric_name, metric_path in x_axes.items():
        export_plots_over_models_datasets(metric_name, y_axis, models, dataset_pro_attributes, rename_columns,
                                          filepath_prefix=f"results/{metric_path}")


def quick_plot():
    rename_columns = {'DisparateImpactRemover': 'DIR'}

    # settings
    x_axis = 'Disparate Impact Obj'
    y_axis = 'AUC'
    model = ['KNeighborsClassifier']

    dataset_pro_attributes = [('adult', 'sex')]

    export_plots_over_models_datasets(x_axis, y_axis, model, dataset_pro_attributes, rename_columns, show=True)


def main():
    experiments = {'quick_plot': quick_plot,
                   'plot_all_datasets': plot_all_datasets,
                   'plot_all_dataset_all_metrics': plot_all_datasets_all_metrics,
                   'plot_fairness_agnostic': plot_fairness_agnostic}

    pick = 'plot_fairness_agnostic'

    experiments[pick]()


if __name__ == '__main__':
    main()
