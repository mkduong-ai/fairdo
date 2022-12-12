import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(style='darkgrid')


def plot_dataframe_aggregate(results_df: pd.DataFrame,
                             x_axis='Mutual Information', y_axis='F1 Score',
                             dataset='COMPAS',
                             protected_attribute='race',
                             groups=None,
                             model=None,
                             filepath=None,
                             show=False,
                             save=True):
    """
    Plot results_df. Aggregate the results over the column given by 'groups'
    and show mean and standard deviation.

    Parameters
    ----------
    results_df: DataFrame
        columns: Metrics, ..., Preprocessor, Model
    x_axis: str
        column of DataFrame for x-axis
    y_axis: str
        column of DataFrame for y-axis
    dataset: str
        Name of dataset
    protected_attribute: str
        column name of protected attribute in results_df
    groups: None
        list of strings to group by
    model: str
        Estimator name
    filepath: str
        Path of .csv file
    show: boolean
        Bool whether to show the plot.
    save: boolean
        Bool whether plot should be saved.
    Returns
    -------

    """
    # init
    if groups is None:
        groups = ['ModelPreprocessor']
    if model is not None:
        results_df = results_df[results_df['Model'] == model]

    # means and standard deviation
    results_df = results_df.drop(columns=['Model', 'Preprocessor'])
    x_mean = results_df.groupby(groups).mean()[x_axis]
    y_mean = results_df.groupby(groups).mean()[y_axis]
    xerr_std = results_df.groupby(groups).std()[x_axis]
    yerr_std = results_df.groupby(groups).std()[y_axis]

    xerr = xerr_std
    yerr = yerr_std

    # figsize
    plt.figure(figsize=(5, 2.5), dpi=80)
    for i in range(len(x_mean)):
        if model is not None:
            label = eval(x_mean.index[i])[1]
        else:
            label = x_mean.index[i]
        plt.errorbar(x=x_mean.iloc[i], y=y_mean.iloc[i],
                     xerr=xerr.iloc[i], yerr=yerr.iloc[i],
                     fmt='.', label=label, alpha=0.7)

    # title, legend, labels
    # plt.title(f"{dataset.upper()} Dataset")
    plt.legend(loc='lower right', prop={'size': 8})
    #plt.xlabel(f"{protected_attribute.capitalize()} Discrimination ({x_axis})")
    plt.xlabel(f"{dataset.capitalize()} ({x_axis})")
    plt.ylabel(y_axis)

    # axes ranges
    ax = plt.gca()
    if all(x_mean < 1):
        ax.set_xlim([0, 1])
        if all(x_mean < 0.5):
            ax.set_xlim([0, 0.5])
            #if all(x_mean < 0.1):
            #    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 1])

    # filename
    disc_name = x_axis.replace(" ", "")
    filename = f"{filepath.split('.')[0]}_{model}_{y_axis}_{disc_name}.pdf"
    # save plot
    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        print(f"Figure saved under {filename}")

    if show:
        plt.show()

    plt.close()


def save_plots_over_models_datasets(x_axis: str, y_axis: str, models: list, dataset_pro_attributes: list,
                                    rename_columns=None, show=False,
                                    filepath_prefix='results'):
    """
    Create separate plots for each machine learning model and dataset.

    Parameters
    ----------
    x_axis
    y_axis
    models
    dataset_pro_attributes
    rename_columns
    show
    filepath_prefix

    Returns
    -------

    """
    for model in models:
        for dataset, protected_attribute in dataset_pro_attributes:
            # read results.csv file
            filepath = f"{filepath_prefix}/{dataset}/{protected_attribute}_classification_results.csv"
            clf_results = pd.read_csv(filepath)

            # rename columns
            if rename_columns is not None:
                # rename_columns is a dictionary
                clf_results.rename(columns=rename_columns, inplace=True)

            # plot
            plot_dataframe_aggregate(clf_results, x_axis=x_axis, y_axis=y_axis,
                                     dataset=dataset,
                                     protected_attribute=protected_attribute,
                                     model=model,
                                     filepath=filepath,
                                     show=show)


def save_plots_over_xy_axes(x_axes: list, y_axes: list, models: list, dataset_pro_attributes: list,
                            show=False, filepath_prefix='results'):
    """

    Parameters
    ----------
    x_axes: list of strings
        ['Statistical Parity Abs Diff', 'Normalized MI']
    y_axes: list of strings
        ['AUC', 'Balanced Accuracy']
    models: list of strings
        ['KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier']
    dataset_pro_attributes: list of 2-tuples of strings consisting of dataset name and protected attribute
        [('adult', 'sex'),
         ('bank', 'age'),
         ('compas', 'race')]
    show: bool
        Whether to show the plot.
    filepath_prefix: str
    Returns
    -------

    """
    for x_axis in x_axes:
        for y_axis in y_axes:
            save_plots_over_models_datasets(x_axis=x_axis, y_axis=y_axis,
                                            models=models, dataset_pro_attributes=dataset_pro_attributes,
                                            show=show,
                                            filepath_prefix=filepath_prefix)
