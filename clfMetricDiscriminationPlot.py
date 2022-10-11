import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')


def plot_classification_results(results_df: pd.DataFrame, x_axis='Mutual Information', y_axis='F1 Score',
                                dataset='compas', protected_attribute='race',
                                groups=None,
                                model=None,
                                filepath='.pdf',
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
        filename of figure
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
    # xerr = 1.96 * results_df.groupby(['Model']).std()[x_axis]
    # yerr = 1.96 * results_df.groupby(['Model']).std()[y_axis]
    # std error
    # xerr_std = results_df.groupby(['Model']).std()[x_axis]/np.sqrt(results_df.groupby(['Model'])[x_axis].count())
    # yerr_std = results_df.groupby(['Model']).std()[y_axis]/np.sqrt(results_df.groupby(['Model'])[y_axis].count())

    xerr = xerr_std
    yerr = yerr_std

    for i in range(len(x_mean)):
        if model is not None:
            print(x_mean)
            label = eval(x_mean.index[i])[1]
        else:
            label = x_mean.index[i]
        plt.errorbar(x=x_mean.iloc[i], y=y_mean.iloc[i],
                     xerr=xerr.iloc[i], yerr=yerr.iloc[i],
                     fmt='.', label=label, alpha=0.7)

    plt.legend(loc='lower right', prop={'size': 8})
    plt.xlabel(f"{protected_attribute.capitalize()} Discrimination ({x_axis})")
    plt.ylabel(y_axis)
    plt.title(f"{dataset.upper()} Dataset")
    ax = plt.gca()
    if all(x_mean < 1):
        ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # save plot
    # create plot folder
    plt.savefig(f"{filepath.split('.')[0]}_{model}_{y_axis}_{x_axis}.pdf", bbox_inches='tight')
    if show:
        plt.show()


def main():
    # templates
    x_axes = ['Mutual Information', 'Randomized Dependence Coefficient',
              'Pearson Correlation',
              'Statistical Parity Abs Diff',
              'Disparate Impact', 'Disparate Impact Obj',
              'Equal Opportunity Abs Diff',
              'Predictive Equality Abs Diff',
              'Average Odds Abs Diff'
              'Average Odds Error']
    y_axes = ['Accuracy', 'F1 Score', 'Balanced Accuracy', 'AUC']

    # settings
    x_axis = 'Statistical Parity Abs Diff'
    y_axis = 'AUC'

    # dataset_pro_attributes = [('adult', 'sex'),
    #                           ('compas', 'race'),
    #                           ('german', 'foreign_worker')]
    dataset_pro_attributes = [('compas', 'race')]
    for dataset, protected_attribute in dataset_pro_attributes:
        # read results.csv file
        filepath = f"results/{dataset}/{protected_attribute}_classification_results.csv"
        clf_results = pd.read_csv(filepath)
        # plot
        plot_classification_results(clf_results, x_axis=x_axis, y_axis=y_axis,
                                    dataset=dataset,
                                    protected_attribute=protected_attribute,
                                    model='KNeighborsClassifier',
                                    filepath=filepath,
                                    show=True)


if __name__ == '__main__':
    main()
