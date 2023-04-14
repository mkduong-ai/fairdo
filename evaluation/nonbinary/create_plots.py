import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# Type 1/TrueType fonts are supported natively by most platforms
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from fado.metrics.nonbinary import nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information


def plot_results(results_df,
                 groups=None,
                 disc_dict=None,
                 save_path=None):
    """
    Plots the results

    Parameters
    ----------
    results_df: pandas DataFrame
    groups: list of strings
        list of columns to group by
    disc_dict: dict
        dictionary of discrimination measures
    save_path: str
        path to save the plot

    Returns
    -------
    None
    """
    if groups is None:
        groups = ['Method']

    # get the list of discrimination measures
    disc_list = list(map(lambda x: x.__name__, disc_dict.values()))
    disc_time_list = ['time_' + disc_measure_str for disc_measure_str in disc_list]

    # reformat the dataframe
    id_vars = list(set(results_df.columns) - set(disc_list))
    df_plot = results_df.melt(id_vars=id_vars,
                              value_vars=disc_list,
                              var_name="Discrimination Measure",
                              value_name="Value")
    df_plot = df_plot.drop(columns=disc_time_list)

    # rename the discrimination measures
    rename_dict = dict(zip(disc_list, disc_dict.keys()))
    df_plot['Discrimination Measure'] = df_plot['Discrimination Measure'].replace(rename_dict)

    # plot the results
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    ax = sns.barplot(data=df_plot,
                     x='Discrimination Measure',
                     y='Value',
                     hue='Method',
                     errorbar='sd',
                     capsize=.2,
                     palette='deep',
                     )
    ax.legend(loc='best')
    # Set the title of the plot
    plt.title(results_df['data'][0].capitalize() + ' Dataset')

    if save_path is not None:
        plt.savefig(save_path + '.pdf', format='pdf')

    # Show the plot
    plt.show()


def plot_results_deprecated(results, save_path=None):
    # Plot the mean and standard deviation of the results using Matplotlib
    for method, method_results in results.items():
        for func, results in method_results.items():
            mean = np.mean(results['func_values'])
            std = np.std(results['func_values'])
            plt.scatter(f"{method} {func}", mean, label=f"{method} {func}")
            plt.errorbar(f"{method} {func}", mean, yerr=std, fmt='o')
            plt.xlabel('Method and Objective Function')
            plt.ylabel('Discrimination')
            plt.legend()
    if save_path is not None:
        plt.savefig(save_path, format='pdf')

    plt.show()


def main():
    data_str = 'adult'
    disc_dict = {  # 'Absolute Statistical Disparity': statistical_parity_absolute_difference,
        # 'Absolute Statistical Disparity Sum (non-binary)': nb_statistical_parity_sum_abs_difference,
        'Maximal Statistical Disparity': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information}

    # load the results
    save_path = f'results/nonbinary/{data_str}'
    results = pd.read_csv(save_path+'/results.csv')

    # plot the results
    plot_results(results,
                 disc_dict=disc_dict,
                 save_path=f'{save_path}/results')


if __name__ == "__main__":
    main()
