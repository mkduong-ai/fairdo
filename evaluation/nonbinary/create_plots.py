import numpy as np
import pandas as pd
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
# Type 1/TrueType fonts are supported natively by most platforms
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(font_scale=0.8)

from fado.metrics.nonbinary import nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information, nb_statistical_parity_sum_abs_difference

# load fake metrics that serve other purposes
from measures import count_size, count_groups, sanity_check


def plot_results(results_df,
                 groups=None,
                 disc_dict=None,
                 save_path=None,
                 show_plot=True):
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
    figure(figsize=(6, 2), dpi=80)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    ax = sns.barplot(data=df_plot,
                     x='Discrimination Measure',
                     y='Value',
                     hue='Method',
                     errorbar='sd',
                     capsize=.2,
                     palette='deep',
                     )
    ax.set_yscale('log')
    # ax.legend(loc='best')
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
                    fontsize=8)
    # Set the title of the plot
    # plt.title(results_df['data'][0].capitalize() + ' Dataset')

    if save_path is not None:
        plt.savefig(save_path + '.pdf', format='pdf',  bbox_inches='tight', pad_inches=0)

    # Show the plot
    if show_plot:
        plt.show()


def settings(data_str='compas', objective_str='remove_synthetic'):
    disc_dict = {  # 'Absolute Statistical Disparity': statistical_parity_absolute_difference,
        'Sum SDP': nb_statistical_parity_sum_abs_difference,
        'Maximal SDP': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information}

    # load the results
    save_path = f'evaluation/results/nonbinary/{data_str}/{data_str}_{objective_str}'
    results = pd.read_csv(f'{save_path}.csv')

    # plot the results
    plot_results(results,
                 disc_dict=disc_dict,
                 save_path=save_path,
                 show_plot=False)


def plot_time(results_df,
                 groups=None,
                 disc_dict=None,
                 save_path=None,
                 show_plot=True):
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
    id_vars = list(set(results_df.columns) - set(disc_time_list))
    df_plot = results_df.melt(id_vars=id_vars,
                              value_vars=disc_time_list,
                              var_name="Discrimination Measure",
                              value_name="Runtime (s)")

    # rename the discrimination measures
    rename_dict = dict(zip(disc_time_list, disc_dict.keys()))
    df_plot['Discrimination Measure'] = df_plot['Discrimination Measure'].replace(rename_dict)

    # plot the results
    figure(figsize=(6, 2), dpi=80)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    ax = sns.barplot(data=df_plot,
                     x='Discrimination Measure',
                     y='Runtime (s)',
                     hue='Method',
                     errorbar='sd',
                     capsize=.2,
                     palette='deep',
                     )
    # ax.set_yscale('log')
    # ax.legend(loc='best')
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
                    fontsize=8)
    # Set the title of the plot
    # plt.title(results_df['data'][0].capitalize() + ' Dataset')

    if save_path is not None:
        plt.savefig(save_path + '_time.pdf', format='pdf',  bbox_inches='tight', pad_inches=0)

    # Show the plot
    if show_plot:
        plt.show()


def settings_time(data_str='compas', objective_str='remove_synthetic'):
    disc_dict = {  # 'Absolute Statistical Disparity': statistical_parity_absolute_difference,
        'Sum SDP': nb_statistical_parity_sum_abs_difference,
        'Maximal SDP': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information,}
        #'Size': count_size,
        #'Groups': count_groups}

    # load the results
    save_path = f'evaluation/results/nonbinary/{data_str}/{data_str}_{objective_str}'
    results = pd.read_csv(f'{save_path}.csv')

    # plot the results
    plot_time(results,
                 disc_dict=disc_dict,
                 save_path=save_path,
                 show_plot=False)


def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas']
    for data_str in data_strs:
        for obj_str in obj_strs:
            #settings(data_str=data_str, objective_str=obj_str)
            settings_time(data_str=data_str, objective_str=obj_str)


if __name__ == "__main__":
    main()
