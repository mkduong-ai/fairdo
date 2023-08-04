# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure

# Configuration for matplotlib and seaborn
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(font_scale=0.8)

# Local application/library specific imports
from fado.metrics import (normalized_mutual_information,
                          statistical_parity_abs_diff,
                          statistical_parity_abs_diff_max,
                          statistical_parity_abs_diff_mean)
from measures import count_groups, count_size, sanity_check


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
    # Filter out rows where the 'Method' column starts with 'GA'
    results_df = results_df[results_df['Method'].str.startswith('GA')]

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

    # Extract the population size and number of generations from the 'Method' column
    results_df['pop_size'] = results_df['Method'].str.extract(r'(pop_size=)(\d+)')[1].astype(int)
    results_df['num_generations'] = results_df['Method'].str.extract(r'(num_generations=)(\d+)')[1].astype(int)

    # Create a pivot table with 'pop_size' and 'num_generations' as the index and columns, respectively,
    # and 'statistical_parity_abs_diff' as the values
    pivot = results_df.pivot_table(values='statistical_parity_abs_diff', index='pop_size', columns='num_generations')

    # Create a pivot table with 'pop_size' and 'num_generations' as the index and columns, respectively,
    # and 'statistical_parity_abs_diff' as the values
    pivot = results_df.pivot_table(values='statistical_parity_abs_diff', index='pop_size', columns='num_generations')

    # Draw the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma",
                vmin=pivot.min().min(),
                vmax=pivot.max().max())
    plt.title("Heatmap of Statistical Parity Absolute Difference")
    plt.show()


def settings(data_str='compas', objective_str='remove_synthetic'):
    disc_dict = {  # 'Absolute Statistical Disparity': statistical_parity_absolute_difference,
        'Sum SDP': statistical_parity_abs_diff,
        #'Maximal SDP': statistical_parity_abs_diff_max,
        #'NMI': normalized_mutual_information
    }

    # load the results
    save_path = f'evaluation/results/nonbinary/hyperparameter/{data_str}/{data_str}_{objective_str}'
    results = pd.read_csv(f'{save_path}.csv')

    # plot the results
    plot_results(results,
                 disc_dict=disc_dict,
                 save_path=save_path,
                 show_plot=False)


def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas']
    # Experiments
    obj_strs = ['remove']
    data_strs = ['compas']
    for data_str in data_strs:
        for obj_str in obj_strs:
            settings(data_str=data_str, objective_str=obj_str)
            #settings_time(data_str=data_str, objective_str=obj_str)


if __name__ == "__main__":
    main()