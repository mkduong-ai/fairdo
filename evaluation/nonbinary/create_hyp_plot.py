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
from fairdo.metrics import (normalized_mutual_information,
                            statistical_parity_abs_diff,
                            statistical_parity_abs_diff_max,
                            statistical_parity_abs_diff_mean)
from measures import count_groups, count_size, sanity_check


def plot_results(results_df,
                 groups=None,
                 disc_dict=None,
                 save_path=None,
                 time=False,
                 show_plot=False):
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
    time: bool
        whether to plot the time
    show_plot: bool
        whether to show the plot

    Returns
    -------
    None
    """
    # Filter out rows where the 'Method' column starts with 'GA'
    results_df = results_df[results_df['Method'].str.startswith('GA')]

    # Extract the population size and number of generations from the 'Method' column
    results_df['pop_size'] = results_df['Method'].str.extract(r'(pop_size=)(\d+)')[1].astype(int)
    results_df['num_generations'] = results_df['Method'].str.extract(r'(num_generations=)(\d+)')[1].astype(int)

    # Go through each discrimination measure
    for key_disc_measure, disc_measure_str in disc_dict.items():
        # Create a pivot table with 'pop_size' and 'num_generations' as the index and columns, respectively,
        # and 'statistical_parity_abs_diff' as the values
        pivot_mean = results_df.pivot_table(values=('time_' if time else '') + disc_measure_str.__name__,
                                            index='pop_size',
                                            columns='num_generations', aggfunc=np.mean)

        # Create a pivot table for standard deviation
        pivot_std = results_df.pivot_table(values=('time_' if time else '') + disc_measure_str.__name__,
                                           index='pop_size',
                                           columns='num_generations', aggfunc=np.std)

        # Create annotation matrix with mean and standard deviation
        annotations = pivot_mean.round(2).astype(str) + ' ± ' + pivot_std.round(2).astype(str)

        # Draw the heatmap
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        sns.heatmap(pivot_mean,
                    annot=annotations,
                    fmt="", cmap="magma",
                    vmin=pivot_mean.min().min(),
                    vmax=pivot_mean.max().max(),
                    ax=ax,
                    annot_kws={"size": 8})
        ax.set_xlabel('Number of Generations')
        ax.set_ylabel('Population Size')
        plt.title(key_disc_measure)

        # save plot
        if save_path is not None:
            plt.savefig(save_path + ('_time_' if time else '_') + f'{key_disc_measure.replace(" ", "")}' + '.pdf',
                        format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

        # show plot
        if show_plot:
            plt.show()


def settings(data_str='compas', objective_str='remove_synthetic',
             time=False):
    disc_dict = {
        'Sum SDP': statistical_parity_abs_diff,
        # 'Maximal SDP': statistical_parity_abs_diff_max,
        # 'NMI': normalized_mutual_information
    }

    # load the results
    save_path = f'evaluation/results/nonbinary/hyperparameter/{data_str}/{data_str}_{objective_str}'
    results = pd.read_csv(f'{save_path}.csv')

    # plot the results
    plot_results(results,
                 disc_dict=disc_dict,
                 save_path=save_path,
                 time=time,
                 show_plot=False)


def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas', 'bank']
    # Experiments
    # obj_strs = ['remove']
    # data_strs = ['compas']
    for data_str in data_strs:
        for obj_str in obj_strs:
            settings(data_str=data_str,
                     objective_str=obj_str,
                     time=True)
            # settings_time(data_str=data_str, objective_str=obj_str)


if __name__ == "__main__":
    main()
