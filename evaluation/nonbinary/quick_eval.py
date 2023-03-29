import time
import datetime
import csv
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# load metrics
from fado.metrics import statistical_parity_absolute_difference, normalized_mutual_information
from fado.metrics.nonbinary import nb_statistical_parity_sum_abs_difference, nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information

# load fado library
# from fado.preprocessing import MetricOptimizer, MetricOptRemover

# load optimization methods
from optimize import SimulatedAnnealing, GeneticAlgorithm, MetricOptimizer, Baseline


def load_data(dataset_str):
    """

    Parameters
    ----------
    dataset_str: str

    Returns
    -------
    df: pandas DataFrame
    label: str
    protected_attributes: list of str
    """
    if dataset_str == 'adult':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None,
                           names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                  "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                  "hours-per-week", "native-country", "income"])
        print('Data downloaded.')
        # drop columns
        cols_to_drop = ['fnlwgt', 'workclass', 'education', 'occupation', 'native-country']
        data = data.drop(columns=cols_to_drop)
        # label encoding protected_attribute and label
        label = 'income'
        protected_attributes = ['race']
        cols_to_labelencode = protected_attributes.copy()
        cols_to_labelencode.append(label)
        data[cols_to_labelencode] =\
            data[cols_to_labelencode].apply(LabelEncoder().fit_transform)
        # one-hot encoding categorical columns
        categorical_cols = list(data.select_dtypes(include='object'))
        data = pd.get_dummies(data, columns=categorical_cols)

        return data, label, protected_attributes


def f(binary_vector, dataframe, label, protected_attributes, disc_measure=statistical_parity_absolute_difference):
    """

    Parameters
    ----------
    binary_vector: np.array
    dataframe: pandas DataFrame
    label: str
    protected_attributes: list of strings
    disc_measure: callable
        takes in x, y, z and return a numeric value

    Returns
    -------
    numeric
    """
    y = dataframe[label]
    z = dataframe[protected_attributes]
    cols_to_drop = protected_attributes + [label]
    x = dataframe.drop(columns=cols_to_drop)

    # only keep the columns that are selected by the heuristic
    mask = np.array(binary_vector) == 1
    x, y, z = x[mask], y[mask], z[mask]

    # Note: This does not handle multiple protected attributes
    y = y.to_numpy().flatten()
    z = z.to_numpy().flatten()
    return disc_measure(x=x, y=y, z=z)


def convert_results_to_dataframe(results):
    """
    Converts the results to a pandas dataframe

    Parameters
    ----------
    results: dict
        results['method']['objective/time/func_values']

    Returns
    -------

    """
    for i in results.keys():
        # convert each result to a dataframe
        results[i] = pd.DataFrame.from_dict(results[i], orient='index')

        # Reset the index (method) to convert it into a column
        results[i].reset_index(inplace=True)

        # Rename the index column
        results[i] = results[i].rename({'index': 'Method'}, axis=1)

    results_list = list(results.values())
    results_df = pd.concat(results_list, axis=0).reset_index()
    results_df = results_df.drop(columns=['index'])
    return results_df


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


def plot_results(results_df,
                 groups=None,
                 save_path=None):
    """
    Plots the results

    Parameters
    ----------
    results_df: pandas DataFrame
    groups: list of strings
        list of columns to group by
    save_path: str
        path to save the plot

    Returns
    -------
    None
    """
    if groups is None:
        groups = ['Method', 'Objective Function']

    # aggregate the results
    discs_mean = results_df.groupby(groups).mean()
    discs_std = results_df.groupby(groups).std()

    # Plot the mean and standard deviation of the results using Matplotlib
    plt.figure(figsize=(10, 8))

    # Create the error bar plot using Matplotlib and Seaborn
    sns.set(style="whitegrid")
    ax = sns.errorbar(x=discs_mean.index, y=discs_mean['values'], yerr=discs_std['values'], fmt='o', capsize=5)

    # Set the x and y axis labels
    ax.set_xlabel('Group', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)

    # Set the title of the plot
    ax.set_title('Error Bar Plot', fontsize=16)

    # Set the legend
    plt.legend(['Values'], fontsize=12)

    # Show the plot
    plt.show()

    if save_path is not None:
        plt.savefig(save_path, format='pdf')


def run_experiment(data_str, disc_measures, methods):
    """
    Runs the experiment
    Parameters
    ----------
    data_str: str
        name of the dataset
    disc_measures: list of callables
        list of discrimination measures
    methods: dict
        dictionary of methods

    Returns
    -------
    results: dict
        results['method']['objective/time/func_values']
    """
    # settings
    df, label, protected_attributes = load_data(data_str)
    print('Successfully loaded data.')

    # create objective function
    f_obj = lambda x, disc_measure: f(x, dataframe=df, label=label, protected_attributes=protected_attributes,
                                      disc_measure=disc_measure)
    functions = [lambda x: f_obj(x, disc_measure=disc_measure) for disc_measure in disc_measures]
    for func, disc_measure in zip(functions, disc_measures):
        func.__name__ = disc_measure.__name__

    # run experiments
    results = {}
    for method_name, method in methods.items():
        print(f'Running {method_name}...')
        results[method_name] = {}
        for func in functions:
            print(f'Optimizing {func.__name__}...')
            start_time = time.time()
            results[method_name][func.__name__] = method(f=func, d=df.shape[0])[1]
            end_time = time.time()
            results[method_name]['time'] = end_time - start_time
            # information about the data
            results[method_name]['data'] = data_str
            results[method_name]['label'] = label
            results[method_name]['protected_attributes'] = protected_attributes

    return results


def run_experiments(data_str, disc_measures, methods, n_runs=10):
    """
    Runs the experiments for n_runs times
    Parameters
    ----------
    data_str: str
        name of the dataset
    disc_measures: list
        list of discrimination measures
    methods: dict
        dictionary of methods
    n_runs: int
        number of runs

    Returns
    -------
    results: dict
        results['run']['method']['objective/time/func_values']
    """
    results = {}
    for i in range(n_runs):
        print(f'Run {i+1} of {n_runs}')
        results[i] = run_experiment(data_str, disc_measures, methods)
    return results


def main():
    data_str = 'adult'
    n_runs = 2
    # create objective functions
    disc_measures = [# statistical_parity_absolute_difference,
                     # normalized_mutual_information,
                     # nb_statistical_parity_sum_abs_difference,
                     # nb_statistical_parity_max_abs_difference,
                     nb_normalized_mutual_information]
    # create methods
    methods = {'Baseline (Original)': Baseline.method_original,
               'Baseline (Random)': Baseline.method_random,
               #'Simulated Annealing': SimulatedAnnealing.simulated_annealing_method,
               #'Genetic Algorithm': GeneticAlgorithm.genetic_algorithm_method,
               #'Metric Optimizer': MetricOptimizer.metric_optimizer_remover}
               }

    # create save path
    save_path = f'results/nonbinary/{data_str}.csv'
    if not os.path.exists(f'results/nonbinary/{data_str}'):
        os.makedirs(f'results/nonbinary/{data_str}')
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'results/nonbinary/{data_str}/{filename_date}.csv'

    # run experiment
    results = run_experiments(data_str=data_str,
                              disc_measures=disc_measures,
                              methods=methods,
                              n_runs=n_runs)

    # convert results to proper dataframe
    results_df = convert_results_to_dataframe(results)

    # save results
    results_df.to_csv(save_path, index_label='index')

    # plot results
    plot_results(results_df, save_path=save_path)


if __name__ == "__main__":
    main()
