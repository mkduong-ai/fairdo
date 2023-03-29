import time
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load metrics
from create_plots import plot_results
from fado.metrics import statistical_parity_absolute_difference
from fado.metrics.nonbinary import nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information

# load fado library
# from fado.preprocessing import MetricOptimizer, MetricOptRemover

# load optimization methods
from optimize import Baseline


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
        data[cols_to_labelencode] = \
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


def run_experiment(data_str, disc_dict, methods):
    """
    Runs the experiment
    Parameters
    ----------
    data_str: str
        name of the dataset
    disc_dict: dict
        dictionary of discrimination measures
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
    disc_measures = list(disc_dict.values())
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
            results[method_name]['time' + f'_{func.__name__}'] = end_time - start_time

        # information about the data
        results[method_name]['data'] = data_str
        results[method_name]['label'] = label
        results[method_name]['protected_attributes'] = protected_attributes

    return results


def run_experiments(data_str, disc_dict, methods, n_runs=10):
    """
    Runs the experiments for n_runs times
    Parameters
    ----------
    data_str: str
        name of the dataset
    disc_dict: dict
        dictionary of discrimination measures
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
        print(f'Run {i + 1} of {n_runs}')
        results[i] = run_experiment(data_str=data_str, disc_dict=disc_dict, methods=methods)
    return results


def main():
    data_str = 'adult'
    n_runs = 2
    # create objective functions
    disc_dict = {  # 'Absolute Statistical Disparity': statistical_parity_absolute_difference,
        # 'Absolute Statistical Disparity Sum (non-binary)': nb_statistical_parity_sum_abs_difference,
        'Abs Statistical Disparity Max (non-binary)': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information}
    # create methods
    methods = {'Baseline (Original)': Baseline.method_original,
               'Baseline (Random)': Baseline.method_random,
               # 'Simulated Annealing': SimulatedAnnealing.simulated_annealing_method,
               # 'Genetic Algorithm': GeneticAlgorithm.genetic_algorithm_method,
               # 'Metric Optimizer': MetricOptimizer.metric_optimizer_remover}
               }

    # create save path
    save_path = f'results/nonbinary/{data_str}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename_date = 'results'
    save_path = f'{save_path}/{filename_date}'

    # run experiment
    results = run_experiments(data_str=data_str,
                              disc_dict=disc_dict,
                              methods=methods,
                              n_runs=n_runs)

    # convert results to proper dataframe
    results_df = convert_results_to_dataframe(results)

    # save results
    results_df.to_csv(save_path + '.csv', index_label='index')

    # plot results
    # plot_results(results_df=results_df,
    #              disc_dict=disc_dict,
    #              save_path=save_path)


if __name__ == "__main__":
    main()
