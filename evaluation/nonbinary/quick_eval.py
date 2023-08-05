# Standard library imports
import datetime
import itertools
import os
import time
from functools import partial

# Related third-party imports
from sklearn.preprocessing import LabelEncoder
from sdv.tabular import GaussianCopula
import pandas as pd

# fado imports
from fado.utils.dataset import load_data
from fado.metrics import (normalized_mutual_information,
                          statistical_parity_abs_diff,
                          statistical_parity_abs_diff_max,
                          statistical_parity_abs_diff_mean)
from fado.preprocessing.solverwrapper import f_add, f_remove
# from fado.preprocessing import MetricOptimizer, MetricOptRemover
import fado.optimize.baseline as baseline
import fado.optimize.geneticalgorithm as ga
from fado.optimize.geneticoperators import *

# Local application/library specific imports
from measures import count_groups, count_size, sanity_check


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


def create_save_path(data_str, objective_str, prefix='test'):
    """
    Creates the save path for the experiment results.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.
    prefix: str
        Prefix for the save path.
    Returns
    -------
    save_path: str
        The save path for the experiment results.
    """

    # create save path
    save_path = f'evaluation/results/nonbinary/{prefix}/{data_str}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'{save_path}/{data_str}_{objective_str}'

    return save_path


def run_experiment(data_str, disc_dict, methods,
                   objective_str='remove'):
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
    objective_str: str
        objective function

    Returns
    -------
    results: dict
        results['method']['objective/time/func_values']
    """
    # settings
    # TODO: add test data?
    df, label, protected_attributes = load_data(data_str)
    num_synthetic_data = df.shape[0]
    print('Successfully loaded data.')

    # create task
    if objective_str in ['remove', 'synthetic', 'remove_and_synthetic']:
        if objective_str == 'synthetic':
            # create synthetic data
            gc = GaussianCopula()
            gc.fit(df)
            df_syn = gc.sample(num_synthetic_data)
            df = df_syn
        if objective_str == 'remove_and_synthetic':
            # create synthetic data
            gc = GaussianCopula()
            gc.fit(df)
            df_syn = gc.sample(num_synthetic_data)
            df = pd.concat([df, df_syn], axis=0)

        # create objective function
        f = f_remove
        dims = df.shape[0]
    elif objective_str == 'add':
        # create synthetic data
        gc = GaussianCopula()
        gc.fit(df)
        df_syn = gc.sample(num_synthetic_data)

        # create objective function
        f = partial(f_add, sample_dataframe=df_syn)
        dims = df_syn.shape[0]
    else:
        raise ValueError(f'Objective {objective_str} not supported.')

    # create objective function
    disc_measures = list(disc_dict.values())
    functions = [partial(f, dataframe=df, label=label, protected_attributes=protected_attributes,
                         disc_measure=disc_measure) for disc_measure in disc_measures]
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
            results[method_name][func.__name__] = method(f=func, d=dims)[1]
            end_time = time.time()
            results[method_name]['time' + f'_{func.__name__}'] = end_time - start_time

        # information about the data
        results[method_name]['data'] = data_str
        results[method_name]['label'] = label
        results[method_name]['protected_attributes'] = protected_attributes

    return results


def run_experiments(data_str, disc_dict, methods, n_runs=10,
                    objective_str='remove'):
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
    objective_str: str
        objective function

    Returns
    -------
    results: dict
        results['run']['method']['objective/time/func_values']
    """
    results = {}
    for i in range(n_runs):
        print(f'Run {i + 1} of {n_runs}')
        results[i] = run_experiment(data_str=data_str, disc_dict=disc_dict, methods=methods,
                                    objective_str=objective_str)
    return results


def setup_experiment(data_str, objective_str):
    """
    Sets up the experiment.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.

    Returns
    -------
    save_path: str
        The save path for the experiment results.
    disc_dict: dict
        Dictionary of discrimination measures.
    methods: dict
        Dictionary of methods.
    """

    # create objective functions
    disc_dict = {
        'Statistical Disparity Sum': statistical_parity_abs_diff,
        # 'Maximal Statistical Disparity': statistical_parity_abs_diff_max,
        # 'NMI': normalized_mutual_information,
        # 'Size': count_size,
        # 'Distinct Groups': count_groups,
        'Sanity Check': sanity_check}

    # create methods
    methods = {  # 'Baseline (Original)': baseline.original_method,
        # 'Random Heuristic': baseline.random_method,
        'GA (1-Point Crossover)': partial(ga.genetic_algorithm,
                                          crossover=onepoint_crossover, num_generations=3),
        'GA (3-Point Crossover)': partial(ga.genetic_algorithm,
                                          crossover=kpoint_crossover, num_generations=3),
        # 'GA (Uniform Crossover)': genetic_algorithm_method_wrapper('uniform')
    }

    # create save path
    save_path = create_save_path(data_str, objective_str)

    return save_path, disc_dict, methods


def setup_experiment_hyperparameter(data_str, objective_str):
    """
    Sets up the experiment.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.

    Returns
    -------
    save_path: str
        The save path for the experiment results.
    disc_dict: dict
        Dictionary of discrimination measures.
    methods: dict
        Dictionary of methods.
    """

    # create objective functions
    disc_dict = {
        'Statistical Disparity Sum': statistical_parity_abs_diff,
        # 'Maximal Statistical Disparity': statistical_parity_abs_diff_max,
        # 'NMI': normalized_mutual_information,
        # 'Size': count_size,
        # 'Distinct Groups': count_groups,
        # 'Sanity Check': sanity_check
    }

    # create hyperparameters
    hyperparams = {'pop_size': [20, 50, 100, 200],
                   'num_generations': [50, 100, 200, 500],
                   # 'selection': [elitist_selection, roulette_wheel_selection],
                   # 'crossover': [onepoint_crossover, kpoint_crossover, uniform_crossover],
                   # 'mutation': [bit_flip_mutation, swap_mutation]
                   }

    # create methods
    methods = {}

    # Use itertools.product to generate all combinations of hyperparameters
    for combination in itertools.product(*hyperparams.values()):
        # Create a dictionary of the current combination of hyperparameters
        params = dict(zip(hyperparams.keys(), combination))
        # Create the method name
        name = 'GA (' + ', '.join(f'{k}={v.__name__ if callable(v) else v}' for k, v in params.items()) + ')'
        # Create the method and add it to the methods dictionary
        methods[name] = partial(ga.genetic_algorithm, **params)

    # create save path
    save_path = create_save_path(data_str, objective_str, prefix='hyperparameter')

    return save_path, disc_dict, methods


def run_and_save_experiment(data_str, objective_str, n_runs=10):
    """
    Runs the experiment and saves the results.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.
    n_runs: int
        Number of runs.
    """

    # setup the experiment
    save_path, disc_dict, methods = setup_experiment_hyperparameter(data_str, objective_str)

    # run experiment
    results = run_experiments(data_str=data_str,
                              disc_dict=disc_dict,
                              methods=methods,
                              n_runs=n_runs,
                              objective_str=objective_str)

    # convert results to proper dataframe
    results_df = convert_results_to_dataframe(results)

    # save results
    results_df.to_csv(save_path + '.csv', index_label='index')


def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas', 'bank']
    # Experiments
    #obj_strs = ['remove']
    #data_strs = ['compas']

    n_runs = 15
    for data_str in data_strs:
        print('------------------------------------')
        print(f'Running experiments for {data_str}...')
        for obj_str in obj_strs:
            print('------------------------------------')
            print(f'Running experiments for {obj_str}...')
            run_and_save_experiment(data_str=data_str, objective_str=obj_str, n_runs=n_runs)


if __name__ == "__main__":
    main()
