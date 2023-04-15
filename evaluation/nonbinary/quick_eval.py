import time
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# generate synthetic datapoints
from sdv.tabular import GaussianCopula

# load non-binary evaluation
from objectives import f_remove, f_add

# load metrics
from fado.metrics.nonbinary import nb_statistical_parity_sum_abs_difference,\
    nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information

# load fake metrics that serve other purposes
from measures import count_size, count_groups, sanity_check

# load fado library
# from fado.preprocessing import MetricOptimizer, MetricOptRemover

# load optimization methods
from optimize import Baseline, SimulatedAnnealing, GeneticAlgorithm


def downcast(data):
    """
    Downcast float and integer columns to save memory.
    Parameters
    ----------
    data: pandas DataFrame

    Returns
    -------
    data: pandas DataFrame
    """
    fcols = data.select_dtypes('float').columns
    icols = data.select_dtypes('integer').columns

    data[fcols] = data[fcols].apply(pd.to_numeric, downcast='float')
    data[icols] = data[icols].apply(pd.to_numeric, downcast='integer')

    return data


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
        print(data[protected_attributes].iloc[:, 0].unique())
        print(data[protected_attributes].iloc[:, 0].value_counts())
        cols_to_labelencode = protected_attributes.copy()
        cols_to_labelencode.append(label)
        data[cols_to_labelencode] = \
            data[cols_to_labelencode].apply(LabelEncoder().fit_transform)
        # one-hot encoding categorical columns
        categorical_cols = list(data.select_dtypes(include='object'))
        data = pd.get_dummies(data, columns=categorical_cols)
        # downcast
        data = downcast(data)

        return data, label, protected_attributes
    elif dataset_str == 'compas':
        use_cols = ['race', 'priors_count', 'age_cat', 'c_charge_degree', 'two_year_recid']
        data = pd.read_csv(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            usecols=use_cols)
        print('Data downloaded.')
        # drop rows with missing values
        data = data.dropna(axis=0, how='any')
        # label encoding protected_attribute and label
        label = 'two_year_recid'
        protected_attributes = ['race']
        print(data[protected_attributes].iloc[:, 0].unique())
        print(data[protected_attributes].iloc[:, 0].value_counts())
        cols_to_labelencode = protected_attributes.copy()
        cols_to_labelencode.append(label)
        data[cols_to_labelencode] = \
            data[cols_to_labelencode].apply(LabelEncoder().fit_transform)
        # one-hot encoding categorical columns
        categorical_cols = list(data.select_dtypes(include='object'))
        data = pd.get_dummies(data, columns=categorical_cols)
        # downcast
        data = downcast(data)

        return data, label, protected_attributes


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
        f = lambda binary_vector, dataframe, label, protected_attributes, disc_measure: \
            f_add(binary_vector, dataframe=dataframe, label=label, protected_attributes=protected_attributes,
                  disc_measure=disc_measure, synthetic_dataframe=df_syn)
        dims = df_syn.shape[0]
    else:
        raise ValueError(f'Objective {objective_str} not supported.')

    # create objective function
    disc_measures = list(disc_dict.values())
    f_obj = lambda x, disc_measure: f(x, dataframe=df, label=label, protected_attributes=protected_attributes,
                                      disc_measure=disc_measure)
    functions = [lambda x, disc_measure=disc_measure: f_obj(x, disc_measure=disc_measure)
                 for disc_measure in disc_measures]
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


def settings(data_str, objective_str):
    # objective_str = 'add'
    # data_str = 'compas'
    n_runs = 15
    # create objective functions
    disc_dict = {
        'Statistical Disparity Sum': nb_statistical_parity_sum_abs_difference,
        'Maximal Statistical Disparity': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information,
        'Size': count_size,
        'Distinct Groups': count_groups,
        'Sanity Check': sanity_check}
    # create methods
    methods = {'Baseline (Original)': Baseline.method_original,
               'Random Heuristic': Baseline.method_random,
               'Simulated Annealing': SimulatedAnnealing.simulated_annealing_method,
               'GA (1-Point Crossover)': GeneticAlgorithm.genetic_algorithm_method,
               'GA (Uniform Crossover)': GeneticAlgorithm.genetic_algorithm_uniform_method,
               # 'Metric Optimizer': MetricOptimizer.metric_optimizer_remover}
               }

    # create save path
    save_path = f'results/nonbinary/{data_str}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # filename_date = 'results'
    #save_path = f'{save_path}/{filename_date}'
    save_path = f'{save_path}/{data_str}_{objective_str}'

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

    # plot results
    # plot_results(results_df=results_df,
    #              disc_dict=disc_dict,
    #              save_path=save_path)


def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas']
    for data_str in data_strs:
        print('------------------------------------')
        print(f'Running experiments for {data_str}...')
        for obj_str in obj_strs:
            print('------------------------------------')
            print(f'Running experiments for {obj_str}...')
            settings(data_str=data_str, objective_str=obj_str)


if __name__ == "__main__":
    main()
