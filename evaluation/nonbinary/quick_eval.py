import time
import datetime
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# load metrics
from fado.metrics import statistical_parity_absolute_difference, normalized_mutual_information
from fado.metrics.nonbinary import nb_statistical_parity_sum_abs_difference, nb_statistical_parity_max_abs_difference, \
    nb_normalized_mutual_information

# load methods
# from fado.preprocessing import MetricOptimizer, MetricOptRemover

# load optimization methods
from optimize import SimulatedAnnealing, GeneticAlgorithm, MetricOptimizer


def load_data(dataset_str):
    """

    Parameters
    ----------
    dataset_str: str

    Returns
    -------
    df, label, protected_attribute
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


# def metricopt_wrapper(dataframe, label, protected_attributes, disc_measure=statistical_parity_absolute_difference):
#     preproc = MetricOptRemover(frac=0.75,
#                                protected_attribute=protected_attributes[0],
#                                label=label,
#                                fairness_metric=disc_measure)
#
#     preproc = preproc.fit(dataframe)
#     return preproc.transform()


def method_original(f, d):
    """

    Parameters
    ----------
    f: callable
    d: int

    Returns
    -------

    """
    return np.ones(d), f(np.ones(d))


def method_random(f, d, pop_size=50, num_generations=100):
    """

    Parameters
    ----------
    f: callable
    d: int

    Returns
    -------

    """
    current_solution = np.random.randint(0, 2, size=d)
    current_fitness = f(current_solution)
    for i in range(pop_size * num_generations):
        new_solution = np.random.randint(0, 2, size=d)
        new_fitness = f(new_solution)
        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    return current_solution, current_fitness


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


def save_results(results, save_path):
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Objective', 'Value', 'Time'])
        for method, method_results in results.items():
            for func, results in method_results.items():
                for value in results['func_values']:
                    writer.writerow([method, func, value])


def plot_results(results, save_path=None):
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
        plt.savefig(save_path)

    plt.show()


def main():
    # settings
    df, label, protected_attributes = load_data('adult')
    print('Successfully loaded data.')

    # create objective function
    f_obj = lambda x, disc_measure: f(x, dataframe=df, label=label, protected_attributes=protected_attributes,
                                      disc_measure=disc_measure)
    disc_measures = [#nb_statistical_parity_sum_abs_difference,
                     # nb_statistical_parity_max_abs_difference,
                     nb_normalized_mutual_information]
    functions = [lambda x: f_obj(x, disc_measure=disc_measure) for disc_measure in disc_measures]
    for func, disc_measure in zip(functions, disc_measures):
        func.__name__ = disc_measure.__name__

    dims = len(df) # number of dimensions of x
    n_runs = 1 # number of times to run each method
    methods = [#method_random,
               # method_original,
               # SimulatedAnnealing.simulated_annealing_method,
               #GeneticAlgorithm.genetic_algorithm_method,
               MetricOptimizer.metric_optimizer_remover]

    # create results dictionary
    results = {}
    for method in methods:
        results[method.__name__] = {}
        for func in functions:
            results[method.__name__][func.__name__] = {}
            results[method.__name__][func.__name__]['func_values'] = []
            results[method.__name__][func.__name__]['elapsed_time'] = []

    # run experiments and save to results
    print('Running experiments...')
    for method in methods:
        for func in functions:
            for i in range(n_runs):
                print(method.__name__, func.__name__, i)
                start_time = time.time()
                results[method.__name__][func.__name__]['func_values'].append(method(f=func, d=dims)[1])
                results[method.__name__][func.__name__]['elapsed_time'].append(time.time() - start_time)
    print('Done.')

    print(results)

    print('Saving results...')
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_results(results, save_path=f"results_{filename_date}.csv")

    print('Plotting results...')
    plot_results(results, save_path=f"results_{filename_date}.pdf")


if __name__ == "__main__":
    main()
