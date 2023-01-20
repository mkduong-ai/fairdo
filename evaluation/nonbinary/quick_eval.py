import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from fado.metrics import statistical_parity_absolute_difference, normalized_mutual_information
from fado.preprocessing import MetricOptimizer, MetricOptRemover


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


def metricopt_wrapper(dataframe, label, protected_attributes, disc_measure=statistical_parity_absolute_difference):
    preproc = MetricOptRemover(frac=0.75,
                               protected_attribute=protected_attributes[0],
                               label=label,
                               fairness_metric=disc_measure)

    preproc = preproc.fit(dataframe)
    return preproc.transform()


def method_random(f, dims):
    """

    Parameters
    ----------
    f: callable
    dims: int

    Returns
    -------

    """
    current_solution = np.random.randint(0, 2, size=dims)
    current_fitness = f(current_solution)
    for i in range(100):
        new_solution = np.random.randint(0, 2, size=dims)
        new_fitness = f(new_solution)
        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    return current_solution, current_fitness


def keep_xyz_samples(x, y, z, binary_vector):
    assert len(x) == len(y) == len(z), "x, y, z must have the same length"
    assert len(binary_vector) == len(x), "Binary vector must have the same length as x, y, z"
    # Convert binary_vector to a boolean mask
    mask = np.array(binary_vector) == 1
    # Use the boolean mask to filter x, y, z
    return x[mask], y[mask], z[mask]


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

    x, y, z = keep_xyz_samples(x, y, z, binary_vector)

    # TODO: handle multiple protected attributes (?)
    y = y.to_numpy().flatten()
    z = z.to_numpy().flatten()
    return disc_measure(x=x, y=y, z=z)


def plot(results):
    # Plot the mean and standard deviation of the results using Matplotlib
    for method, method_results in results.items():
        for func, values in method_results.items():
            mean = np.mean(values)
            std = np.std(values)
            plt.scatter(f"{method} {func}", mean, label=f"{method} {func}")
            plt.errorbar(f"{method} {func}", mean, yerr=std, fmt='o')
            plt.xlabel('Method and Function')
            plt.ylabel('Minimum Value')
            plt.legend()
    plt.show()


def main():
    # settings
    df, label, protected_attributes = load_data('adult')
    # create objective function
    f_obj = lambda x, disc_measure: f(x, dataframe=df, label=label, protected_attributes=protected_attributes,
                                      disc_measure=disc_measure)
    disc_measures = [statistical_parity_absolute_difference, normalized_mutual_information]
    # Todo: function name always lambda...
    functions = [lambda x: f_obj(x, disc_measure=disc_measure) for disc_measure in disc_measures]

    dims = len(df) # number of dimensions of x
    n_runs = 3 # number of times to run each method
    methods = [method_random]#, method2, method3]

    # create results dictionary
    results = {}
    for method in methods:
        results[method.__name__] = {}
        for func in functions:
            results[method.__name__][func.__name__] = []

    # run experiments and save to results
    for method in methods:
        for func in functions:
            for i in range(n_runs):
                results[method.__name__][func.__name__].append(method(func, dims)[1])

    print(results)
    print('Plotting results...')
    plot(results)


if __name__ == "__main__":
    main()
