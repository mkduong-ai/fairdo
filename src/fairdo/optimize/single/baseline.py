import numpy as np


def ones_array_method(f, d):
    """
    Returns an array of ones and the fitness of that array.
    When used as a binary mask, this method returns the original dataset.

    Parameters
    ----------
    f : callable
        Objective/fitness function to minimize.
    d : int
        Dimension of the flattened numpy array to evaluate on ``f``.

    Returns
    -------
    np.array (d,)
        Numpy array of ones.
    float
        Fitness of the ones array.

    Examples
    --------
    >>> from fairdo.optimize.single import ones_array_method
    >>> ones_array_method(lambda x: x.sum(), 5)
    (array([1., 1., 1., 1., 1.]), 5.0)

    >>> ones_array_method(lambda x: x.sum(), 3)
    (array([1., 1., 1.]), 3.0)
    """
    return np.ones(d), f(np.ones(d))


def random_method(f, d, pop_size=100, num_generations=500):
    """
    Generates a random binary vector (numpy array) and evaluates it on ``f``
    for a total of ``pop_size * num_generations`` times.
    Returns solution with the lowest value.

    Parameters
    ----------
    f : callable
        Objective/fitness function to minimize.
    d : int
        Dimension of the vector.
    pop_size : int
        Size of the population.
    num_generations : int
        Number of generations.

    Returns
    -------
    np.array (d,)
        Numpy array of the best solution found.
    float
        Fitness of the best solution found.

    Examples
    --------
    >>> from fairdo.optimize.single import random_method
    >>> random_method(lambda x: x.sum(), 5)
    (array([0, 0, 0, 0, 0]), 0.0)

    >>> random_method(lambda x: x.sum(), 3)
    (array([0, 0, 0]), 0.0)

    >>> random_method(lambda x: x.sum(), 10)
    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.0)
    """
    best_solution = np.random.randint(2, size=d)
    best_fitness = f(best_solution)
    for _ in range(pop_size * num_generations):
        new_solution = np.random.randint(2, size=d)
        new_fitness = f(new_solution)
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness
    return best_solution, best_fitness


def random_method_vectorized(f, d, pop_size=100, num_generations=500):
    """
    Vectorized version of the ``fairdo.optimize.single.random_method`` function.

    Generates a random binary vector (numpy array) and evaluates it on ``f``
    for a total of ``pop_size * num_generations`` times.
    Returns solution with the lowest value.

    Parameters
    ----------
    f : callable
        Objective/fitness function to minimize.
    d : int
        Dimension of the vector.
    pop_size : int
        Size of the population.
    num_generations : int
        Number of generations.

    Returns
    -------
    np.array (d,)
        Numpy array of the best solution found.
    float
        Fitness of the best solution found.

    Examples
    --------
    >>> from fairdo.optimize.single import random_method
    >>> random_method(lambda x: x.sum(), 5)
    (array([0, 0, 0, 0, 0]), 0.0)

    >>> random_method(lambda x: x.sum(), 3)
    (array([0, 0, 0]), 0.0)

    >>> random_method(lambda x: x.sum(), 10)
    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.0)
    """
    solutions = np.random.randint(2, size=(pop_size * num_generations, d))
    fitness_values = np.apply_along_axis(f, 1, solutions)

    best_index = np.argmin(fitness_values)
    best_solution = solutions[best_index]
    best_fitness = fitness_values[best_index]

    return best_solution, best_fitness
