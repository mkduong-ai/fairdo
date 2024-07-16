"""
Baseline methods for single-objective optimization.
"""

import numpy as np
import warnings


def ones_array_method(f, d, *args, **kwargs):
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


def random_bits_method(f, d, pop_size=100, num_generations=500, *args, **kwargs):
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
    >>> from fairdo.optimize.single import random_bits_method
    >>> random_bits_method(lambda x: x.sum(), 5)
    (array([0, 0, 0, 0, 0]), 0.0)

    >>> random_bits_method(lambda x: x.sum(), 3)
    (array([0, 0, 0]), 0.0)

    >>> random_bits_method(lambda x: x.sum(), 10)
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


def random_bits_method_vectorized(f, d, pop_size=100, num_generations=500, *args, **kwargs):
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
    >>> from fairdo.optimize.single import random_bits_method_vectorized
    >>> random_bits_method_vectorized(lambda x: x.sum(), 5)
    (array([0, 0, 0, 0, 0]), 0.0)

    >>> random_bits_method_vectorized(lambda x: x.sum(), 3)
    (array([0, 0, 0]), 0.0)

    >>> random_bits_method_vectorized(lambda x: x.sum(), 10)
    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.0)
    """
    solutions = np.random.randint(2, size=(pop_size * num_generations, d))
    fitness_values = np.apply_along_axis(f, 1, solutions)

    best_index = np.argmin(fitness_values)
    best_solution = solutions[best_index]
    best_fitness = fitness_values[best_index]

    return best_solution, best_fitness


def brute_force(f, d, pop_size=None, num_generations=None, *args, **kwargs):
    """
    Evaluates all possible binary vectors of length ``d`` and returns the one with the lowest fitness.

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
    >>> from fairdo.optimize.single import brute_force
    >>> brute_force(lambda x: x.sum(), 5)
    (array([0, 0, 0, 0, 1]), 1.0)

    >>> brute_force(lambda x: x.sum(), 3)
    (array([0, 0, 1]), 1.0)

    >>> brute_force(lambda x: x.sum(), 10)
    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 1.0)
    """
    if d > 20:
        warnings.warn("This function is not efficient for d > 20. Consider using other methods.")

    end = 2 ** d
    if pop_size is not None and num_generations is not None:
        warnings.warn(f"Running for pop_size * num_generations iterations instead of {2**d} iterations.")
        end = pop_size * num_generations

    best_solution = np.zeros(d)
    best_fitness = f(best_solution)
    for i in range(1, end):
        solution = np.array([int(x) for x in list(bin(i)[2:].zfill(d))])
        fitness = f(solution)
        if fitness < best_fitness:
            best_solution = solution
            best_fitness = fitness
    return best_solution, best_fitness
