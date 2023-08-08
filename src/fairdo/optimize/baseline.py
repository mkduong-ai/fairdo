import numpy as np


def original_method(f, d):
    """
    Returns a vector of ones and the fitness of that vector.
    This essentially means that the original data is returned without removing any samples.

    Parameters
    ----------
    f : callable
        A function that calculates the fitness of a vector.
    d : int
        The dimension of the vector.

    Returns
    -------
    np.array, float
        A vector of ones and the fitness value of that vector.
    """
    return np.ones(d), f(np.ones(d))


def random_method(f, d, pop_size=50, num_generations=100):
    """
    This function generates a random binary vector and evaluates its performance.
    In a for-loop, it generates a new binary vector and evaluates its performance.
    If the new binary vector performs better than the current one, the current vector is replaced with the new one.
    This process is repeated for a total of ``pop_size * num_generations`` times.

    Parameters
    ----------
    f : callable
        A function that calculates the fitness of a vector.
    d : int
        The dimension of the vector.
    pop_size : int
        The size of the population.
    num_generations : int
        The number of generations.

    Returns
    -------
    np.array, float
        The final solution vector and its fitness value.
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
