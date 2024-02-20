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


def random_method(f, d, pop_size=100, num_generations=500):
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
    best_solution = np.random.randint(0, 2, size=d)
    best_fitness = f(best_solution)
    for _ in range(pop_size * num_generations):
        new_solution = np.random.randint(0, 2, size=d)
        new_fitness = f(new_solution)
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness
    return best_solution, best_fitness


def random_method_vectorized(f, d, pop_size=100, num_generations=500):
    """
    This function is not essentially faster than the original function but requires more memory.

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
    solutions = np.random.randint(0, 2, size=(pop_size * num_generations, d))
    fitness_values = np.apply_along_axis(f, 1, solutions)

    best_index = np.argmin(fitness_values)
    best_solution = solutions[best_index]
    best_fitness = fitness_values[best_index]

    return best_solution, best_fitness
