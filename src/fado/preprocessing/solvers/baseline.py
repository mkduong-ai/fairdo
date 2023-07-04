import numpy as np


def original_method(f, d):
    """
    Returns a vector of ones and the fitness of that vector.
    This essentially means that the original data is returned without removing any samples.

    Parameters
    ----------
    f: callable
    d: int

    Returns
    -------

    """
    return np.ones(d), f(np.ones(d))


def random_method(f, d, pop_size=50, num_generations=100):
    """
    Generate a random solution (binary vector) and evaluate it.
    In to for-loop, generate a new solution and evaluate it.
    If the new solution is better than the current solution,
    replace the current solution with the new solution.
    Repeat this process for pop_size * num_generations times.

    Parameters
    ----------
    f: callable
    d: int
    pop_size: int
        population size
    num_generations: int
        number of generations

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
