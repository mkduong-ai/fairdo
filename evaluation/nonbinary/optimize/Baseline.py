import numpy as np


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
