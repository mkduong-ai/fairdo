import numpy as np


def select_parents(population, fitness, num_parents=2):
    """
    In this example, the select_parents function is used to select the parents for the next generation.
    This function selects the fittest parents from the population.

    Parameters
    ----------
    population: numpy array
        population of individuals
    fitness: numpy array
        fitness of each individual
    num_parents: int
        number of parents to select

    Returns
    -------
    parents: numpy array
    """
    # select the best individuals from the population to be parents
    idx = np.argsort(fitness)
    parents = population[idx[:num_parents]]
    fitness = fitness[idx[:num_parents]]
    return parents, fitness