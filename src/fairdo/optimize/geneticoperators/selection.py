"""
Selection Methods
=================

This module implements various selection methods used in genetic algorithms.
These methods are used to select the individuals from the current generation that will be used to produce the offspring
for the next generation.
Each function in this module takes a population of individuals and their fitness values as input,
and returns a subset of the population to be used as parents for the next generation and the parents' fitness.
The number of parents to select can be specified by the user.

These selection methods are based on the works of the following references:

Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.

Baker, J. E. (1985). Adaptive selection methods for genetic algorithms.
Proceedings of an International Conference on Genetic Algorithms and their Applications.

Whitley, D. (1989). The GENITOR algorithm and selection pressure: Why rank-based allocation of
reproductive trials is best. Proceedings of the Third International Conference on Genetic Algorithms.
"""

import numpy as np


def elitist_selection(population, fitness, num_parents=2):
    """
    This function selects the fittest (max fitness) parents from the population.

    Parameters
    ----------
    population: ndarray, shape (n, d)
        population of individuals
    fitness: ndarray, shape (n,)
        fitness of each individual
    num_parents: int
        number of parents to select

    Returns
    -------
    parents: ndarray, shape (num_parents, d)
    fitness: ndarray, shape (num_parents,)
    """
    # select the best individuals from the population to be parents
    idx = np.argsort(fitness)
    parents = population[idx[-num_parents:]]
    parents_fitness = fitness[idx[-num_parents:]]
    return parents, parents_fitness


def tournament_selection(population, fitness, num_parents=2, tournament_size=3):
    """
    Select parents using Tournament Selection.
    This method randomly selects a few individuals and chooses the best out of them to become a parent.
    The process is repeated until the desired number of parents are selected.

    Parameters
    ----------
    population: ndarray, shape (n, d)
        Population of individuals.
    fitness: ndarray, shape (n,)
        Fitness of each individual.
    num_parents: int
        Number of parents to select.
    tournament_size: int
        Number of individuals participating in each tournament.

    Returns
    -------
    parents: ndarray, shape (num_parents, d)
        Selected parents.
    fitness: ndarray, shape (num_parents,)
        Fitness of the selected parents.
    """
    if population.shape[0] < tournament_size:
        raise ValueError("Tournament size cannot be larger than the population size.")
    if len(population.shape) != 2:
        population = population.reshape(-1, 1)

    parents = np.empty((num_parents, population.shape[1]))
    parents_fitness = np.empty(num_parents)
    for i in range(num_parents):
        tournament_indices = np.random.randint(0, len(population), size=tournament_size)
        tournament_fitnesses = fitness[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        parents[i, :] = population[winner_index, :]
        parents_fitness[i] = fitness[winner_index]
    return parents, parents_fitness


def roulette_wheel_selection(population, fitness, num_parents=2):
    """
    Select parents using Roulette Wheel Selection. The probability of selecting an individual is proportional to its
    fitness. The higher the fitness, the higher the chance of being selected.
    This function assumes that the fitness is non-negative.

    Parameters
    ----------
    population: ndarray, shape (n, d)
        Population of individuals.
    fitness: ndarray, shape (n,)
        Fitness of each individual.
    num_parents: int
        Number of parents to select.

    Returns
    -------
    parents: ndarray, shape (num_parents, d)
        Selected parents.
    fitness: ndarray, shape (num_parents,)
        Fitness of the selected parents.

    Notes
    -----
    This function normally assumes that the fitness is non-negative.
    However, if the fitness is negative, then the fitness values are shifted to be non-negative.

    References
    ----------
    This function is based on the work:
    Holland, J. H. (1975). Adaptation in natural and artificial systems: An introductory analysis with applications
    to biology, control, and artificial intelligence. The Michigan Press.
    """
    # Check if the fitness is non-negative
    if np.any(fitness < 0):
        # shift the fitness values to be non-negative
        # also prevent the fitness values from being zero to avoid division by zero
        fitness = fitness - 2*np.min(fitness)
    fitness_sum = np.sum(fitness)
    selection_probs = fitness / fitness_sum
    parents_idx = np.random.choice(np.arange(len(population)), size=num_parents, p=selection_probs)
    return population[parents_idx], fitness[parents_idx]


def stochastic_universal_sampling(population, fitness, num_parents=2):
    """
    This function selects parents from the population using the Stochastic Universal Sampling (SUS) method.

    Parameters
    ----------
    population: ndarray, shape (n, d)
        The population of individuals.
    fitness: ndarray, shape (n,)
        The fitness of each individual in the population.
    num_parents: int
        The number of parents to select.

    Returns
    -------
    parents: ndarray, shape (num_parents, d)
        The selected parents.
    fitness: ndarray, shape (num_parents,)
        The fitness of the selected parents.

    Notes
    -----
    This function normally assumes that the fitness is non-negative.
    However, if the fitness is negative, then the fitness values are shifted to be non-negative.

    References
    ----------
    This function is based on the work of Baker (1987) in the following paper:
    Baker, J. E. (1987). "Reducing Bias and Inefficiency in the Selection Algorithm".
    Proceedings of the Second International Conference on Genetic Algorithms and Their Application.
    """
    # Check if the population is a 2D array
    if len(population.shape) != 2:
        population = population.reshape(-1, 1)
    # Check if the fitness is non-negative
    if np.any(fitness < 0):
        # shift the fitness values to be non-negative
        # also prevent the fitness values from being zero to avoid division by zero
        fitness = fitness - 2*np.min(fitness)
    # Normalize the fitness values
    fitness_sum = np.sum(fitness)
    normalized_fitness = fitness / fitness_sum

    # Calculate the distance between the pointers
    distance = 1.0 / num_parents

    # Initialize the start of the pointers
    start = np.random.uniform(0, distance)

    # Initialize the parents
    parents = np.empty((num_parents, population.shape[1]))
    parents_fitness = np.empty(num_parents)
    # Perform the SUS
    for i in range(num_parents):
        pointer = start + i * distance
        sum_fitness = 0
        for j in range(len(normalized_fitness)):
            sum_fitness += normalized_fitness[j]
            if sum_fitness >= pointer:
                parents[i] = population[j]
                parents_fitness[i] = fitness[j]
                break

    return parents, parents_fitness


def rank_selection(population, fitness, num_parents=2):
    """
    This function selects parents from the population based on their rank.
    The rank is determined by the fitness of the individual.
    The higher the fitness, the higher the rank.
    The probability of selecting an individual is proportional to its rank.

    Parameters
    ----------
    population: ndarray, shape (n, d)
        The population of individuals.
    fitness: ndarray, shape (n,)
        The fitness of each individual.
    num_parents: int
        The number of parents to select.

    Returns
    -------
    parents: ndarray, shape (num_parents, d)
        The selected parents.
    fitness: ndarray, shape (num_parents,)
        The fitness of the selected parents.
    """
    # Rank individuals based on fitness
    ranks = np.argsort(np.argsort(fitness))

    # Calculate selection probabilities based on rank
    total_ranks = np.sum(ranks)
    selection_probabilities = ranks / total_ranks

    # Select parents based on selection probabilities
    parent_indices = np.random.choice(np.arange(len(population)), size=num_parents, p=selection_probabilities)

    parents = population[parent_indices]
    fitness = fitness[parent_indices]

    return parents, fitness
