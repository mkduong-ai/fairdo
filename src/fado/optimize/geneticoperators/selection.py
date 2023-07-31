import numpy as np


def select_parents(population, fitness, num_parents=2):
    """
    This function selects the fittest parents from the population.

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
    parents = population[idx[:num_parents]]
    fitness = fitness[idx[:num_parents]]
    return parents, fitness


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
    """
    fitness_sum = np.sum(fitness)
    selection_probs = fitness / fitness_sum
    parents_idx = np.random.choice(np.arange(len(population)), size=num_parents, p=selection_probs)
    return population[parents_idx]


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
    """
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        tournament_indices = np.random.randint(0, len(population), size=tournament_size)
        tournament_fitnesses = fitness[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        parents[i, :] = population[winner_index, :]
    return parents