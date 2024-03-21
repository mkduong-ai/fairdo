import numpy as np


def random_initialization(pop_size, d):
    """
    Generate a random population of binary vectors. Each vector has a length of d.
    The values of the vectors are either 0 or 1. The population is generated randomly.

    Parameters
    ----------
    pop_size: int
        The size of the population to generate.
    d: int
        The dimension of the binary vectors.

    Returns
    -------
    population: ndarray, shape (pop_size, d)
        The generated population of binary vectors.
    """
    return biased_random_initialization(pop_size, d, selection_probability=0.5)


def biased_random_initialization(pop_size, d, selection_probability=0.8):
    """
    Initialize the population with a bias towards selecting more items.

    Parameters
    ----------
    pop_size: int
        Size of the population.
    d: int
        Dimensionality of the problem (number of items).
    selection_probability: float
        Probability of initializing a bit as 1.

    Returns:
        np.ndarray: Initialized population with shape (pop_size, d).
    """
    population = np.random.choice([0, 1], size=(pop_size, d), p=[1 - selection_probability, selection_probability])
    return population


def variable_initialization(pop_size, d, min_p=0.5, max_p=0.99):
    """
    Initialize the population with a variable probability of selecting items.

    Parameters
    ----------
    pop_size: int
        Size of the population.
    d: int
        Dimensionality of the problem (number of items).
    min_p: float
        Minimum probability of selecting an item.
    max_p: float
        Maximum probability of selecting an item.

    Returns:
        np.ndarray: Initialized population with shape (pop_size, d).
    """
    probabilities = np.linspace(min_p, max_p, num=pop_size)
    population = np.array([np.random.choice([1, 0], size=d, p=[p, 1-p]) for p in probabilities])
    return population
