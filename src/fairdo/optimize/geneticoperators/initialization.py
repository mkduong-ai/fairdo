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


def variable_probability_initialization(pop_size, d, initial_probability=0.99, min_probability=0.75):
    """
    Initialize the population with a variable probability of selecting items.

    Parameters
    ----------
    pop_size: int
        Size of the population.
    d: int
        Dimensionality of the problem (number of items).
    initial_probability: float
        Initial probability of selecting an item.
    min_probability: float
        Minimum probability of selecting an item.

    Returns:
        np.ndarray: Initialized population with shape (pop_size, d).
    """
    probabilities = np.linspace(initial_probability, min_probability, num=pop_size)
    population = np.array([np.random.choice([0, 1], size=d, p=[1 - p, p]) for p in probabilities])
    return population