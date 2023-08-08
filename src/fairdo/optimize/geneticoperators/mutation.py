"""
Mutation Methods
================

This module implements various mutation methods used in genetic algorithms.
These methods are used to introduce diversity in the population by randomly altering the genes of the individuals.
Each function in this module takes a population of individuals as input and returns a mutated population.
The rate of mutation can be specified by the user.

These mutation methods are based on the works of the following references:

Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.

Bäck, T. (1993). Optimal Mutation Rates in Genetic Search.
Proceedings of the 5th International Conference on Genetic Algorithms.
"""
import numpy as np


def fractional_flip_mutation(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring by flipping a percentage of random bits for each offspring.
    A fixed amount of bits is flipped for each offspring.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The percentage of random bits to flip for each offspring. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    num_mutation = int(mutation_rate * offspring.shape[1])
    for idx in range(offspring.shape[0]):
        # select the random bits to flip
        mutation_bits = np.random.choice(np.arange(offspring.shape[1]),
                                         num_mutation,
                                         replace=False)
        # flip the bits
        offspring[idx, mutation_bits] = 1 - offspring[idx, mutation_bits]
    return offspring


def bit_flip_mutation(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring by flipping each bit with a certain probability.
    Some offspring may not be mutated at all, and some may be mutated more than expected.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The probability of flipping each bit. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    offspring[mutation_mask] = 1 - offspring[mutation_mask]
    return offspring


def swap_mutation(offspring):
    """
    Mutates the given offspring by randomly selecting two bits and swapping their values.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    for idx in range(offspring.shape[0]):
        # select two random bits
        bit1, bit2 = np.random.choice(np.arange(offspring.shape[1]), 2, replace=False)
        # swap the bits
        offspring[idx, bit1], offspring[idx, bit2] = offspring[idx, bit2], offspring[idx, bit1]
    return offspring

