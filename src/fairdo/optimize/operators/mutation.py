"""
Mutation Methods
================

This module implements various mutation methods used in genetic algorithms.
These methods are used to introduce diversity in the population by randomly altering the genes of the individuals.
Each function in this module takes a population of individuals as input and returns a mutated population.
**The mutation happens outplace**, meaning that the original population is not modified.
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
    mutation_rate: float, optional (default=0.05)
        The percentage of random bits to flip for each offspring. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    offspring_out = offspring.copy()
    num_mutation = int(mutation_rate * offspring.shape[1])
    for idx in range(offspring.shape[0]):
        # select the random bits to flip
        mutation_bits = np.random.choice(offspring.shape[1],
                                         num_mutation,
                                         replace=False)
        # flip the bits
        offspring_out[idx, mutation_bits] = 1 - offspring_out[idx, mutation_bits]
    return offspring_out


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
    offspring_out = offspring.copy()
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    offspring_out[mutation_mask] = 1 - offspring_out[mutation_mask]
    return offspring_out


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
    offspring_out = offspring.copy()
    for idx in range(offspring.shape[0]):
        # select two random bits
        bit1, bit2 = np.random.choice(offspring.shape[1], 2, replace=False)
        # swap the bits
        offspring_out[idx, bit1], offspring_out[idx, bit2] = offspring_out[idx, bit2], offspring_out[idx, bit1]
    return offspring_out


def adaptive_mutation(offspring, mutation_rate=0.05, diversity_threshold=0.1):
    """
    Mutates the given offspring with an adaptive mutation rate based on population diversity.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The initial probability of flipping each bit for each offspring. Default is 0.05.
    diversity_threshold: float, optional
        The threshold for population diversity. If diversity falls below this threshold, increase mutation rate. Default is 0.1.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    # Calculate population diversity
    population_diversity = np.mean(np.std(offspring, axis=0))
    # Adjust mutation rate based on diversity
    if population_diversity < diversity_threshold:
        mutation_rate *= 2  # Increase mutation rate
    # Apply random mutation
    return bit_flip_mutation(offspring, mutation_rate)


def diverse_mutation(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring in a diverse manner to prevent convergence towards 50% selection rate.
    Using the uniform mutation rate, the mutation rate is adjusted based on the proportion of 1s in each offspring.
    This means that the mutation rate is higher for offspring with a higher proportion of 1s.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The base mutation rate for each bit. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    # Calculate the proportion of 1s in each offspring
    proportion_ones = np.mean(offspring, axis=1)

    # Adjust the mutation rate based on the proportion of 1s
    mutated_proportion_ones = proportion_ones + np.random.uniform(-0.1, 0.1, size=proportion_ones.shape)
    mutated_proportion_ones = np.clip(mutated_proportion_ones, 0, 1)  # Ensure values are within [0, 1] range

    # Generate mutated offspring
    mutation_probs = mutation_rate * mutated_proportion_ones
    mutated_mask = np.random.rand(*offspring.shape) < mutation_probs[:, np.newaxis]
    mutated_offspring = np.where(mutated_mask, 1 - offspring, offspring)

    return mutated_offspring


def shuffle_mutation(offspring, **kwargs):
    """
    Mutates the given offspring by shuffling the bits of each offspring.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    kwargs: dict
        Additional keyword arguments. Ignored.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    offspring_out = offspring.copy()
    rng = np.random.default_rng()
    rng.shuffle(offspring_out, axis=1)
    return offspring_out
