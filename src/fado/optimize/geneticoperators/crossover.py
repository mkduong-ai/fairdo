"""
Crossover Methods
=================

This module implements various crossover methods used in genetic algorithms.
Crossover is a genetic operator to combine two individuals to produce offspring.

Each function in this module takes a pair of parents and produces a user-given amount of
offsprings by combining the genes of the parents.
The way in which the genes (individuals' features) are combined depends on the specific crossover method.

These crossover methods are based on the works of the following references:

Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
"""
import numpy as np


def onepoint_crossover(parents, num_offspring):
    """
    Perform the crossover operation with One-point crossover on the parents to create the offspring.

    Parameters
    ----------
    parents: numpy array
        Parents of the offspring with shape (2, d).
    num_offspring: int
        Number of offsprings.

    Returns
    -------
    offspring: ndarray, shape (num_offspring, d)
    """
    # perform one-point crossover on the parents to generate new offspring
    return kpoint_crossover(parents, num_offspring, k=1)


def uniform_crossover(parents, num_offspring, p=0.5):
    """
    Perform the crossover operation with Uniform crossover on the parents to create the offspring.

    Parameters
    ----------
    parents: numpy array
        Parents of the offspring with shape (2, d).
    num_offspring: int
        Number of offsprings.
    p: float
        Probability of selecting a gene from the first parent, default is `0.5`.

    Returns
    -------
    offspring: ndarray, shape (num_offspring, d)
    """
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(num_offspring)
    for k in range(num_offspring):
        # create a mask with the same shape as a gene
        if p is None:
            # if p is not specified, randomly choose a probability for each gene
            p = np.random.uniform()
        mask = np.random.uniform(size=parents.shape[1]) < p
        # assign genes to the offspring based on the mask
        offspring[k] = np.where(mask, parents[0], parents[1])
    return offspring


def kpoint_crossover(parents, num_offspring, k=2):
    """
    Perform the crossover operation with K-point crossover on the parents to create the offspring.

    Parameters
    ----------
    parents: numpy array
        Parents of the offspring with shape (2, d).
    num_offspring: int
        Number of offsprings.
    k: int
        number of crossover points, default is 2

    Returns
    -------
    offspring: ndarray, shape (num_offspring, d)
    """
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(num_offspring)
    for i in range(num_offspring):
        # parent selection
        # switch between the two parents randomly at each crossover point to create the offspring
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        # calculate the crossover points
        crossover_points = sorted(np.random.choice(range(1, parents.shape[1]), k, replace=False))
        # alternate between parents in each segment
        segments = np.array_split(range(parents.shape[1]), crossover_points)
        for j in range(len(segments)):
            if j % 2 == 0:
                offspring[i, segments[j]] = parents[parent1_idx, segments[j]]
            else:
                offspring[i, segments[j]] = parents[parent2_idx, segments[j]]
    return offspring
