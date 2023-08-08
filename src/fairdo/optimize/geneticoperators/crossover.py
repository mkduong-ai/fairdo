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
    offspring = np.empty((num_offspring, parents.shape[1]))
    for k in range(num_offspring):
        # create a mask with the same shape as a gene
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
    d = parents.shape[1]
    offspring = np.empty((num_offspring, d))

    for i in range(num_offspring):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]

        # Calculate crossover points
        crossover_points = np.sort(np.random.choice(d - 1, k, replace=False)) + 1
        crossover_points = np.concatenate(([0], crossover_points, [d]))

        for j in range(len(crossover_points) - 1):
            start, end = crossover_points[j], crossover_points[j + 1]
            # Choose parent based on j (even or odd)
            parent_idx = parent1_idx if j % 2 == 0 else parent2_idx
            offspring[i, start:end] = parents[parent_idx, start:end]

    return offspring
