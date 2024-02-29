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


def simulated_binary_crossover(parents, num_offspring, eta=15):
    """
    Perform the crossover operation with Simulated Binary Crossover (SBX) on the parents to create the offspring.
    
    Parameters
    ----------
    parents: numpy array
        Parents of the offspring with shape (2, d).
    num_offspring: int
        Number of offsprings.
    
    Returns
    -------
    offspring: ndarray, shape (num_offspring, d)

    References
    ----------
    Kalyanmoy Deb, Karthik Sindhya, and Tatsuya Okabe. Self-adaptive simulated binary crossover for real-parameter optimization. In Proceedings of the 9th Annual Conference on Genetic and Evolutionary Computation, GECCO ‘07, pages 1187–1194, New York, NY, USA, 2007. Association for Computing Machinery.
    """
    # initialize offspring
    d = parents.shape[1]
    offspring = np.empty((num_offspring, d))

    # generate the offspring
    for i in range(num_offspring):
        # select two parents
        parent1, parent2 = parents[np.random.choice(2, 2, replace=False)]
        # generate a random number
        u = np.random.rand(d)
        beta = np.empty(d)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (1 / (2 * (1 - u[u > 0.5]))) ** (1 / (eta + 1))
        # generate the offspring
        offspring[i] = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    
    return offspring