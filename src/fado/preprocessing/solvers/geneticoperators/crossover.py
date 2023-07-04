import numpy as np


def onepoint_crossover(parents, offspring_size):
    """
    Perform the crossover operation with One-point crossover on the parents to create the offspring

    Parameters
    ----------
    parents: numpy array
        parents of the offspring with shape (2, d)
    offspring_size: tuple
        size of the offspring

    Returns
    -------
    offspring: numpy array
    """
    # perform one-point crossover on the parents to generate new offspring
    return kpoint_crossover(parents, offspring_size, k=1)


def uniform_crossover(parents, offspring_size, p=0.5):
    """
    Perform the crossover operation with Uniform crossover on the parents to create the offspring

    Parameters
    ----------
    parents: numpy array
        parents of the offspring with shape (2, d)
    offspring_size: tuple
        size of the offspring
    p: float
        probability of selecting a gene from the first parent, default is 0.5

    Returns
    -------
    offspring: numpy array
    """
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        # create a mask with the same shape as a gene
        if p is None:
            # if p is not specified, randomly choose a probability for each gene
            p = np.random.uniform()
        mask = np.random.uniform(size=offspring_size[1]) < p
        # assign genes to the offspring based on the mask
        offspring[k] = np.where(mask, parents[0], parents[1])
    return offspring


def kpoint_crossover(parents, offspring_size, k=2):
    """
    Perform the crossover operation with K-point crossover on the parents to create the offspring

    Parameters
    ----------
    parents: numpy array
        parents of the offspring with shape (2, d)
    offspring_size: tuple
        size of the offspring
    k: int
        number of crossover points, default is 2

    Returns
    -------
    offspring: numpy array
    """
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(offspring_size)
    for i in range(offspring_size[0]):
        # parent selection
        # switch between the two parents randomly at each crossover point to create the offspring
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        # calculate the crossover points
        crossover_points = sorted(np.random.choice(range(1, offspring_size[1]), k - 1, replace=False))
        # alternate between parents in each segment
        segments = np.array_split(range(offspring_size[1]), len(crossover_points) + 1)
        for j in range(len(segments)):
            if j % 2 == 0:
                offspring[i, segments[j]] = parents[parent1_idx, segments[j]]
            else:
                offspring[i, segments[j]] = parents[parent2_idx, segments[j]]
    return offspring