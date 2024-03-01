import numpy as np


def non_dominated_sort(fitness_values):
    """
    Perform non-dominated sorting on the given fitness values.

    Parameters
    ----------
    fitness_values : ndarray, shape (pop_size, num_fitness_functions)
        The fitness values of each individual in the population for each fitness function.

    Returns
    -------
    fronts : list of ndarrays
        List of fronts, where each front contains the indices of individuals in that front.
    """
    # TODO: Sign reversal for maximization
    pop_size = fitness_values.shape[0]
    fronts = []
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    # Calculate the dominating counts and the indices of individuals that are dominated by each individual
    for i in range(pop_size):
        dominating_counts[i] = np.sum(np.all(fitness_values[i] <= fitness_values, axis=1)) - 1
        dominated_indices[i] = np.where(np.all(fitness_values[i] >= fitness_values, axis=1) & ~(np.arange(pop_size) == i))[0].tolist()

    # Find the first front
    current_front = np.where(dominating_counts == 0)[0]
    # Iterate over the fronts
    while current_front.size > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_indices[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front)

    return fronts


def test(pop_size, num_objectives):
    # Generate random fitness values
    fitness_values = np.random.rand(pop_size, num_objectives)
    print(fitness_values)
    
    fronts = non_dominated_sort(fitness_values)
    
    return fronts

# Main
pop_size = 4
num_objectives = 2
print(test(pop_size, num_objectives))
