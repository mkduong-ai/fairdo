import numpy as np

from fairdo.optimize.geneticoperators.initialization import random_initialization
from fairdo.optimize.geneticoperators.selection import elitist_selection, tournament_selection
from fairdo.optimize.geneticoperators.crossover import onepoint_crossover, uniform_crossover, simulated_binary_crossover
from fairdo.optimize.geneticoperators.mutation import fractional_flip_mutation, shuffle_mutation


def nsga2(fitness_functions, d, pop_size, num_generations,
          initialization=random_initialization,
          crossover=uniform_crossover,
          mutation=shuffle_mutation,
          maximize=False,
          tol=1e-6,
          patience=50):
    """
    Perform NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.

    NSGA-II maintains a population of solutions and uses non-dominated sorting and crowding distance to select
    the best solutions.

    Fitness functions are minimized by default.

    Parameters
    ----------
    fitness_functions : list of callables
        The list of fitness functions to optimize. Each function should take a binary vector and return a scalar value.
    d : int
        The number of dimensions.
    pop_size : int
        The size of the population.
    num_generations : int
        The number of generations.
    initialization : callable, optional
        The function to initialize the population. Default is random_initialization.
    selection : callable, optional
        The function to perform selection. Default is tournament_selection.
    crossover : callable, optional
        The function to perform crossover. Default is simulated_binary_crossover.
    mutation : callable, optional
        The function to perform mutation. Default is polynomial_mutation.
    maximize : bool, optional
        Whether to maximize or minimize the fitness functions. Default is False.
    tol : float, optional
        The tolerance for early stopping. Default is 1e-6.
    patience : int, optional
        The number of generations to wait before early stopping. Default is 50.

    Returns
    -------
    best_population : ndarray, shape (d,)
        The best solution found by NSGA-II.
    best_fitness : ndarray, shape (num_fitness_functions,)
        The fitness values of the best solution found by NSGA-II.

    Notes
    -----
    The fitness functions must map the binary vector to a scalar value, i.e., :math:`f: \{0, 1\}^d \rightarrow \mathbb{R}`.

    References
    ----------
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
    """
    rng = np.random.default_rng()

    # Negate the fitness functions if not maximizing

    # Generate the initial population
    population = initialization(pop_size=pop_size, d=d)
    
    # Evaluate the fitness of each individual in the population
    fitness_values = evaluate_population(fitness_functions, population)

    # Perform NSGA-II for the specified number of generations
    for _ in range(num_generations):
        # Select parents
        parents = rng.choice(population, size=2, replace=False, axis=0)
        # Perform crossover
        offspring = crossover(parents, pop_size)
        # Perform mutation
        offspring = mutation(offspring)
        # Evaluate the fitness of the offspring
        offspring_fitness_values = evaluate_population(fitness_functions, offspring)
        
        # Combine the parents and the offspring
        combined_population = np.concatenate((population, offspring))
        combined_fitness_values = np.concatenate((fitness_values, offspring_fitness_values))
        
        # Select the best individuals using non-dominated sorting and crowding distance
        fronts = non_dominated_sort(combined_fitness_values)
        # print('population indices', population_indices)
        selected_indices = selection_indices(combined_fitness_values, fronts, pop_size)

        # Update the population and fitness values
        population = combined_population[selected_indices]
        fitness_values = combined_fitness_values[selected_indices]
    
    # Return Pareto front
    pareto_front = population[:len(fronts[0])]
    pareto_front_fitness = fitness_values[:len(fronts[0])]

    return pareto_front, pareto_front_fitness


def evaluate_population(fitness_functions, population):
    """
    Evaluate the fitness of each individual in the population using the given fitness functions.

    Parameters
    ----------
    fitness_functions : list of callables
        The list of fitness functions to evaluate.
    population : ndarray, shape (pop_size, d)
        The population of binary vectors.

    Returns
    -------
    fitness_values : ndarray, shape (pop_size, num_fitness_functions)
        The fitness values of each individual in the population for each fitness function.
    """
    num_fitness_functions = len(fitness_functions)
    fitness_values = np.zeros((population.shape[0], num_fitness_functions))
    for i, fitness_function in enumerate(fitness_functions):
        # TODO: Parallelize this loop
        fitness_values[:, i] = np.apply_along_axis(fitness_function, axis=1, arr=population).flatten()
    
    return fitness_values


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
    dominating_counts, dominated_indices = dom_counts_indices(fitness_values)

    fronts = []
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


def dom_counts_indices(fitness_values):
    """
    Parameters
    ----------
    fitness_values : ndarray, shape (pop_size, num_fitness_functions)
        The fitness values of each individual in the population for each fitness function.

    Returns
    -------
    dominating_counts : ndarray, shape (pop_size,)
        The number of individuals that dominate each individual.
    dominated_indices : list of ndarrays
        The indices of individuals that are dominated by each individual.
    """
    pop_size = fitness_values.shape[0]
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if all(fitness_values[j] <= fitness_values[i]):
                dominating_counts[i] += 1
                dominated_indices[j].append(i)
            elif all(fitness_values[i] <= fitness_values[j]):
                dominating_counts[j] += 1
                dominated_indices[i].append(j)

    return dominating_counts, dominated_indices


def dom_counts_indices_fast(fitness_values):
    """
    TODO: Two solutions can dominate each other. This is not handled in the current implementation.
    Parameters
    ----------
    fitness_values : ndarray, shape (pop_size, num_fitness_functions)
        The fitness values of each individual in the population for each fitness function.

    Returns
    -------
    dominating_counts : ndarray, shape (pop_size,)
        The number of individuals that dominate each individual.
    dominated_indices : list of ndarrays
        The indices of individuals that are dominated by each individual.
    """
    pop_size = fitness_values.shape[0]
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        dominating_counts[i] = np.sum(np.all(fitness_values[i] >= fitness_values, axis=1)) - 1
        dominated_indices[i] = np.where(np.all(fitness_values[i] <= fitness_values, axis=1) & ~(np.arange(pop_size) == i))[0].tolist()

    return dominating_counts, dominated_indices


def selection_indices(combined_fitness_values, fronts, pop_size):
    """
    Select the best individuals from the combined population based on the non-dominated sorting results and crowding distance to maintain diversity.

    Parameters
    ----------
    combined_fitness_values : ndarray, shape (N, d)
        Combined population containing both original population and offspring.
    fronts : list of ndarrays
        List of fronts, where each front contains the indices of individuals in that front.
    pop_size : int
        The size of the population.

    Returns
    -------
    selected_indices : list
        Selected indices from the combined population.
    """
    selected_indices = []
    remaining_space = pop_size
    front_idx = 0

    # Iterate over fronts until the selected population size reaches pop_size
    while remaining_space > 0 and front_idx < len(fronts):
        current_front = fronts[front_idx]
        if len(current_front) <= remaining_space:
            # If the current front can fit entirely into the selected population, add it
            selected_indices.extend(current_front)
            remaining_space -= len(current_front)
        else:
            # If the current front cannot fit entirely, select individuals based on crowding distance
            crowding_distances = crowding_distance(combined_fitness_values[current_front])
            # Select individuals with larger crowding distances first
            sorted_indices = np.argsort(crowding_distances)[::-1]
            selected_indices.extend(current_front[sorted_indices[:remaining_space]])
            remaining_space = 0
        front_idx += 1
        #print(selected_indices)

    return selected_indices


def crowding_distance(fitness_values):
    """
    Calculate crowding distance for each individual in the population.
    Normalization skipped because it is not necessary for selection
    and risk of division by zero.

    Parameters
    ----------
    fitness_values : ndarray, shape (N, num_fitness_functions)
        Fitness values of the population.

    Returns
    -------
    crowding_distances : ndarray, shape (N,)
        Crowding distances for each individual.
    """
    pop_size, num_objectives = fitness_values.shape
    crowding_distances = np.zeros(pop_size)

    for obj_index in range(num_objectives):
        sorted_indices = np.argsort(fitness_values[:, obj_index])
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf

        for i in range(1, pop_size - 1):
            crowding_distances[sorted_indices[i]] += (fitness_values[sorted_indices[i + 1], obj_index]
                                                      - fitness_values[sorted_indices[i - 1], obj_index])

    return crowding_distances