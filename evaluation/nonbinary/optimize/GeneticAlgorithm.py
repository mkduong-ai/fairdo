import numpy as np

from optimize.Penalty import penalty_normalized


def f(x):
    # replace this with your own blackbox function
    return sum(x)


def generate_population(pop_size, d):
    # generate a population of binary vectors
    return np.random.randint(2, size=(pop_size, d))


def evaluate_population(f, n, population, penalty_function=penalty_normalized):
    # evaluate the function for each vector in the population
    fitness = np.apply_along_axis(f, 1, population)

    if n > 0:
        # add a penalty to the fitness of all individuals
        for i in range(population.shape[0]):
            fitness[i] += penalty_function(population[i], n)

    return fitness


def select_parents(population, fitness, num_parents=2):
    """
    In this example, the select_parents function is used to select the parents for the next generation.
    This function selects the fittest parents from the population.

    Parameters
    ----------
    population: numpy array
        population of individuals
    fitness: numpy array
        fitness of each individual
    num_parents: int
        number of parents to select

    Returns
    -------
    parents: numpy array
    """
    # select the best individuals from the population to be parents
    idx = np.argsort(fitness)
    parents = population[idx[:num_parents]]
    fitness = fitness[idx[:num_parents]]
    return parents, fitness


def crossover(parents, offspring_size):
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
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        # the crossover point is a random index between 1 and d-1
        crossover_point = np.random.randint(1, offspring_size[1] - 1)
        # parent selection
        # switch between the two parents randomly at each crossover point to create the offspring
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        # offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # second half from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


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
        crossover_points = sorted(np.random.choice(range(1, offspring_size[1]), k-1, replace=False))
        # alternate between parents in each segment
        segments = np.array_split(range(offspring_size[1]), len(crossover_points)+1)
        for j in range(len(segments)):
            if j % 2 == 0:
                offspring[i, segments[j]] = parents[parent1_idx, segments[j]]
            else:
                offspring[i, segments[j]] = parents[parent2_idx, segments[j]]
    return offspring


def mutate(offspring, mutation_rate=0.05):
    # mutate the offspring by flipping a percentage of random bits of each offspring
    num_mutation = int(mutation_rate * offspring.shape[1])
    for idx in range(offspring.shape[0]):
        # select the random bits to flip
        mutation_bits = np.random.choice(np.arange(offspring.shape[1]),
                                         num_mutation,
                                         replace=False)
        # flip the bits
        offspring[idx, mutation_bits] = 1 - offspring[idx, mutation_bits]
    return offspring


def genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
                                 select_parents=select_parents,
                                 crossover=crossover,
                                 mutate=mutate,):
    """
    Here is an example of a research paper that uses the genetic algorithm to solve
    optimization problems with constraints:

    @article{Deb2002,
      title={A fast and elitist multiobjective genetic algorithm: NSGA-II},
      author={Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and Meyarivan, T},
      journal={IEEE Transactions on Evolutionary Computation},
      volume={6},
      number={2},
      pages={182--197},
      year={2002},
      publisher={IEEE}
    }

    This paper describes an efficient multi-objective genetic algorithm called NSGA-II that can handle constraints by
    applying a penalty function approach.

    To use this for your problem, add the constraint to your fitness function and apply
    a penalty function to the solutions that do not satisfy the constraint.

    Parameters
    ----------
    f: callable
        function to minimize
    d: int
        number of dimensions
    n: int
        constraint value (sum of 1-entries in the vector must equal n)
    pop_size: int
        population size (number of individuals)
    num_generations: int
        number of generations
    select_parents: callable
        function to select the parents from the population
    crossover: callable
        function to perform the crossover operation
    mutate: callable
        function to perform the mutation operation
    Returns
    -------
    population: np.array of size (d,)
        The best solution found by the algorithm
    fitness: float
        The fitness of the best solution found by the algorithm
    """
    # generate the initial population
    population = generate_population(pop_size, d)
    best_population = population
    # evaluate the function for each vector in the population
    fitness = evaluate_population(f, n, population, penalty_function=penalty_normalized)
    best_fitness = fitness
    # perform the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # select the parents
        parents, fitness = select_parents(population, fitness, num_parents=2)
        # create the offspring
        offspring_size = (pop_size - parents.shape[0], d)
        offspring = crossover(parents, offspring_size)
        # mutate the offspring
        offspring = mutate(offspring, mutation_rate=0.05)
        # evaluate the function for the new offspring
        offspring_fitness = evaluate_population(f, n, offspring, penalty_function=penalty_normalized)
        # create the new population (allow the parents to be part of the next generation)
        population = np.concatenate((parents, offspring))
        fitness = np.concatenate((fitness, offspring_fitness))
        # select the best individuals to keep for the next generation (elitism)
        idx = np.argsort(fitness)[:pop_size]
        population = population[idx]
        fitness = fitness[idx]
    return population[0], fitness[0]


def genetic_algorithm(f, d, pop_size, num_generations,
                      select_parents=select_parents,
                      crossover=crossover,
                      mutate=mutate,):
    """

    Parameters
    ----------
    f: function
        function to optimize
    d: int
        number of dimensions
    pop_size: int
        population size (number of individuals)
    num_generations: int
        number of generations
    select_parents: callable
        function to select the parents from the population
    crossover: callable
        function to perform the crossover operation
    mutate: callable
        function to perform the mutation operation

    Returns
    -------
    population: np.array of size (d,)
        The best solution found by the algorithm
    fitness: float
        The fitness of the best solution found by the algorithm
    """
    return genetic_algorithm_constraint(f=f, d=d, n=0, pop_size=pop_size, num_generations=num_generations,
                                        select_parents=select_parents,
                                        crossover=crossover,
                                        mutate=mutate,)


def genetic_algorithm_method(f, d,
                             select_parents=select_parents,
                             crossover=crossover,
                             mutate=mutate,):
    """
    Genetic Algorithm method
    Parameters
    ----------
    f: function
        function to optimize
    d: int
        number of dimensions
    select_parents: callable
        function to select the parents from the population
    crossover: callable
        function to perform the crossover operation
    mutate: callable
        function to perform the mutation operation

    Returns
    -------
    population: np.array of size (d,)
        The best solution found by the algorithm
    fitness: float
        The fitness of the best solution found by the algorithm
    """
    return genetic_algorithm(f=f, d=d, pop_size=50, num_generations=100,
                             select_parents=select_parents,
                             crossover=crossover,
                             mutate=mutate,)
