import numpy as np


def f(x):
    # replace this with your own blackbox function
    return sum(x)


def penalty_normalized(x, n):
    """
    Percentage of the sum of the entries of the vector x that is greater than n

    Parameters
    ----------
    x: numpy array
        vector
    n: int
        constraint

    Returns
    -------
    penalty: float
    """
    return abs(sum(x) - n) / n


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
    Perform the crossover operation on the parents to create the offspring

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
        crossover_point = np.random.randint(1, offspring_size[1]-1)
        # parent selection
        # switch between the two parents randomly at each crossover point to create the offspring
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        # offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # second half from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
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


def genetic_algorithm_constraint(f, d, n, pop_size, num_generations):
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
    Returns
    -------

    """
    # generate the initial population
    population = generate_population(pop_size, d)
    # evaluate the function for each vector in the population
    fitness = evaluate_population(f, n, population, penalty_function=penalty_normalized)
    # perform the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # select the parents
        parents, fitness = select_parents(population, fitness, num_parents=2)
        # create the offspring
        offspring_size = (pop_size-parents.shape[0], d)
        offspring = crossover(parents, offspring_size)
        # mutate the offspring
        offspring = mutate(offspring, mutation_rate=0.05)
        # evaluate the function for the new offspring
        offspring_fitness = evaluate_population(f, n, offspring, penalty_function=penalty_normalized)
        # create the new population (allow the parents to be part of the next generation)
        population = np.concatenate((parents, offspring))
        fitness = np.concatenate((fitness, offspring_fitness))
        # select the best individuals to keep for the next generation
        idx = np.argsort(fitness)[:pop_size]
        population = population[idx]
        fitness = fitness[idx]
    return population[0], fitness[0]


def genetic_algorithm(f, d, pop_size, num_generations):
    """

    Parameters
    ----------
    f
    d
    pop_size
    num_generations

    Returns
    -------

    """
    return genetic_algorithm_constraint(f=f, d=d, n=0, pop_size=pop_size, num_generations=num_generations)


def genetic_algorithm_method(func, dims):
    """
    Genetic Algorithm method
    Parameters
    ----------
    func: function
        function to optimize
    dims: int
        number of dimensions

    Returns
    -------

    """
    return genetic_algorithm(f=func, d=dims, pop_size=50, num_generations=100)
