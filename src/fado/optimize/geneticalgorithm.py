import numpy as np

from fado.optimize.geneticoperators.crossover import onepoint_crossover, uniform_crossover
from fado.optimize.geneticoperators.mutation import mutate
from fado.optimize.geneticoperators.selection import select_parents
from fado.utils.penalty import relative_difference_penalty


def generate_population(pop_size, d):
    """
    Generate a population of binary vectors.

    Parameters
    ----------
    pop_size: int
        The size of the population to generate.
    d: int
        The dimension of the binary vectors.

    Returns
    -------
    population: ndarray, shape (pop_size, d)
        The generated population of binary vectors.
    """
    return np.random.randint(2, size=(pop_size, d))


def evaluate_population(f, n, population, penalty_function=relative_difference_penalty):
    """
    Calculates the fitness of each individual in a population. The fitness is the value of the fitness function
    plus a penalty for individuals that do not satisfy the size constraint.

    Parameters
    ----------
    f: callable
        The fitness function to evaluate.
    n: int
        The constraint value.
    population: ndarray
        The population of vectors to evaluate.
    penalty_function: callable, optional
        The penalty function to apply to individuals that do not satisfy the size constraint.

    Returns
    -------
    fitness: ndarray, shape (pop_size,)
        The fitness values of the population.
    """
    fitness = np.apply_along_axis(f, axis=1, arr=population)

    if n > 0:
        # add a absolute_difference_penalty to the fitness of all individuals that do not satisfy the size constraint
        fitness += np.apply_along_axis(lambda x: penalty_function(x, n), axis=1, arr=population)

    return fitness


def genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
                                 select_parents=select_parents,
                                 crossover=onepoint_crossover,
                                 mutate=mutate, ):
    """
    Perform a genetic algorithm with constraints.

    Parameters
    ----------
    f: callable
        The fitness function to minimize.
    d: int
        The number of dimensions.
    n: int
        The constraint value.
    pop_size: int
        The size of the population.
    num_generations: int
        The number of generations.
    select_parents: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
    mutate: callable
        The function to perform the mutation operation.

    Returns
    -------
    best_solution : ndarray, shape (d,)
        The best solution found by the algorithm.
    best_fitness : float
        The fitness of the best solution found by the algorithm.
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
                      crossover=onepoint_crossover,
                      mutate=mutate, ):
    """
    Perform a genetic algorithm.

    Parameters
    ----------
    f: callable
        The fitness function to optimize.
    d: int
        The number of dimensions.
    pop_size: int
        The size of the population.
    num_generations: int
        The number of generations.
    select_parents: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
    mutate: callable
        The function to perform the mutation operation.

    Returns
    -------
    best_solution: ndarray
        The best solution found by the algorithm.
    best_fitness: float
        The fitness of the best solution found by the algorithm.
    """
    return genetic_algorithm_constraint(f=f, d=d, n=0, pop_size=pop_size, num_generations=num_generations,
                                        select_parents=select_parents,
                                        crossover=crossover,
                                        mutate=mutate, )


def genetic_algorithm_method(f, d,
                             select_parents=select_parents,
                             crossover=onepoint_crossover,
                             mutate=mutate, ):
    """
    Genetic Algorithm method

   Parameters
    ----------
    f: callable
        The fitness function to optimize.
    d: int
        The number of dimensions.
    select_parents: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
        Default: onepoint_crossover
    mutate: callable
        The function to perform the mutation operation.

    Returns
    -------
    best_solution: ndarray, shape (d,)
        The best solution found by the algorithm.
    best_fitness: float
        The fitness of the best solution found by the algorithm.
    """
    return genetic_algorithm(f=f, d=d, pop_size=50, num_generations=100,
                             select_parents=select_parents,
                             crossover=crossover,
                             mutate=mutate, )


def genetic_algorithm_uniform_method(f, d,
                                     select_parents=select_parents,
                                     crossover=uniform_crossover,
                                     mutate=mutate, ):
    """
    Genetic Algorithm method

   Parameters
    ----------
    f: callable
        The fitness function to optimize.
    d: int
        The number of dimensions.
    select_parents: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
        Default: uniform_crossover
    mutate: callable
        The function to perform the mutation operation.

    Returns
    -------
    best_solution: ndarray, shape (d,)
        The best solution found by the algorithm.
    best_fitness: float
        The fitness of the best solution found by the algorithm.
    """
    return genetic_algorithm(f=f, d=d, pop_size=50, num_generations=100,
                             select_parents=select_parents,
                             crossover=crossover,
                             mutate=mutate, )
