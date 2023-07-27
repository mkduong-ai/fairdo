import numpy as np

from .geneticoperators.crossover import onepoint_crossover, uniform_crossover
from .geneticoperators.mutation import mutate
from .geneticoperators.selection import select_parents
from fado.utils.penalty import relative_difference_penalty


def generate_population(pop_size, d):
    # generate a population of binary vectors
    return np.random.randint(2, size=(pop_size, d))


def evaluate_population(f, n, population, penalty_function=relative_difference_penalty):
    # evaluate the function for each vector in the population
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
                      crossover=onepoint_crossover,
                      mutate=mutate, ):
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
                                        mutate=mutate, )


def genetic_algorithm_method(f, d,
                             select_parents=select_parents,
                             crossover=onepoint_crossover,
                             mutate=mutate, ):
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
                             mutate=mutate, )


def genetic_algorithm_uniform_method(f, d,
                                     select_parents=select_parents,
                                     crossover=uniform_crossover,
                                     mutate=mutate, ):
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
                             mutate=mutate, )
