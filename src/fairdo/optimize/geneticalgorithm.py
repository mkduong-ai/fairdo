"""
Genetic Algorithm
=================

This module implements the Genetic Algorithm for combinatorial optimization problems.
It reflects the process of natural selection where the fittest individuals are selected for
reproduction to produce the offspring of the next generation.

The main functions in this module are **genetic_algorithm_constraint** and **genetic_algorithm**,
which perform the Genetic Algorithm on a given fitness function to be maximized.
These functions take as input the fitness `f`, the dimensionality `d` of `f`,
the population size `pop_size`, and number of generation `num_generations`.
The algorithm iteratively applies selection, crossover, and mutation operations to
evolve the population over a number of generations.

The `select_parents`, `crossover`, and `mutate` functions can be passed to `genetic_algorithm_constraint` or
`genetic_algorithm` to customize the selection, crossover, and mutation operations.
These functions are defined in the :mod:`geneticoperators.crossover`, :mod:`geneticoperators.mutation`, and
:mod:`geneticoperators.selection` modules respectively.

This implementation of the Genetic Algorithm is based on the works of the following references:

Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.

Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation.
"""

import pathos.multiprocessing as mp
import numpy as np

from fairdo.optimize.geneticoperators.crossover import onepoint_crossover, uniform_crossover
from fairdo.optimize.geneticoperators.mutation import fractional_flip_mutation
from fairdo.optimize.geneticoperators.selection import elitist_selection, tournament_selection
from fairdo.utils.penalty import relative_difference_penalty


def generate_population(pop_size, d):
    """
    Generate a random population of binary vectors. Each vector has a length of d.
    The values of the vectors are either 0 or 1. The population is generated randomly.

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


def evaluate_individual(args):
    """
    Calculates the fitness of an individual. The fitness is the value of the fitness function
    plus a penalty for individuals that do not satisfy the size constraint.

    Parameters
    ----------
    args: tuple
        The arguments to pass to the function. The arguments are (f, n, individual, penalty_function).

    Returns
    -------
    fitness: float
        The fitness of the individual.
    """
    f, n, individual, penalty_function = args
    fitness = f(individual)
    if n > 0:
        fitness += penalty_function(individual, n)
    return fitness


def evaluate_population_single_cpu(f, n, population, penalty_function=relative_difference_penalty):
    """
    Calculates the fitness of each individual in a population. The fitness is the value of the fitness function
    plus a penalty for individuals that do not satisfy the size constraint.

    Parameters
    ----------
    f: callable
        The fitness function to evaluate.
    n: int
        The constraint value.
    population: ndarray, shape (pop_size, d)
        The population of vectors to evaluate.
    penalty_function: callable, optional
        The penalty function to apply to individuals that do not satisfy the size constraint.

    Returns
    -------
    fitness: ndarray, shape (pop_size,)
        The fitness values of the population.
    """
    # fallback to single process execution if multiprocessing fails
    fitness = np.apply_along_axis(f, axis=1, arr=population)
    if n > 0:
        # add absolute_difference_penalty to the fitness of all individuals
        # that do not satisfy the size constraint
        fitness += np.apply_along_axis(lambda x: penalty_function(x, n), axis=1, arr=population)

    return fitness


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
    population: ndarray, shape (pop_size, d)
        The population of vectors to evaluate.
    penalty_function: callable, optional
        The penalty function to apply to individuals that do not satisfy the size constraint.

    Returns
    -------
    fitness: ndarray, shape (pop_size,)
        The fitness values of the population.
    """
    try:
        # use multiprocessing to speed up the evaluation if the population is large enough
        if mp.cpu_count() > 1 and population.shape[0] >= 200:
            # use multiprocessing to speed up the evaluation
            with mp.Pool() as pool:
                fitness = pool.map(evaluate_individual, [(f, n, individual, penalty_function)
                                                         for individual in population])

            return np.array(fitness)
        else:
            return evaluate_population_single_cpu(f, n, population, penalty_function)
    except Exception as e:
        print(f"Multiprocessing pool failed with error: {e}")
        print("Falling back to single process execution")
        return evaluate_population_single_cpu(f, n, population, penalty_function)


def genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
                                 selection=elitist_selection,
                                 crossover=uniform_crossover,
                                 mutation=fractional_flip_mutation,
                                 maximize=False,
                                 tol=1e-6,
                                 patience=50):
    r"""
    Perform a genetic algorithm with constraints. The constraint is that the sum of the binary vector must be equal
    to n. The fitness function is the value of the fitness function plus a penalty for individuals that do not satisfy
    the size constraint.

    The genetic algorithm consists of the following steps which are repeated for a specified number of generations:

    1. Generate an initial population of binary vectors.
    2. Evaluate the fitness of each vector in the population.
    3. Select the best individuals to be parents. (Selection)
    4. Create offspring by performing crossover on the parents. (Crossover)
    5. Mutate the offspring. (Mutation)

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
    selection: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
    mutation: callable
        The function to perform the mutation operation.
    maximize: bool, optional
        Whether to maximize or minimize the fitness function.
    tol: float, optional
        The tolerance for early stopping. If the best solution found is within tol of the previous best solution,
        then the algorithm stops.
    patience: int, optional
        The number of generations to wait before early stopping.

    Returns
    -------
    best_solution : ndarray, shape (d,)
        The best solution found by the algorithm.
    best_fitness : float
        The fitness of the best solution found by the algorithm.

    Notes
    -----
    The genetic algorithm is used to maximize the given fitness function.
    To avoid having to rewrite the selection, crossover, and mutation functions to work with minimization problems,
    the fitness function is negated if we are minimizing.
    The fitness function must map the binary vector to a positive value, i.e.,
    :math:`f: \{0, 1\}^d \rightarrow \mathbb{R}^+`.
    """
    # negate the fitness function if we are minimizing
    if not maximize:
        f_orig = f
        f = lambda x: -f_orig(x)

    # generate the initial population
    population = generate_population(pop_size, d)
    # evaluate the function for each vector in the population
    fitness = evaluate_population(f, n, population, penalty_function=relative_difference_penalty)
    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_population = population[best_idx]
    no_improvement_streak = 0
    # perform the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # select the parents
        parents, fitness = selection(population, fitness)
        # create the offspring
        num_offspring = pop_size - parents.shape[0]
        offspring = crossover(parents, num_offspring)
        # mutate the offspring
        offspring = mutation(offspring)
        # evaluate the function for the new offspring
        offspring_fitness = evaluate_population(f, n, offspring, penalty_function=relative_difference_penalty)
        # create the new population (allow the parents to be part of the next generation)
        population = np.concatenate((parents, offspring))
        fitness = np.concatenate((fitness, offspring_fitness))
        # save the best solution found so far
        best_idx = np.argmax(offspring_fitness)
        if offspring_fitness[best_idx] > best_fitness + tol:
            best_fitness = offspring_fitness[best_idx]
            best_population = offspring[best_idx]
            no_improvement_streak = 0
        else:
            # early stopping if the best solution is found
            no_improvement_streak += 1
            if no_improvement_streak >= patience:
                print(f"Stopping after {generation + 1} generations due to lack of improvement.")
                break

    if not maximize:
        # negate the fitness back to its original form
        best_fitness = -best_fitness
    return best_population, best_fitness


def genetic_algorithm(f, d, pop_size, num_generations,
                      selection=elitist_selection,
                      crossover=uniform_crossover,
                      mutation=fractional_flip_mutation,
                      maximize=False,
                      tol=1e-6,
                      patience=50):
    r"""
    Perform a genetic algorithm. The genetic algorithm is used to maximize the given fitness function.
    It consists of the following steps which are repeated for a specified number of generations:

    1. Generate an initial population of binary vectors.
    2. Evaluate the fitness of each vector in the population.
    3. Select the best individuals to be parents. (Selection)
    4. Create offspring by performing crossover on the parents. (Crossover)
    5. Mutate the offspring. (Mutation)

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
    selection: callable
        The function to select the parents from the population.
    crossover: callable
        The function to perform the crossover operation.
    mutation: callable
        The function to perform the mutation operation.
    maximize: bool, optional
        Whether to maximize or minimize the fitness function.
    tol: float, optional
        The tolerance for early stopping. If the best solution found is within tol of the previous best solution,
        then the algorithm stops.
    patience: int, optional
        The number of generations to wait before early stopping.

    Returns
    -------
    best_solution: ndarray
        The best solution found by the algorithm.
    best_fitness: float
        The fitness of the best solution found by the algorithm.

    Notes
    -----
    The genetic algorithm is used to maximize the given fitness function.
    To avoid having to rewrite the selection, crossover, and mutation functions to work with minimization problems,
    the fitness function is negated if we are minimizing.
    The fitness function must map the binary vector to a positive value, i.e.,
    :math:`f: \{0, 1\}^d \rightarrow \mathbb{R}^+`.
    """
    return genetic_algorithm_constraint(f=f, d=d, n=0,
                                        pop_size=pop_size,
                                        num_generations=num_generations,
                                        selection=selection,
                                        crossover=crossover,
                                        mutation=mutation,
                                        maximize=maximize,
                                        tol=tol,
                                        patience=patience)
