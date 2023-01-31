import numpy as np

'''
Here is an example of a research paper that uses the genetic algorithm to solve optimization problems with constraints:

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
The algorithm has been widely used in various fields of research, such as engineering, economics, and finance.

You can use this approach in your research work, by adding the constraint to your fitness function and applying
a penalty function to the solutions that do not satisfy the constraint.
'''


def f(x):
    # replace this with your own blackbox function
    return sum(x)


def generate_population(pop_size, d):
    # generate a population of binary vectors
    return np.random.randint(2, size=(pop_size, d))


def evaluate_population(f, population):
    # evaluate the function for each vector in the population
    return np.apply_along_axis(f, 1, population)


def select_parents(population, fitness, num_parents):
    # select the best individuals from the population to be parents
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        idx = np.random.choice(np.flatnonzero(fitness == fitness.min()))
        parents[i, :] = population[idx, :]
    return parents


def crossover(parents, offspring_size):
    # perform crossover on the parents to generate new offspring
    offspring = np.empty(offspring_size)
    # the crossover point is a random index between 1 and d-1
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # parent selection
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        # offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # second half from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutate(offspring):
    # mutate the offspring by flipping random bits
    mutation_rate = 0.01
    for idx in range(offspring.shape[0]):
        random_value = np.random.uniform(0, 1, 1)
        offspring[idx,:] = np.where(random_value < mutation_rate, np.logical_not(offspring[idx,:]), offspring[idx,:])
    return offspring
    
'''
In this example, the select_parents_constraint function is used to select the parents for the next generation.
This function selects parents from the population randomly but only those individuals whose sum of entries is equal to n.
The genetic_algorithm_constraint function is modified to use this new selection process.
The final population is returned and the fitness of the best individual is returned.

It's important to note that the results of this algorithm may vary depending on the specific parameter values you choose
(e.g. mutation rate, population size, number of generations),
as well as the specific function you're trying to minimize.
'''


def select_parents_constraint(population, fitness, pop_size, n):
    parents = []
    while len(parents) < pop_size:
        idx = np.random.randint(0, pop_size)
        if sum(population[idx]) == n:
            parents.append(population[idx])
    return np.array(parents)


def genetic_algorithm(f, pop_size, d, num_generations):
    """

    Parameters
    ----------
    f: function
        function to minimize
    pop_size: int
        population size (number of individuals)
    d: int
        number of dimensions
    num_generations: int
        number of generations
    Returns
    -------

    """
    # generate the initial population
    population = generate_population(pop_size, d)
    # evaluate the function for each vector in the population
    fitness = evaluate_population(population)
    # perform the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # select the parents
        parents = select_parents(population, fitness, pop_size)
        # create the offspring
        offspring_size = (pop_size-parents.shape[0], d)
        offspring = crossover(parents, offspring_size)
        # mutate the offspring
        offspring = mutate(offspring)
        # evaluate the function for the new offspring
        offspring_fitness = evaluate_population(f, offspring)
        # create the new population
        population = np.concatenate((parents, offspring))
        fitness = np.concatenate((fitness[:parents.shape[0]], offspring_fitness))
        # select the best individuals to keep for the next generation
        idx = np.argsort(fitness)[:pop_size]
        population = population[idx]
        fitness = fitness[idx]
    
    return population, fitness


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
    return genetic_algorithm(f=func, pop_size=50, d=dims, num_generations=100)


def genetic_algorithm_constraint(pop_size, d, num_generations, n):
    # generate the initial population
    population = generate_population(pop_size, d)
    # evaluate the function for each vector in the population
    fitness = evaluate_population(population)
    # perform the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # select the parents
        parents = select_parents_constraint(population, fitness, pop_size, n)
        # create the offspring
        offspring_size = (pop_size-parents.shape[0], d)
        offspring = crossover(parents, offspring_size)
        # mutate the offspring
        offspring = mutate(offspring)
        # evaluate the function for the new offspring
        offspring_fitness = evaluate_population(offspring)
        # create the new population
        population = np.concatenate((parents, offspring))
        fitness = np.concatenate((fitness[:parents.shape[0]], offspring_fitness))
        # select the best individuals to keep for the next generation
        idx = np.argsort(fitness)[:pop_size]
        population = population[idx]
    return population[0], fitness[0]