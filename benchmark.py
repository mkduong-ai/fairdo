import time
import numpy as np
import multiprocessing as mp

# Define a simple fitness function and penalty function
def f(x):
    return np.sum(x)
def penalty_function(individual, n):
    return abs(sum(individual) - n)

def evaluate_population(f, n, population, penalty_function=penalty_function):
    fitness = np.apply_along_axis(f, axis=1, arr=population)

    if n > 0:
        # add a absolute_difference_penalty to the fitness of all individuals that do not satisfy the size constraint
        fitness += np.apply_along_axis(lambda x: penalty_function(x, n), axis=1, arr=population)

    return fitness

def evaluate_individual(args):
    f, n, individual, penalty_function = args
    fitness = f(individual)
    if n > 0:
        fitness += penalty_function(individual, n)
    return fitness

def evaluate_population_pool(f, n, population, penalty_function=penalty_function):
    with mp.Pool() as pool:
        fitness = pool.map(evaluate_individual, [(f, n, individual, penalty_function) for individual in population])
        # print(len(fitness))
    return np.array(fitness)


# Benchmark the three approaches
def benchmark(f, n, population, penalty_function):
    start = time.time()
    evaluate_population(f, n, population, penalty_function)
    print("Time Single CPU: ", time.time() - start)
    
    start = time.time()
    evaluate_population_pool(f, n, population, penalty_function)
    print("Time with pool: ", time.time() - start)
    
    #start = time.time()
    #evaluate_population_processes(f, n, population, penalty_function)
    #print("Time with processes: ", time.time() - start)

# Run the benchmark
population = np.random.uniform(-10, 10, size=(20,20000))
n = 10
benchmark(f, n, population, penalty_function)
