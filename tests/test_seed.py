import time
import numpy as np
import pathos.multiprocessing as mp
from functools import partial


def evaluate_individual(i):
    np.random.seed(i)
    print(i)
    time.sleep(0.1)
    print(1-i/10)
    time.sleep(1-i/10)
    return np.random.randint(1, 1000)

def evaluate_population_pool():
    with mp.Pool() as pool:
        fitness = pool.map(evaluate_individual,[1, 2, 3, 4])
    return np.array(fitness)


# Run the benchmark
print(evaluate_population_pool())
