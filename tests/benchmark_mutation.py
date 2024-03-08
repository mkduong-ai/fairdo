import numpy as np
from utils import benchmark

def fractional_flip_mutation_vectorized(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring by flipping a percentage of random bits for each offspring.
    A fixed amount of bits is flipped for each offspring.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The percentage of random bits to flip for each offspring. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    offspring_out = offspring.copy()
    num_mutation = int(mutation_rate * offspring.shape[1])

    # Generate mutation indices directly without reshaping
    mutation_mask = np.random.randint(0, offspring.shape[1], size=(offspring.shape[0], num_mutation))

    # Flip the bits using advanced indexing
    offspring_out[np.arange(offspring.shape[0])[:, np.newaxis], mutation_mask] = 1 - offspring_out[np.arange(offspring.shape[0])[:, np.newaxis], mutation_mask]
    return offspring_out

def fractional_flip_mutation(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring by flipping a percentage of random bits for each offspring.
    A fixed amount of bits is flipped for each offspring.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional (default=0.05)
        The percentage of random bits to flip for each offspring. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    offspring_out = offspring.copy()
    num_mutation = int(mutation_rate * offspring.shape[1])
    for idx in range(offspring.shape[0]):
        # select the random bits to flip
        mutation_bits = np.random.choice(np.arange(offspring.shape[1]),
                                         size=num_mutation,
                                         replace=False)
        # flip the bits
        offspring_out[idx, mutation_bits] = 1 - offspring_out[idx, mutation_bits]
    return offspring_out

# Benchmark the two functions
n = 100
d = 10000
offspring = np.random.randint(0, 2, size=(n, d))
avg_time = benchmark(lambda: fractional_flip_mutation(offspring), repeats=10)
print(f"Fractional flip mutation: {avg_time:.6f} seconds")
avg_time = benchmark(lambda: fractional_flip_mutation_vectorized(offspring), repeats=10)
print(f"Fractional flip mutation (vectorized, efficient): {avg_time:.6f} seconds")

mutated_offspring = fractional_flip_mutation_vectorized(offspring, mutation_rate=0.2)
print(np.mean(offspring != mutated_offspring))
mutated_offspring = fractional_flip_mutation(offspring, mutation_rate=0.2)
print(np.mean(offspring != mutated_offspring))


def bit_flip_mutation(offspring, mutation_rate=0.05):
    """
    Mutates the given offspring by flipping each bit with a certain probability.
    Some offspring may not be mutated at all, and some may be mutated more than expected.

    Parameters
    ----------
    offspring: ndarray, shape (n, d)
        The offspring to be mutated. Each row represents an offspring, and each column represents a bit.
    mutation_rate: float, optional
        The probability of flipping each bit. Default is 0.05.

    Returns
    -------
    offspring: ndarray, shape (n, d)
        The mutated offspring. Each row represents an offspring, and each column represents a bit.
    """
    offspring_out = offspring.copy()
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    offspring_out[mutation_mask] = 1 - offspring_out[mutation_mask]
    return offspring_out

mutated_offspring2 = bit_flip_mutation(offspring, mutation_rate=0.2)
print(np.mean(offspring != mutated_offspring2))