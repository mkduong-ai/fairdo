import numpy as np


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