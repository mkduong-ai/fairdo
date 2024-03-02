import numpy as np
import time

# Original method
def original_method(fitness_values):
    pop_size = len(fitness_values)
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if all(fitness_values[j] <= fitness_values[i]):
                dominating_counts[i] += 1
                dominated_indices[j].append(i)
            elif all(fitness_values[i] <= fitness_values[j]):
                dominating_counts[j] += 1
                dominated_indices[i].append(j)

    return dominating_counts, dominated_indices

# NumPy method
def numpy_method(fitness_values):
    pop_size = len(fitness_values)
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        dominating_counts[i] = np.sum(np.all(fitness_values[i] >= fitness_values, axis=1)) - 1
        dominated_indices[i] = np.where(np.all(fitness_values[i] <= fitness_values, axis=1) & ~(np.arange(pop_size) == i))[0].tolist()

    return dominating_counts, dominated_indices

# Benchmark function
def benchmark(pop_size, num_objectives):
    # Generate random fitness values
    fitness_values = np.random.rand(pop_size, num_objectives)
    # fitness_values = np.array([[0, 4],
    #                            [0, 4],
    #                            [1, 1],
    #                            [1, 1],
    #                            ])
    # print(fitness_values)

    # Benchmark original method
    start_time = time.time()
    counts, indices = original_method(fitness_values)
    original_time = time.time() - start_time

    # Benchmark NumPy method
    start_time = time.time()
    counts2, indices2 = numpy_method(fitness_values)
    numpy_time = time.time() - start_time
    
    print(np.all(counts==counts2))
    print(np.all(indices==indices2))
    print(counts)
    print(counts2)
    
    return original_time, numpy_time

# Main
pop_size = 2000
num_objectives = 5
original_time, numpy_time = benchmark(pop_size, num_objectives)

print(f"Original method time: {original_time:.6f} seconds")
print(f"NumPy method time: {numpy_time:.6f} seconds")

