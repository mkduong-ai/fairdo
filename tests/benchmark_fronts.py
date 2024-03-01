import numpy as np
import time

# Original method
def original_find_fronts(dominating_counts, dominated_indices):
    fronts = []
    current_front = np.where(dominating_counts == 0)[0]

    while current_front.size > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_indices[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front)
        print('endless loop')
    
    return fronts

# Optimized method
def optimized_find_fronts(dominating_counts, dominated_indices):
    fronts = []
    current_front = np.where(dominating_counts == 0)[0]

    while current_front.size > 0:
        fronts.append(current_front)

        # Update dominating counts for individuals in the next front
        for i in current_front:
            dominating_counts[dominated_indices[i]] -= 1

        # Find the next front
        next_front_mask = dominating_counts == 0
        current_front = np.where(next_front_mask)[0]

        # Break out of the loop if no individuals are found for the next front
        if current_front.size == 0:
            break
        print('endless loop')
        
    return fronts

# Benchmark function
def benchmark(pop_size, num_objectives):
    # Generate random fitness values
    fitness_values = np.random.rand(pop_size, num_objectives)

    # Calculate dominating_counts and dominated_indices
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_indices = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        dominating_counts[i] = np.sum(np.all(fitness_values[i] <= fitness_values, axis=1)) - 1
        dominated_indices[i] = np.where(np.all(fitness_values[i] >= fitness_values, axis=1) & ~(np.arange(pop_size) == i))[0].tolist()

    # Benchmark original method
    start_time = time.time()
    original_find_fronts(dominating_counts.copy(), [indices.copy() for indices in dominated_indices])
    original_time = time.time() - start_time

    # Benchmark optimized method
    start_time = time.time()
    optimized_find_fronts(dominating_counts.copy(), [indices.copy() for indices in dominated_indices])
    optimized_time = time.time() - start_time

    return original_time, optimized_time

# Main
pop_size = 10
num_objectives = 5
original_time, optimized_time = benchmark(pop_size, num_objectives)

print(f"Original method time: {original_time:.6f} seconds")
print(f"Optimized method time: {optimized_time:.6f} seconds")

