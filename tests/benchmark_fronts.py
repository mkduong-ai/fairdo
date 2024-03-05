import numpy as np
import time
from fairdo.optimize.nsga2 import dom_counts_indices_fast


def benchmark(func, repeats=10):
    """
    Benchmark the execution time of a function.

    Parameters
    ----------
    func: callable
        The function to benchmark.
    repeats: int, optional
        The number of times to repeat the benchmark.

    Returns
    -------
    float
        The average execution time in seconds.
    """
    total_time = 0
    for _ in range(repeats):
        start_time = time.time()
        func()
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time / repeats


# Original method
def original_find_fronts(dominating_counts, dominated_indices):
    fronts = []
    # Find the first front
    current_front = np.where(dominating_counts == 0)[0]
    # Iterate over the fronts
    while current_front.size > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_indices[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front)

    return fronts


def optimized_find_fronts(dominating_counts, dominated_indices):
    fronts = []
    current_front = np.where(dominating_counts == 0)[0]

    while current_front.size > 0:
        fronts.append(current_front)
        # Next front is the set of all indices dominated by the current front
        next_front = np.concatenate([dominated_indices[i] for i in current_front])
        # Count the number of dominating solutions for each solution
        unique_next_front, counts = np.unique(next_front, return_counts=True)
        # Decrement the dominating counts
        dominating_counts[unique_next_front] -= counts
        # Next front is the set of all solutions with no dominating solutions
        current_front = np.where(dominating_counts[unique_next_front] == 0)[0]

    return fronts

# Benchmark function
def test(pop_size, num_objectives):
    fitness_values = np.random.rand(pop_size, num_objectives)
    counts, dom_list = dom_counts_indices_fast(fitness_values)

    time_original = benchmark(lambda: original_find_fronts(counts, dom_list), repeats=5)
    time_optimized = benchmark(lambda: optimized_find_fronts(counts, dom_list), repeats=5)

    fronts = original_find_fronts(counts, dom_list)
    fronts2 = optimized_find_fronts(counts, dom_list)
    print(np.all([np.all(fronts[i] == fronts2[i]) for i in range(len(fronts))]))


    print(f"Average execution time over 5 runs (original): {time_original:.4f} seconds")
    print(f"Average execution time over 5 runs (optimized): {time_optimized:.4f} seconds")
    print(f"Speedup: {time_original / time_optimized:.2f}")

# Main
pop_size = 15000
num_objectives = 10
test(pop_size, num_objectives)


