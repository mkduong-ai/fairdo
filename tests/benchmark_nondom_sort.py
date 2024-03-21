import numpy as np
import time
from fairdo.optimize.multi import dom_counts_indices, dom_counts_indices_fast
from fairdo.optimize.multi import non_dominated_sort, non_dominated_sort_fast
from utils import benchmark


def test(pop_size, num_objectives):
    repeats = 5

    # Generate random fitness values
    fitness_values = np.random.rand(pop_size, num_objectives)
    fitness_values[0, :] = [0, 0]
    fitness_values[1, :] = [0, 0]

    time = benchmark(lambda: dom_counts_indices(fitness_values), repeats=repeats)
    time_broadcast = benchmark(lambda: dom_counts_indices_fast(fitness_values), repeats=repeats)

    print(f"Average execution time over {repeats} runs: {time:.4f} seconds")
    print(f"Average execution time over {repeats} runs (broadcast): {time_broadcast:.4f} seconds")
    print(f"Speedup: {time / time_broadcast:.2f}")

def test2(pop_size, num_objectives):
    fitness_values = np.random.rand(pop_size, num_objectives)
    fitness_values[0, :] = [0, 0]
    fitness_values[1, :] = [0, 0]

    counts, dom_list = dom_counts_indices(fitness_values)
    counts_broadcast, dom_list_broadcast = dom_counts_indices_fast(fitness_values)

    print(counts)
    print(counts_broadcast)
    print('---')
    print(dom_list)
    print(dom_list_broadcast)

# Main
pop_size = 100
num_objectives = 2
test(pop_size, num_objectives)