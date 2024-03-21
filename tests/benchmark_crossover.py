import numpy as np
from fairdo.optimize.geneticoperators import kpoint_crossover,\
    uniform_crossover
from utils import benchmark

def test1():
    parents = np.ones((2, 1000))

    offspring = uniform_crossover(parents, 1000, p=0.5)

def test2():
    parents = np.ones((2, 1000))

    offspring = kpoint_crossover(parents, 1000, k=1)


if __name__ == "__main__":
    avg_time = benchmark(test1, repeats=10)
    print(f"Average execution time over 10 runs: {avg_time:.4f} seconds")
    avg_time = benchmark(test2, repeats=10)
    print(f"Average execution time over 10 runs: {avg_time:.4f} seconds")

