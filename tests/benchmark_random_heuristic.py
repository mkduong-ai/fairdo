import numpy as np
import time
from fairdo.optimize import random_method, random_method_vectorized
from utils import benchmark


def f(x):
    return np.sum(x)


def test1():
    random_method(f, 40_00, 200, 500)


def test2():
    random_method_vectorized(f, 40_00, 200, 500)


if __name__ == "__main__":
    avg_time = benchmark(test1, repeats=10)
    print(f"Average execution time over 10 runs: {avg_time:.4f} seconds")
    avg_time = benchmark(test2, repeats=10)
    print(f"Average execution time over 10 runs: {avg_time:.4f} seconds")


