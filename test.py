import numpy as np
from fado.optimize.geneticoperators import kpoint_crossover, uniform_crossover


def test():
    parents = np.ones((2, 10))
    parents[0] = parents[0] * 2

    offspring = kpoint_crossover(parents, (4, 10), k=2)

    print(offspring)


if __name__ == "__main__":
    test()