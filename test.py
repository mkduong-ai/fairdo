import numpy as np
from fairdo.optimize.geneticoperators import kpoint_crossover, uniform_crossover


def test():
    parents = np.ones((2, 10))
    parents[0] = parents[0] * 2

    offspring = kpoint_crossover(parents, 4, k=2)
    offspring2 = uniform_crossover(parents, 4, p=0.5)

    print(offspring)


if __name__ == "__main__":
    test()