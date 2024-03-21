import unittest
import numpy as np


def crowding_distance(fitness_values):
    """
    Calculate crowding distance for each individual in the population.

    Parameters
    ----------
    fitness_values : ndarray, shape (N, num_fitness_functions)
        Fitness values of the population.

    Returns
    -------
    crowding_distances : ndarray, shape (N,)
        Crowding distances for each individual.
    """
    pop_size, num_objectives = fitness_values.shape
    crowding_distances = np.zeros(pop_size)

    for obj_index in range(num_objectives):
        # Sort the fitness values based on the current objective in ascending order. Best values first.
        sorted_indices = np.argsort(fitness_values[:, obj_index])
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf

        f_max = fitness_values[sorted_indices[-1], obj_index]
        f_min = fitness_values[sorted_indices[0], obj_index]

        if f_max == f_min:
            continue

        # Crowding distance is the sum of the distances to the previous and next individuals.
        # It geometrically describes the sum of the lengths of a cuboid.
        for i in range(1, pop_size - 1):
            crowding_distances[sorted_indices[i]] += (fitness_values[sorted_indices[i + 1], obj_index]
                                                      - fitness_values[sorted_indices[i - 1], obj_index])/ (f_max - f_min)
    return crowding_distances


def crowding_distance_vectorized(fitness_values):
    """
    Calculate crowding distance for each individual in the population.
    Parameters
    ----------
    fitness_values : ndarray, shape (N, num_fitness_functions)
        Fitness values of the population.
    Returns
    -------
    crowding_distances : ndarray, shape (N,)
        Crowding distances for each individual.
    """
    pop_size, num_objectives = fitness_values.shape
    crowding_distances = np.zeros(pop_size)
    for obj_index in range(num_objectives):
        # Sort the fitness values based on the current objective in ascending order. Best values first.
        sorted_indices = np.argsort(fitness_values[:, obj_index])
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf
        f_max = fitness_values[sorted_indices[-1], obj_index]
        f_min = fitness_values[sorted_indices[0], obj_index]
        if f_max == f_min:
            continue

        # Crowding distance is the sum of the distances to the previous and next individuals. It geometrically describes the sum of the lengths of a cuboid.
        nxt = fitness_values[sorted_indices[2:], obj_index]
        prv = fitness_values[sorted_indices[:-2], obj_index]
        crowding_distances[sorted_indices[1:-1]] += (nxt - prv)/ (f_max - f_min)

    return crowding_distances


class TestCrowdingDistance(unittest.TestCase):
    def test_crowding_distance(self):
        # Test case with simple fitness values
        for i in range(1,100):
            fitness_values = np.random.rand(100, i)
            np.testing.assert_array_equal(crowding_distance(fitness_values),
                                          crowding_distance_vectorized(fitness_values))
            print("Test passed for {} objectives.".format(i))

if __name__ == '__main__':
    unittest.main()


