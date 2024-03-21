import unittest
import numpy as np

from fairdo.optimize.multi import non_dominated_sort, non_dominated_sort_fast


class TestCrowdingDistance(unittest.TestCase):
    def test_crowding_distance(self):
        # Test case with simple fitness values
        for i in range(1,100):
            fitness_values = np.random.rand(100, i)
            np.testing.assert_array_equal(non_dominated_sort(fitness_values),
                                          non_dominated_sort_fast(fitness_values))
            print("Test passed for {} objectives.".format(i))

if __name__ == '__main__':
    unittest.main()


