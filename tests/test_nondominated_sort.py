import unittest
import numpy as np

from fairdo.optimize.multi import non_dominated_sort, non_dominated_sort_fast


class TestCrowdingDistance(unittest.TestCase):
    def test_crowding_distance(self):
        # Test case with simple fitness values
        for i in range(1,20):
            fitness_values = np.random.rand(100, i)
            fronts1 = non_dominated_sort(fitness_values)
            fronts2 = non_dominated_sort_fast(fitness_values)

            self.assertEqual(len(fronts1), len(fronts2), 'Number of fronts do not match.')
            
            for front1, front2 in zip(fronts1, fronts2):
                np.testing.assert_array_equal(np.sort(front1), front2, 'Fronts do not match.')
            print("Test passed for {} objectives.".format(i))


if __name__ == '__main__':
    unittest.main()
