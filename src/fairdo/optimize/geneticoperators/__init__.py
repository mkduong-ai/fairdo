"""
Genetic Operators Module
========================

The Genetic Operators module provides methods for genetic operators used in the genetic algorithm.
It is divided into three submodules: `crossover`, `mutation`, and `selection`.
If the user wants to use different parameter settings for the provided genetic operators,
they can do so by using Python's `functools.partial` method.

Example
-------
>>> from functools import partial
>>> from fairdo.optimize import genetic_algorithm_constraint
>>> from fairdo.optimize.geneticoperators import onepoint_crossover, fractional_flip_mutation, elitist_selection
>>> # Specify your mutation function with some preset parameters
>>> my_mutation_function = partial(fractional_flip_mutation, mutation_rate=0.10)
>>> # Define f, d, n, pop_size, num_generations...
>>> f = lambda x: 1 if x[0] == 1 and x[1] == 1 else 0
>>> d = 2
>>> n = 0 # no constraints
>>> pop_size = 100
>>> num_generations = 100
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm_constraint
>>> result = genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
>>>                                       select_parents=elitist_selection,
>>>                                       crossover=onepoint_crossover,
>>>                                       mutate=my_mutation_function, maximize=False)

"""

from fairdo.optimize.geneticoperators.crossover import *
from fairdo.optimize.geneticoperators.mutation import *
from fairdo.optimize.geneticoperators.selection import *
