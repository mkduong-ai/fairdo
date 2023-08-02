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
>>> # Specify your mutation function with some preset parameters
>>> my_mutation_function = partial(fractional_flip_mutation, mutation_rate=0.10)
>>> # Define f, d, n, pop_size, num_generations...
>>> # ...
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm_constraint
>>> result = genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
>>>                                       select_parents=elitist_selection,
>>>                                       crossover=onepoint_crossover,
>>>                                       mutate=my_mutation_function, maximize=False)

"""

from fado.optimize.geneticoperators.crossover import *
from fado.optimize.geneticoperators.mutation import *
from fado.optimize.geneticoperators.selection import *
