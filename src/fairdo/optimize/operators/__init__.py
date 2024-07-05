"""
Genetic Operators
========================

This package contains numerous methods for different genetic operators.
The methods are implemented in respective submodules:
``initialization``, ``selection``, ``crossover``, and ``mutation``.

Notes
-----
To change the default settings of the genetic method,
use the built-in function ``functools.partial`` to set different parameters.

Example
-------
>>> from functools import partial
>>> from fairdo.optimize.single import genetic_algorithm
>>> from fairdo.optimize.operators import onepoint_crossover,\\
fractional_flip_mutation, elitist_selection
>>> # Specify your mutation function with customized parameters
>>> my_mutation_function = partial(fractional_flip_mutation, mutation_rate=0.10)
>>> # Define f, d, n, pop_size, num_generations...
>>> f = lambda x: 1 if x[0] == 1 and x[1] == 1 else 0
>>> d = 2
>>> n = 0 # no constraints
>>> pop_size = 100
>>> num_generations = 100
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm
>>> result = genetic_algorithm(f, d, pop_size, num_generations,
>>>                            selection=elitist_selection,
>>>                            crossover=onepoint_crossover,
>>>                            mutation=my_mutation_function, maximize=False)

"""
from .initialization import *
from .crossover import *
from .mutation import *
from .selection import *
