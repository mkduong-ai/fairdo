"""
Optimize Module
===============

The Optimize module provides methods for optimizing fairness in datasets.
**The solvers are used with Preprocessors** to optimize the fairness of a dataset.
All solvers take an objective function :math:`f`, number of dimensions of the problem :math:`d`,
and specific hyperparameters as parameters.

Wrapping it as a pre-processing method, all solvers return a **binary mask** that can be used to filter the dataset.

The Optimize module is divided into two submodules:

1. `baseline`: This submodule provides baseline methods for fairness optimization.
These methods serve as a starting point for fairness optimization and can be used for comparison
with more advanced methods.

2. `geneticalgorithm`: This submodule provides a genetic algorithm for fairness optimization.
Genetic algorithms are a type of evolutionary algorithm that can find solutions to optimization problems that
are difficult or impossible to solve with traditional methods, e.g., multiple local optima, non-differentiable,
non-continuous, or discontinuous fitness functions. (Note that fitness functions is used as a term in this context and
are equivalent to objective functions.)

Example
-------
>>> from fairdo.optimize import genetic_algorithm_constraint
>>> from fairdo.optimize.geneticoperators import onepoint_crossover, fractional_flip_mutation, elitist_selection
>>> # Define f, d, n, pop_size, num_generations...
>>> f = lambda x: 1 if x[0] == 1 and x[1] == 1 else 0
>>> d = 2
>>> n = 0 # no constraints
>>> pop_size = 100
>>> num_generations = 100
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm_constraint
>>> best_solution, fitness = genetic_algorithm_constraint(f, d, n, pop_size, num_generations,
>>>                                                       select_parents=elitist_selection,
>>>                                                       crossover=onepoint_crossover,
>>>                                                       mutate=fractional_flip_mutation, maximize=False)

"""

from fairdo.optimize.baseline import *
from fairdo.optimize.geneticalgorithm import *
