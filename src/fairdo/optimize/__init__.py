"""
Optimize Module
===============

The ``optimize`` module provides optimization algorithms (single- and multi-objective) that take an objective function or multiple objective functions
and the dimensionality of the problem
as inputs and returns the best solution found by the algorithm
as a numpy array.

The solvers can be used with classes in ``preprocessing`` to optimize the fairness of a dataset.

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
>>> from fairdo.optimize.so import genetic_algorithm
>>> from fairdo.optimize.geneticoperators import onepoint_crossover, fractional_flip_mutation, elitist_selection
>>> # Define f, d, n, pop_size, num_generations...
>>> f = lambda x: 1 if x[0] == 1 and x[1] == 1 else 0
>>> d = 2
>>> pop_size = 100
>>> num_generations = 100
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm
>>> best_solution, fitness = genetic_algorithm_constraint(f, d, pop_size, num_generations,
>>>                                                       selection=elitist_selection,
>>>                                                       crossover=onepoint_crossover,
>>>                                                       mutate=fractional_flip_mutation, maximize=False)

"""
from . import baseline, single, multi
from fairdo.optimize.baseline import *
from fairdo.optimize.single import genetic_algorithm
from fairdo.optimize.multi import nsga2, dom_counts_indices, dom_counts_indices_fast, crowding_distance