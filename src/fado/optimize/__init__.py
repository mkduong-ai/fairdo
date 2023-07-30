"""
Optimize Module
===============

The Optimize module provides methods for optimizing fairness in datasets.
**The solvers are used with Preprocessors** to optimize the fairness of a dataset.
All solvers take an objective function :math:`f`, number of dimensions of the problem :math:`d`,
and specific hyperparameters as parameters.

Wrapping it as a pre-processing method, all solvers return a binary mask that can be used to filter the dataset.

The Optimize module is divided into two submodules:

1. `baseline`: This submodule provides baseline methods for fairness optimization.
These methods serve as a starting point for fairness optimization and can be used for comparison
with more advanced methods.

2. `geneticalgorithm`: This submodule provides a genetic algorithm for fairness optimization.
Genetic algorithms are a type of evolutionary algorithm that can find solutions to optimization problems that
are difficult or impossible to solve with traditional methods, e.g., multiple local optima, non-differentiable,
non-continuous, or discontinuous fitness functions. (Note that fitness functions is used as a term in this context and
are equivalent to objective functions.)
"""

from fado.optimize.baseline import *
from fado.optimize.geneticalgorithm import *
