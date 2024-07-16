"""
This module contains optimization methods for single-objective optimization problems.
The optimization algorithms are designed to minimize an objective function that takes a binary vector as input.
The optimization problem is defined as follows:

.. math::

        \min_{\mathbf{x} \in \{0, 1\}^d} \quad f(\mathbf{x})

where :math:`f` is the objective function to minimize.
"""
from .baseline import ones_array_method, random_bits_method, random_bits_method_vectorized, brute_force
from .ga import genetic_algorithm
