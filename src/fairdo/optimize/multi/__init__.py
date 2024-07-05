"""
This module contains implementations of multi-objective optimization algorithms.
All algorithms are designed to minimize objective functions which
take a binary vector as input.
The multi-objective optimization problem is defined as follows:

.. math::

        \min_{\mathbf{x} \in \{0, 1\}^d} \quad (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_n(\mathbf{x}))

where :math:`f_1, f_2, \ldots, f_n` are objective functions.
"""
from .nsga import nsga2
