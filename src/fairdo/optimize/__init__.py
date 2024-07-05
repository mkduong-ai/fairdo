"""
Optimization Algorithms
=======================

The ``optimize`` package provides optimization algorithms (single- and multi-objective) that take an objective function
or multiple objective functions and the dimensionality of the problem
as inputs and returns the best solution found by the algorithm
as a numpy array.

The algorithms can be used with classes in ``preprocessing.wrapper`` to optimize the fairness of a dataset.
Most solvers return a **binary mask** that is used to filter the dataset.
Let :math:`\psi` be the objective function, :math:`\mathcal{D}` be the dataset,
and :math:`\mathbf{b}` be the binary mask, then the optimization problem is defined as follows [1]_:

.. math::

    \\mathbf{b}^* = \\arg\\!\\min_{\\mathbf{b} \\in \\{0, 1\\}^d} \\quad \\psi(\\{d_i \\in \\mathcal{D} \\mid b_i = 1\\}).

The ``optimize`` module is divided into following subpackages:

- ``single``: This subpackage contains single-objective optimization algorithms.
- ``multi``: This subpackage contains multi-objective optimization algorithms.
- ``operators``: This subpackage contains ``initialization``, ``selection``, ``crossover``, and ``mutation`` operators used in genetic algorithms.

References
----------

.. [1] Manh Khoi Duong and Stefan Conrad, "Towards Fairness and Privacy:
    A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes",
    Data Science and Machine Learning. AusDM 2023.
    Communications in Computer and Information Science, vol 1943. Springer, Singapore.
    https://doi.org/10.1007/978-981-99-8696-5_8

Example
-------
>>> from fairdo.optimize.single import genetic_algorithm
>>> from fairdo.optimize.operators import onepoint_crossover,\\
fractional_flip_mutation, elitist_selection
>>> # Define f, d, n, pop_size, num_generations...
>>> f = lambda x: 1 if x[0] == 1 and x[1] == 1 else 0
>>> d = 2
>>> pop_size = 100
>>> num_generations = 100
>>> # Now you can use my_mutation_function as an argument to genetic_algorithm
>>> best_solution, fitness = genetic_algorithm(f, d, pop_size, num_generations,
>>>                                            selection=elitist_selection,
>>>                                            crossover=onepoint_crossover,
>>>                                            mutation=fractional_flip_mutation,
>>>                                            maximize=False)

"""
from . import single, multi, operators
