"""
Preprocessing Module
====================

The Preprocessing module provides methods for pre-processing datasets to achieve fairness.
The base class `Preprocessing` defines the required methods for a pre-processor,
which are `fit` and `transform`. The `fit` method is used to assign the dataset to an internal variable,
and the `transform` method is used to apply the pre-processing method to the dataset. The `transform` method
returns a pre-processed version of the dataset that can be used for training a machine learning model.
The pre-processed dataset is considered fair with respect to the discrimination measure that is given
when initializing the pre-processor. The discrimination measure is a metric from the `fairdo.metrics` module.

The `DefaultPreprocessing` object is the default pre-processor that internally uses
a genetic algorithm to optimize the fairness of a dataset. It comes with default parameters
that are optimal for most use cases. The user can also manually set the population size
and number of generations when initializing the `DefaultPreprocessing` object.

The `HeuristicWrapper` is a pre-processor that is used with a given combinatorial optimization solver
to optimize the fairness of a dataset. The user can use any heuristic solver from the `fairdo.optimize` module.
This requires manually setting the parameters of the heuristic solver
and is recommended for advanced users.

The `MetricOptimizer` is a pre-processor that is used with a given optimization algorithm
to optimize the fairness of a dataset. This pre-processor is **deprecated**. Use `DefaultPreprocessing` instead.

Other pre-processors are implemented in the `base` submodule which include a `Random` pre-processor that
randomly removes data points from the dataset, a `OriginalData` that returns the original dataset,
and `Unawareness` that removes all columns of protected attributes from the dataset.
These pre-processors are used for comparison purposes.
"""
from fairdo.preprocessing.base import Preprocessing, OriginalData, Unawareness, Random
from fairdo.preprocessing.metricoptimizer import MetricOptimizer, MetricOptGenerator, MetricOptRemover
from fairdo.preprocessing.solverwrapper import HeuristicWrapper, DefaultPreprocessing
