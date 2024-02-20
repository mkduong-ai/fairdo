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

The `HeuristicWrapper` is a pre-processor that is used with a given combinatorial solver
to optimize the fairness of a dataset. With this pre-processor, the user can use more advanced algorithms such as
genetic algorithms to optimize the fairness.

The `MetricOptimizer` is a pre-processor that is used with a given optimization algorithm
to optimize the fairness of a dataset.

Other pre-processors are implemented in the `base` submodule which include a `Random` pre-processor that
randomly removes data points from the dataset and a `OriginalData` that returns the original dataset.
Both of these pre-processors are used for comparison purposes.
"""
from fairdo.preprocessing.base import Preprocessing, OriginalData, Random
from fairdo.preprocessing.metricoptimizer import MetricOptimizer, MetricOptGenerator, MetricOptRemover
from fairdo.preprocessing.solverwrapper import HeuristicWrapper, DefaultPreprocessing
