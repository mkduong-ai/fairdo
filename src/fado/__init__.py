from fado import metrics
from fado import optimize
from fado import preprocessing
from fado import utils
from fado.__about__ import *

"""
Metrics
-------

The `metrics` module provides a set of functions for measuring fairness and discrimination in data.
These metrics can be utilized to evaluate the fairness of a dataset before and after preprocessing.

Optimize
--------

The `optimize` module contains methods for optimizing fairness in data.
These methods can adjust the features of data with the aim of reducing discrimination and improving fairness.

Preprocessing
-------------

The `preprocessing` module includes tools for preparing data for fairness optimization.
This encompasses functions for cleaning data, handling missing values, and encoding categorical variables.

Utils
-----

The `utils` module offers a set of utility functions that are used by the other modules in the Fado package.
These include helper functions for handling data structures, performing mathematical operations, and other common tasks.

Each of these modules plays a crucial role in the functionality of the Fado package,
working together to provide a comprehensive toolkit for fairness optimization in datasets.
"""