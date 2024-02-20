"""
Metrics
-------

The `metrics` module provides a set of functions for measuring fairness and discrimination in data.
These metrics can be utilized to evaluate the fairness of a dataset before and after preprocessing.

Optimize
--------

The `optimize` module contains methods for optimizing fairness in data.
These methods can adjust the features of data with the aim of reducing discrimination and improving fairness.

Pre-processing
--------------

The `preprocessing` module is equipped with tools designed to prepare datasets for fairness optimization.
Each method returns a pre-processed dataset that is optimized for fairness,
ensuring that any subsequent analysis or machine learning model trained on this data will
inherently be more fair and less discriminatory.

Utils
-----

The `utils` module offers a set of utility functions that are used by the other modules in the fado package.
These include helper functions for performing mathematical operations, and other common tasks.

Each of these modules plays a crucial role in the functionality of the fado package,
working together to provide a comprehensive toolkit for fairness optimization in datasets.
"""

#from fairdo import metrics
#from fairdo import optimize
#from fairdo import preprocessing
#from fairdo import utils
from fairdo.__about__ import *
