.. fado documentation master file, created by
   sphinx-quickstart on Fri Jul 28 22:17:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fado's documentation!
================================

Fairness-Agnostic Data Optimization (fado) is a Python package designed to
optimize fairness in datasets. It provides robust, fairness-agnostic methods
to preprocess data, ensuring machine learning models trained on these datasets
deliver high performance while significantly reducing discrimination.

Metrics
-------

The `metrics` module provides a set of functions for measuring fairness and discrimination in data. These metrics can be utilized to evaluate the fairness of a dataset before and after preprocessing.

Optimize
--------

The `optimize` module contains methods for optimizing fairness in data. These methods can adjust the features of data with the aim of reducing discrimination and improving fairness.

Preprocessing
-------------

The `preprocessing` module includes tools for preparing data for fairness optimization. This encompasses functions for cleaning data, handling missing values, and encoding categorical variables.

Utils
-----

The `utils` module offers a set of utility functions that are used by the other modules in the Fado package. These include helper functions for handling data structures, performing mathematical operations, and other common tasks.

Each of these modules plays a crucial role in the functionality of the Fado package, working together to provide a comprehensive toolkit for fairness optimization in datasets.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   fado.metrics
   fado.optimize
   fado.preprocessing
   fado.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
