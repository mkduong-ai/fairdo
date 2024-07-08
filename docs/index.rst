.. fairdo documentation master file, created by
   sphinx-quickstart on Fri Jul 28 22:17:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FairDo Documentation
====================

**Fairness-Agnostic Data Optimization (FairDo)** is a Python package for
mitigating bias in datasets. It provides robust *fairness-agnostic* methods
to pre-process data. Machine learning models trained on these datasets
do not come with compromises in performance but significantly discriminate less.

The pipeline to mitigate bias in datasets consists of three main steps:

1. ``fairdo.metrics``: Select a fairness metric to evaluate the dataset.
2. ``fairdo.optimize``: Select optimization method.
3. ``fairdo.preprocessing``: Choose pre-processing method with selected metric and optimizer
   and apply it to the dataset. All pre-processors come with ``.fit()``, ``.transform()``,
   ``.fit_transform()`` interfaces.


.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   fairdo.metrics
   fairdo.optimize
   fairdo.preprocessing
   fairdo.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
