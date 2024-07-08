"""
The ``fairdo.metrics`` package provides a collection of metrics [1]_ to measure fairness/discrimination in datasets.
The metrics are divided into following subpackages:

- ``fairdo.metrics.group``: This subpackage provides metrics to measure `group fairness`.
- ``fairdo.metrics.individual``: This subpackage provides metrics to measure `individual fairness`.
- ``fairdo.metrics.dependence``: This subpackage provides metrics to measure the `(in)dependency` between two variables.
- ``fairdo.metrics.penalty``: This subpackage contains specialized penalty functions to penalize fairness metrics to\
    guarantee certain constraints such as group coverage [2]_.

Notes
-----

.. [1] These metrics are used to measure fairness in datasets. They are not to be confused with the metrics
    used to evaluate the performance of machine learning models. Strictly speaking, these metrics do not
    necessarily satisfy the properties of a metric, but they are commonly referred to as metrics in the
    fairness literature or other fairness packages [3]_.

References
----------

.. [2] Manh Khoi Duong and Stefan Conrad. (2024). Trusting Fair Data: Leveraging Quality in
    Fairness-Driven Data Removal Techniques. In DaWaK 2024: Big Data Analytics
    and Knowledge Discovery, 26th International Conference. Springer Nature Switzerland.

.. [3] Rachel K. E. Bellamy, Kuntal Dey, Michael Hind, Samuel C. Hoffman, Stephanie
    Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta,
    Aleksandra Mojsilovic, Seema Nagar, Karthikeyan Natesan Ramamurthy, John T.
    Richards, Diptikalyan Saha, Prasanna Sattigeri, Moninder Singh, Kush R. Varsh-
    ney, and Yunfeng Zhang. (2018). AI Fairness 360: An Extensible Toolkit for
    Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias. CoRR, abs/1810.01943.
"""

from .group import *
from .individual import *
from .dependence import *
from .penalty import *
