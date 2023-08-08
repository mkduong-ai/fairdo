"""
Metrics Module
==============

The Metrics module provides a collection of metrics to measure fairness and discrimination in datasets.
It is divided into four submodules:

1. `dataset`: This submodule provides metrics to evaluate fairness at the dataset level.
Fairness is measured by comparing the distribution of the label :math:`y` across different groups defined by
a protected attribute :math:`z`., i.e., :math:`P(y|z)`.
:math:`y` can be the true label or a prediction, and :math:`z` can be a single protected attribute
or a list of protected attributes depending on the metric.

2. `independence`: This submodule provides metrics (derived from statistics) to measure the independence
between two variables. They are used to measure the independence between the label :math:`y` and the
protected attribute :math:`z`.
They are similar to the metrics in the `dataset` submodule when it comes to the requirements.

3. `individual`: This submodule provides metrics to measure individual fairness, i.e.,
similar individuals should be treated similarly. Metrics in this submodule require the label :math:`y`
and the data :math:`x`.
:math:`x` is a matrix of individuals and features (n_individuals, n_features),
and :math:`y` can be the true label or a prediction.

4. `prediction`: This submodule provides metrics to evaluate the fairness of predictions made by a
machine learning model. This submodule requires the true label :math:`y_{true}`,
the predicted label :math:`y_{pred}`.

Each submodule provides a different perspective on fairness, and together they provide a comprehensive toolkit
for measuring fairness in datasets.
"""

from fairdo.metrics.dataset import *
from fairdo.metrics.independence import *
from fairdo.metrics.individual import *
from fairdo.metrics.prediction import *
