# Fairness-Agnostic Data Optimization
**fado** is a Python package for optimizing datasets on fairness notions.
The approaches are _fairness-agnostic_: Any fairness criterion that
measures discrimination on a dataset can be optimized on.
The algorithms return a debiased dataset according
to the given fairness notion.

The resulting datasets do not come with qualitative compromises.
Machine Learning estimators trained on debiased datasets
perform similarly to the original datasets w.r.t. performance measures while
notably reducing the discrimination.

## Installation

### Dependencies
Python (>=3.8, <4), numpy, pandas, scikit-learn, copulas

### Manual Installation

```bash
python setup.py install
```

### Development

```python
pip install -e.
```

## Example Usage

In the following example, we use the iris dataset. The protected attribute
is the species and the label is petal length. We binarize both features.

```python
# Imports
from fado.preprocessing import MetricOptimizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Loading a sample database and encoding for appropiate usage
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
le = LabelEncoder()
iris_transform = iris.apply(le.fit_transform)
iris_transform['species'] = iris_transform['species'] == 0
iris_transform['petal_length'] = iris_transform['petal_length'] > 9

# Initialize
preproc = MetricOptimizer(frac=0.75,
                          protected_attribute='species',
                          label='petal_length')
                          
iris_fair = preproc.fit_transform(iris_transform)
```

More ``jupyter notebooks`` examples can be viewed in ``tutorials/``.


## Evaluation

Create results
```bash
cd evaluation
python run_evaluation.py
```
The results are saved under ``evaluation/results/...``.

Create plots from results
```bash
cd evaluation
python create_plots.py
```
