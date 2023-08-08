# Fairness-Agnostic Data Optimization
**FairDo** is a Python package for optimizing datasets on fairness criteria.
The approaches are _fairness-agnostic_, meaning it can optimize any fairness
criterion that quantifies discrimination within a dataset.
The algorithms return a biased-reduced dataset accordingly.

In our experiments, the debiased datasets do not come with significant
qualitative compromises. Machine Learning estimators trained on our debiased datasets
perform similarly to the original datasets w.r.t. performance measures (AUC, Accuracy)
while notably reducing the discrimination.

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
is the species and the label is the petal length. We binarize both features.
The iris dataset is not biased nor a person-related dataset, but we use it as
an example to show the functionality of the package.

```python
# Imports
from fairdo.preprocessing import MetricOptimizer
from fairdo.utils.dataset import load_data

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas')

# Initialize MetricOptimizer
preproc = MetricOptimizer(frac=0.75,
                          protected_attribute=protected_attributes[0],
                          label='label')
                          
data_fair = preproc.fit_transform(data)
```

More ``jupyter notebooks`` examples can be viewed in ``tutorials/``.


## Evaluation

As the evaluation script depends on other algorithms, it is necessary to install the appropriate packages by:

```bash
cd evaluation
pip install -r requirements.txt
```

### Evaluate MetricOptimizer

To evaluate the MetricOptimizer, run the following command:

```bash
python evaluation/run_evaluation.py
```
The results are saved under ``evaluation/results/...``.

Create plots from results
```bash
python evaluation/create_plots.py
```
The plots are stored in the same directory as their corresponding .csv file.

To modify or change several settings (datasets, metrics, #runs) in the evaluation,
change the file ``evaluation/settings.py``.

### Evaluate Heuristics for Non-Binary Protected Attribute Fairness

To evaluate the heuristics for non-binary protected attributes, run the following command:
```bash
python evaluation/nonbinary/quick_eval.py
```

After, results are exported, plots can be created by running:
```bash
python evaluation/nonbinary/create_plots.py
```
