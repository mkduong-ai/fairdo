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

### Installation Manual

### Developer

```python
pip install -e.
```

## Usage

## Evaluation

Create results
```python
python main.py
```

Create plots from results
```python
python evaluation/plot_settings.py
```
