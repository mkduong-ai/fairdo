# Fairness-Agnostic Data Optimization

Official repository of [Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes](https://link.springer.com/chapter/10.1007/978-981-99-8696-5_8)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/mkduong-ai/fairdo">
<img align="center" width=100% src="https://github.com/mkduong-ai/fairdo/blob/main/assets/Pipeline.png"></img>
</a>
</p>
</div>

**FairDo** is a Python package for mitigating bias in data.
It works specifically for tabular data (`pandas.DataFrame`) where the data is pre-processed in
such a way that it becomes fair according to a user-given fairness metric.
The pre-processing approach is **fairness-agnostic**, enabling the optimization
of different fairness criteria.
Our framework is able to deal with **non-binary** protected attributes
such as nationality, race, and gender that naturally arise in many
datasets.
Due to the possibility of choosing between any of the available fairness metrics,
it is possible to aim for the least fortunate group
(Rawls' A Theory of Justice [2]) or the general utility of all groups
(Utilitarianism).

The pre-processing methods work by **removing discriminatory data points**.
By doing so, the dataset becomes much more balanced and less biased towards
a particular social group.
We approach this task as a **combinatorial optimization problem**, which
means selecting a subset of the dataset that minimizes the discrimination score.
Because there are exponentially many possibilities for selecting a subset,
our approach uses **genetic algorithms** to find a fair subset.

## Installation

### Dependencies
Python (>=3.8, <=3.9), `numpy`, `pandas`, `sklearn`, `sdv`

### Setup Python Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS and Linux:
source .venv/bin/activate
```

### PyPI Distribution (recommended)

The package is distributed via PyPI and can be directly installed with:
```bash
pip install fairdo
```

### Manual Installation (latest version)

To install the latest version, execute following commands:
```bash
# Clone repo
git clone https://github.com/mkduong-ai/fairdo.git

# Move to repo folder
cd fairdo

# Install from source
python setup.py install
```

### Development Installation

Installing in development mode is useful to make changes in the source code take effect instantly.
This means that the package is installed in such a way that changes to the source code
are immediately reflected without the need to reinstall the package. This can be done in the following
way:
```python
# Clone repo
git clone https://github.com/mkduong-ai/fairdo.git

# Move to repo folder
cd fairdo

# Development installation
pip install -e.
```

## Example Usage

In the following example, we use the COMPAS [1] dataset.
The protected attribute is _race_ and the label is _recidivism_.
Here, we deploy the **default pre-processor**, which internally uses a genetic algorithm,
to remove discriminatory samples of the given dataset.
The default pre-processor prevents removing all individuals of a single group.

```python
# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import DefaultPreprocessing
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample dataset with all required information
# data is a pandas.DataFrame
data, label, protected_attributes = load_data('compas', print_info=False)

# Initialize DefaultPreprocessing object
preprocessor = DefaultPreprocessing(protected_attribute=protected_attributes[0],
                                    label=label)

# Fit and transform the data
data_fair = preprocessor.fit_transform(dataset=data)

# Print no. samples and discrimination before and after
disc_before = statistical_parity_abs_diff_max(data[label],
                                              data[protected_attributes[0]].to_numpy())
disc_after = statistical_parity_abs_diff_max(data_fair[label],
                                             data_fair[protected_attributes[0]].to_numpy())
print(len(data), disc_before)
print(len(data_fair), disc_after)
```

By running this example, the **resulting dataset** usually has a statistical **disparity score of <1%** (max. score between all five races),
while the **original dataset exhibits 27% statistical disparity**.

## Documentation

The package follows the PEP8 style guide and is documented with NumPy style
docstrings. To view the HTML pages of the documentation,
follow these instructions:

Activate virtual environment and install sphinx.
```bash
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS and Linux:
source .venv/bin/activate

# Install Sphinx and a required theme
pip install sphinx furo
```

Run document generation script in UNIX-Systems:
```bash
# Move to /docs
cd docs

# Run script to generate documentation
bash generate_docs.sh
```

The HTML pages are then located in `docs/_build/html`.
Open `docs/_build/html/index.html` to view the front page.

## Citation

When using **FairDo** in your work, cite our paper:

```
@inproceedings{duong2023framework,
  title={Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes},
  author={Duong, Manh Khoi and Conrad, Stefan},
  booktitle={The 21st Australasian Data Mining Conference 2023},
  year={2023},
  organization={Springer Nature}
}
```

## References
[1] Larson, J., Angwin, J., Mattu, S.,  Kirchner, L.: Machine
bias (May 2016),
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

[2] Rawls, J.: A Theory of Justice (1971), Belknap Press, ISBN: 978-0-674-00078-0
