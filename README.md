# Fairness-Agnostic Data Optimization
**FairDo** is a Python package for mitigating bias in data.
The approaches, which are _fairness-agnostic_, enable optimization of diverse
fairness criteria quantifying discrimination within datasets,
leading to the generation of biased-reduced datasets.
Our framework is able to deal with non-binary protected attributes
such as nationality, race, and gender that naturally arise in many
applications.
Due to the possibility to choose between any of the available fairness metrics,
it is possible to aim for the least fortunate group
(Rawls' A Theory of Justice [2]) or the general utility of all groups
(Utilitarianism).

## Installation

### Dependencies
Python (>=3.8, <4), `numpy`, `pandas`, `sklearn`, `sdv`

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

### PyPI Distribution

The package is distributed via PyPI and can be installed with:
```bash
pip install fairdo
```

### Manual Installation

```bash
python setup.py install
```

### Development Installation

```python
pip install -e.
```

## Example Usage

### Genetic Algorithms
In the following example, we use the COMPAS [1] dataset.
The protected attribute is race and the label is recidivism.
Here, we deploy a genetic algorithm to remove discriminatory samples
of the merged original and synthetic dataset:

```python
# Standard library
from functools import partial

# Related third-party imports
from sdv.tabular import GaussianCopula
import pandas as pd

# fairdo package
from fairdo.utils.dataset import load_data
from fairdo.preprocessing import HeuristicWrapper
from fairdo.optimize.geneticalgorithm import genetic_algorithm
from fairdo.metrics import statistical_parity_abs_diff_max

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas')

# Create synthetic data
gc = GaussianCopula()
gc.fit(data)
data_syn = gc.sample(data.shape[0])

# Merge/concat original and synthetic data
data = pd.concat([data, data_syn.copy()], axis=0)

# Initial settings for the Genetic Algorithm
ga = partial(genetic_algorithm,
             pop_size=100,
             num_generations=100)
             
# Optimization step
preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                disc=statistical_parity_abs_diff_max)
data_fair = preprocessor.fit_transform(dataset=data,
                                       approach='remove')                                
```


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

Run document generation script:
```bash
# Move to /docs
cd /docs

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
