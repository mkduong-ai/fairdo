<div align="left">
<br/>
<p align="center">
<a href="https://github.com/mkduong-ai/fairdo">
<img align="center" width=80% src="https://media.githubusercontent.com/media/mkduong-ai/fairdo/main/assets/FairdoLogo.drawio.png"></img>
</a>
</p>
</div>

**FairDo** is a Python package for mitigating bias in data.
It can be used to create datasets to comply with the **AI Act**:

> [...] data sets should also have the appropriate statistical properties, including as regards the persons or groups of persons in relation to whom the high-risk AI system is intended to be used, with specific attention to the mitigation of possible biases in the data sets. [...]

- **Official repository** of: [Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes](https://link.springer.com/chapter/10.1007/978-981-99-8696-5_8)
- **Documentation**: [https://fairdo.readthedocs.io/en/latest/](https://fairdo.readthedocs.io/en/latest/)
- **Source Code**: [https://github.com/mkduong-ai/fairdo/tree/main](https://github.com/mkduong-ai/fairdo/tree/main)


## Why FairDo?
- Fairness: Minimizes discrimination in datasets
- Interpretability and integrity: **Under**- and **oversampling** technique
- Works with tabular data: `pandas.DataFrame`
- Simplicity: Follows `.fit_transform()` convention and includes many examples
- Handles a variety of cases: **non-binary groups, multiple protected attributes, individual fairness** 
- Customizable: custom fairness definition, solver, objective

## How does it work?
The pre-processing methods (`fairdo.preprocessing.HeuristicWrapper` and `fairdo.preprocessing.DefaultPreprocessing`)
work by **removing discriminatory data points**.
By doing so, the dataset becomes much more balanced and less biased towards
a particular social group.
We approach this task as a **combinatorial optimization problem**, which
means selecting a subset of the dataset that minimizes the discrimination score.
Because there are exponentially many subsets,
our approach uses **genetic algorithms**.

## Quick Example

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

## Advices
:rocket: **For a quick start**, use the `DefaultPreprocessing` class with the default settings.
An example is given in `tutorials/1. Default Preprocessor`.

:white_check_mark: **For data quality**, you want to keep the original data $D$
and only add fair synthetic data $G$ on top of it.
(We include examples in `tutorials/` where we use the [SDV](https://github.com/sdv-dev/SDV) package
to generate synthetic data.)
For this, you need to specify
`.fit_transform(approach='add')` for the pre-processors `fairdo.preprocessing.HeuristicWrapper` and
`fairdo.preprocessing.DefaultPreprocessing`.
This will only add the fair pre-processed synthetic data
to the original data.

:dash: **When having limited data**, we advise employing synthetic data $G$ additionally
and merge it with the original data $D$, i.e., $D \cup G$.
The pre-processor can then be used on the merged data $D \cup G$ to ensure fairness.
It is also possible to use the methodology for **data quality**, as described above.

:briefcase: **When anonymity is required**, only use synthetic data $G$ and do not
merge it with the original data $D$. The generated data $G$ can then be
pre-processed with our methods to ensure fairness.


## Installation
First, setup a Python environment. We recommend using [Miniconda](https://docs.anaconda.com/free/miniconda/index.html). Activate the created environment afterwards and finally install our package.
A detailed guide is given as follows.

### Dependencies
Python (==3.8), `numpy`, `scipy`, `pandas`, `sklearn`

### Setup Conda Environment
Download Miniconda [here](https://docs.anaconda.com/free/miniconda/index.html).

```bash
# Create a conda virtual environment
conda create -n "venv" python=3.8

# Activate conda environment
conda activate venv
```

**OR**

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

To install the latest (development) version, execute following commands:
```bash
# Clone repo
git clone https://github.com/mkduong-ai/fairdo.git

# Move to repo folder
cd fairdo

# Install from source
python setup.py install
```

### Install Optional Dependencies

To use the synthetic data generation, you can install the `SDV` package
by executing the following command:
```bash
pip install sdv==1.10.0
```

We did not include the `SDV` package as a dependency, because it is not required
for the core functionality of the **FairDo** package.
Using any other synthetic data generation package is also possible.
Still, some examples in the `tutorials/` folder require the `SDV` package.

## Citation

When using **FairDo** in your work, cite our paper:

```BibTeX
@inproceedings{duong2023framework,
  title={Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes},
  author={Duong, Manh Khoi and Conrad, Stefan},
  booktitle={Data Science and Machine Learning},
  publisher={Springer Nature Singapore},
  number={CCIS 1943},
  series={AusDM: Australasian Conference on Data Science and Machine Learning},
  year={2023},
  pages={105--120},
  isbn={978-981-99-8696-5},
}
```

## Notes
We credit OpenMoji for the emojis used in our logo.
