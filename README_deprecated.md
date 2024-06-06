### MetricOptimizer
In the following example, we use the COMPAS [1] dataset.
The protected attribute is race and the label is recidivism.
Here, we use the package's own heuristic to yield for fair data.
25% synthetic data is added to reduce bias in this example:

```python
# Imports
from fairdo.preprocessing import MetricOptimizer
from fairdo.utils.dataset import load_data

# Loading a sample database and encoding for appropriate usage
# data is a pandas dataframe
data, label, protected_attributes = load_data('compas')

# Initialize MetricOptimizer
preproc = MetricOptimizer(frac=1.25,
                          protected_attribute=protected_attributes[0],
                          label='label')
                          
data_fair = preproc.fit_transform(data)
```

More ``jupyter notebooks`` examples can be viewed in ``tutorials/``.


## Evaluation

As the evaluation script depends on other algorithms, it is necessary to install
the appropriate packages by:

```bash
cd evaluation
pip install -r requirements.txt
```

### Evaluate Heuristics for Non-Binary Protected Attribute Fairness

To evaluate the heuristics for non-binary protected attributes, run the
following command:
```bash
python evaluation/nonbinary/quick_eval.py
```
Experiments on tuning population size and number of generations
as well as comparing different operators and heuristics can all be done
in `quick_eval.py`. Modify the function `run_and_save_experiment` by
renaming the appropriate settings function
`setup_experiment`/`setup_experiment_hyperparameter`.
Although the experiments make use of multiprocessing,
it runs through all settings, heuristics, datasets, trials and can
therefore take a while.

After the results are exported, plots can be created by running:
```bash
python evaluation/nonbinary/create_plots.py
```

### Evaluate MetricOptimizer

To evaluate MetricOptimizer, run the following command:

```bash
python evaluation/run_evaluation.py
```
The results are saved under ``evaluation/results/...``.

Create plots from results
```bash
python evaluation/create_plots.py
```
The plots are stored in the same directory as their corresponding .csv file.

To modify or change several settings (datasets, metrics, #runs) in the
evaluation, change the file ``evaluation/settings.py``.

## Documentation

The documentation is available at [https://fairdo.readthedocs.io/en/latest/](https://fairdo.readthedocs.io/en/latest/).
To build the documentation manually, follow this guide:

The package follows the PEP8 style guide and is documented with NumPy style
DocStrings. To build the HTML pages from the documentation manually,
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

### Development Installation

Installing in development mode is useful to make changes in the source code take effect instantly.
This means that the package is installed in such a way that changes to the source code
are immediately reflected without the need to reinstall the package. This can be done in the following
way:
```bash
# Clone repo
git clone https://github.com/mkduong-ai/fairdo.git

# Move to repo folder
cd fairdo

# Development installation
pip install -e.
```


## References
[1] Larson, J., Angwin, J., Mattu, S.,  Kirchner, L.: Machine
bias (May 2016),
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

[2] Rawls, J.: A Theory of Justice (1971), Belknap Press, ISBN: 978-0-674-00078-0
