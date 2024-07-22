"""
This module contains utility functions to load, preprocess, and synthesize datasets.
"""

# Standard library imports
import io
import zipfile

# Related third-party imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from requests import get

# Attempt to import (optional) sdv libraries
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    sdv_installed = True
except ModuleNotFoundError:
    sdv_installed = False


def downcast(data):
    """
    Downcast float and integer columns of the given data to save memory.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame to downcast.

    Returns
    -------
    data : pandas DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> from fairdo.utils.dataset import downcast
    >>> data = pd.DataFrame({'a': [1, 2], 'b': [1.0, 2.0]})
    >>> data = downcast(data)
    >>> print(data.dtypes)
    a      int8
    b    float32
    """
    fcols = data.select_dtypes('float').columns
    icols = data.select_dtypes('integer').columns

    data[fcols] = data[fcols].apply(pd.to_numeric, downcast='float')
    data[icols] = data[icols].apply(pd.to_numeric, downcast='integer')

    return data


def dataset_intersectional_column(data, protected_attributes):
    """
    Combine the protected attributes into a single column named ``subgroup``.
    This column will be used to identify the intersectional groups.

    Parameters
    ----------
    data: pandas DataFrame
        DataFrame with protected attributes.
    protected_attributes : list of str
        List of protected attributes. Each attribute should be a column in the data.

    Returns
    -------
    data : pandas DataFrame
        Returns a DataFrame with an extra column of combined protected attributes.
    protected_attribute : str
        The name of the column with the combined protected attributes.

    Examples
    --------
    >>> import pandas as pd
    >>> from fairdo.utils.dataset import dataset_intersectional_column
    >>> data = pd.DataFrame({'sex': ['male', 'female'], 'race': ['white', 'black']})
    >>> pas = ['sex', 'race']
    >>> data_new, pa = dataset_intersectional_column(data, pas)
    >>> print(data_new)
          sex   race       subgroup
    0    male  white    male_white_
    1  female  black  female_black_
    >>> print(pa)
    subgroup
    """
    protected_attribute = 'subgroup'
    
    # Initialize the protected attribute column with empty strings
    data[protected_attribute] = ''
    
    for col in protected_attributes:
        data[protected_attribute] += data[col].astype(str) + '_'
    
    return data, protected_attribute


def load_data(dataset_str, multi_protected_attr=False, print_info=True):
    """
    Load the dataset and preprocess it. The preprocessing steps include:

    - Dropping rows with missing values
    - Label encode protected attributes and label
    - One-hot encode all other categorical variables
    - Downcast float and integer columns to save memory

    Parameters
    ----------
    dataset_str : str
        Name of the dataset to load and preprocess  (e.g., 'adult', 'compas', 'bank', 'german').
    multi_protected_attr : bool
        Whether to use multiple protected attributes or not.
    print_info : bool
        Whether to print information about the dataset or not.

    Returns
    -------
    df : pandas DataFrame
        Preprocessed DataFrame.
    label : str
        Name of the label column.
    protected_attributes : list of str
        List of protected attributes.

    Examples
    --------
    >>> from fairdo.utils.dataset import load_data
    >>> data, label, protected_attributes = load_data('adult')
    >>> print(data.head(2))
       age  education-num  race  ...  relationship_ Wife  sex_ Female  sex_ Male
    0   39             13     4  ...                   0            0          1
    1   50             13     4  ...                   0            0          1
    >>> print(label)
    income
    >>> print(protected_attributes)
    ['race']
    """
    if dataset_str == 'adult':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None,
                           names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                  "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                  "hours-per-week", "native-country", "income"])
        print('Data downloaded.')

        # Drop columns
        cols_to_drop = ['fnlwgt', 'workclass', 'education', 'occupation', 'native-country']
        data = data.drop(columns=cols_to_drop)

        # Label encoding protected_attribute and label
        label = 'income'
        if multi_protected_attr:
            protected_attributes = ['race', 'sex']
        else:
            protected_attributes = ['race']

    elif dataset_str == 'compas':
        use_cols = ['race', 'sex', 'juv_fel_count', 'decile_score',
                    'juv_misd_count', 'juv_other_count', 'is_violent_recid', 'v_decile_score', #'is_recid',
                     'priors_count', 'age_cat', 'c_charge_degree', 'two_year_recid']
        data = pd.read_csv(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            usecols=use_cols)
        print('Data downloaded.')

        # Drop rows with missing values
        data = data.dropna(axis=0, how='any')

        # Label encoding protected_attribute and label
        label = 'two_year_recid'
        if multi_protected_attr:
            protected_attributes = ['race', 'sex', 'age_cat']
        else:
            protected_attributes = ['race']

    elif dataset_str == 'bank':
        # Loading Bank Marketing dataset
        r = get("http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open('bank-additional/bank-additional-full.csv') as csv_file:
            data = pd.read_csv(csv_file, delimiter=';')
        print('Data downloaded.')

        # Drop rows with missing values
        data = data.dropna(axis=0, how='any')

        # Label encoding protected_attribute and label
        label = 'y'
        if multi_protected_attr:
            protected_attributes = ['job', 'marital']
        else:
            protected_attributes = ['job']

    elif dataset_str == 'german':
        # Loading German Credit dataset
        data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                           header=None, delim_whitespace=True)
        print('Data downloaded.')
        # Preprocessing steps for German Credit dataset
        # ...
        pass
        # Define label and protected attributes for German Credit dataset
        label = 'credit_risk'
        protected_attributes = ['age', 'gender']
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # Label encoding protected_attribute and label
    cols_to_labelencode = protected_attributes.copy()
    cols_to_labelencode.append(label)
    data[cols_to_labelencode] = \
        data[cols_to_labelencode].apply(LabelEncoder().fit_transform)
    
    # Encode categorical variables as one-hot
    categorical_cols = list(data.select_dtypes(include='object').columns)
    data = pd.get_dummies(data, columns=categorical_cols)

    # Downcast
    data = downcast(data)
    
    # print info of the data
    if print_info:
        print(data[protected_attributes].iloc[:, 0].unique())
        print(data[protected_attributes].iloc[:, 0].value_counts())
        print(data.shape)

    return data, label, protected_attributes


def generate_data(data, num_rows=100):
    """
    Generate synthetic data using the sdv library.
    The method used is Gaussian Copula.

    Parameters
    ----------
    data : pandas DataFrame
        The real data to be used to generate synthetic data
    num_rows : int
        The number of rows to generate

    Returns
    -------
    synthetic_data : pandas DataFrame or None
        The synthetic data generated

    Examples
    --------
    >>> import pandas as pd
    >>> from fairdo.utils.dataset import generate_data
    >>> data = pd.DataFrame({'age': [39, 50], 'education': ['Bachelors', 'HS-grad'], 'income': ['<=50K', '<=50K']})
    >>> generate_data(data, num_rows=2)
         age  education  income
    0  39.0  Bachelors  <=50K
    1  50.0    HS-grad  <=50K
    """
    if not sdv_installed:
        # Inform the user that sdv library is required
        print("The 'sdv' library is required to generate synthetic data.")
        print("Please install it by running: pip install sdv==1.10.0")
        return None

    # Fit the synthesizer to the real data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)

    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data


def data_generator(data):
    """
    Returns the data generator, from which the user can generate synthetic data

    Parameters
    ----------
    data : pandas DataFrame
        The real data to be used to generate synthetic data

    Returns
    -------
    synthesizer : GaussianCopulaSynthesizer object or None
        The data generator object

    Examples
    --------
    >>> import pandas as pd
    >>> from fairdo.utils.dataset import data_generator
    >>> data = pd.DataFrame({'age': [39, 50], 'education': ['Bachelors', 'HS-grad'], 'income': ['<=50K', '<=50K']})
    >>> synthesizer = data_generator(data)
    >>> synthetic_data = synthesizer.sample(num_rows=2)
    >>> print(synthetic_data)
         age  education  income
    0  39.0  Bachelors  <=50K
    1  50.0    HS-grad  <=50K
    """
    if not sdv_installed:
        # Inform the user that sdv library is required
        print("The 'sdv' library is required to generate synthetic data.")
        print("Please install it by running: pip install sdv==1.10.0")
        return None

    # Fit the synthesizer to the real data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)

    return synthesizer
