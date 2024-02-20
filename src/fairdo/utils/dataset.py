# Standard library imports
import io
import zipfile

# Related third-party imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from requests import get


def downcast(data):
    """
    Downcast float and integer columns of the given data to save memory.

    Parameters
    ----------
    data: pandas DataFrame

    Returns
    -------
    data: pandas DataFrame
    """
    fcols = data.select_dtypes('float').columns
    icols = data.select_dtypes('integer').columns

    data[fcols] = data[fcols].apply(pd.to_numeric, downcast='float')
    data[icols] = data[icols].apply(pd.to_numeric, downcast='integer')

    return data


def load_data(dataset_str, print_info=True):
    """
    Load the dataset and preprocess it. The preprocessing steps include:
    - Dropping rows with missing values
    - Label encode protected attributes and label
    - One-hot encode all other categorical variables
    - Downcast float and integer columns to save memory

    Parameters
    ----------
    dataset_str: str
        Name of the dataset to load and preprocess  (e.g., 'adult', 'compas', 'bank', 'german').
    Returns
    -------
    df: pandas DataFrame
    label: str
    protected_attributes: list of str
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
        protected_attributes = ['race']

    elif dataset_str == 'compas':
        use_cols = ['race', 'priors_count', 'age_cat', 'c_charge_degree', 'two_year_recid']
        data = pd.read_csv(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            usecols=use_cols)
        print('Data downloaded.')

        # Drop rows with missing values
        data = data.dropna(axis=0, how='any')

        # Label encoding protected_attribute and label
        label = 'two_year_recid'
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
