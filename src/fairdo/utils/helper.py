from itertools import combinations
import numpy as np
# Attempt to import (optional) sdv libraries
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    sdv_installed = True
except ModuleNotFoundError:
    sdv_installed = False


def nunique(a, axis=0):
    """
    Count the number of unique elements in an array along a given axis.
    
    Parameters
    ----------
    a: np.array
        The array to count the number of unique elements.
    axis: int, optional
        The axis along which to count the number of unique elements.
        Default is 0.
    
    Returns
    -------
    np.array
        The number of unique elements along the given axis.
    """
    if a.ndim == 1:
        a = a.reshape(-1,1)
    elif a.ndim > 2:
        raise ValueError('a must be 1D or 2D')
    
    if axis == 0:
        a_s = np.sort(a,axis=0)
        out = a.shape[0]-(a_s[:-1,:] == a_s[1:,:]).sum(axis=0)
    elif axis == 1:
        a_s = np.sort(a,axis=1)
        out = a.shape[1]-(a_s[:,:-1] == a_s[:,1:]).sum(axis=1)
    else:
        raise ValueError('axis must be 0 or 1')
    
    return out

def generate_pairs(lst):
    """
    Generate all possible pairs of elements in a list without repetitions

    Parameters
    ----------
    lst: list or np.array
        list of elements

    Returns
    -------
    list
        list of pairs of elements
    """
    return list(combinations(lst, 2))


def generate_data(data, num_rows=100):
    """
    Generate synthetic data using the sdv library.
    The method used is Gaussian Copula.

    Parameters
    ----------
    data : pd.DataFrame
        The real data to be used to generate synthetic data
    num_rows : int
        The number of rows to generate
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
    data : pd.DataFrame
        The real data to be used to generate synthetic data
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
